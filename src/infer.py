import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import TestDataset, collate_fn_infer
from transform import build_image_transforms, TokenizerWrapper
from model import LateFusionClassifier
from utils import get_device, ensure_dir

def load_test_guids(test_txt_path):
    """Load test guids with encoding fallback (utf-8 -> gbk -> latin-1)"""
    for encoding in ['utf-8', 'gbk', 'latin-1']:
        try:
            df = pd.read_csv(test_txt_path, dtype={"guid": str}, encoding=encoding)
            if "guid" not in df.columns:
                raise ValueError("test_without_label.txt must contain a 'guid' column")
            print(f"Successfully loaded {test_txt_path} with encoding: {encoding}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"Cannot decode {test_txt_path} with any known encoding (utf-8/gbk/latin-1)")

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # Load test guids
    df_test = load_test_guids(args.test_txt)
    guid_list = df_test["guid"].tolist()
    print(f"Loaded {len(guid_list)} test samples")

    # Load checkpoint and extract training hyperparameters
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    if "args" in ckpt:
        train_args = ckpt["args"]
        hidden_dim = train_args.get("hidden_dim", 512)
        hidden_dim2 = train_args.get("hidden_dim2", 128)
        dropout = train_args.get("dropout", 0.5)
        fusion = train_args.get("fusion", "concat")
        clip_model = train_args.get("clip_model", args.clip_model)
        print(f"Restored hyperparameters: hidden_dim={hidden_dim}, hidden_dim2={hidden_dim2}, dropout={dropout}, fusion={fusion}")
    else:
        print("Warning: checkpoint missing 'args' field, using defaults")
        hidden_dim = 512
        hidden_dim2 = 128
        dropout = 0.5
        fusion = "concat"
        clip_model = args.clip_model

    # Tokenizer
    tok_wrapper = TokenizerWrapper(
        model_name_or_path=clip_model,
        max_length=args.max_text_len,
        device=device
    )

    # Image transform (eval mode)
    image_transform = build_image_transforms(image_size=args.image_size, train=False)

    # Build dataset & dataloader
    test_dataset = TestDataset(
        guid_list=guid_list,
        data_dir=args.data_dir,
        tokenizer=tok_wrapper.tokenizer,
        image_transform=image_transform,
        max_text_len=args.max_text_len
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_infer,
        pin_memory=True
    )

    # Build model with restored hyperparameters
    model = LateFusionClassifier(
        clip_model_name=clip_model,
        hidden_dims=(hidden_dim, hidden_dim2),
        dropout=dropout,
        freeze_clip=False,
        num_classes=3,
        fusion=fusion
    )
    model.to(device)

    # Load model weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Label mapping
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    # Inference loop
    all_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(image=images, input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

    # Build submission DataFrame
    df_out = df_test.copy()
    df_out["tag"] = [id2label[p] for p in all_preds]

    # Save output
    ensure_dir(os.path.dirname(args.output_path) or ".")
    df_out.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")
    print(f"Prediction distribution: {df_out['tag'].value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for CLIP late-fusion model")
    parser.add_argument("--test_txt", type=str, default="test_without_label.txt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_path", type=str, default="submission.txt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)