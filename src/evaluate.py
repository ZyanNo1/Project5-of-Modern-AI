import os
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import MultiModalDataset, collate_fn
from transform import build_image_transforms, TokenizerWrapper
from model import LateFusionClassifier
from utils import (
    get_device,
    load_checkpoint,
    compute_metrics,
    ensure_dir,
    save_json
)

def _load_run_hparams(checkpoint_path: str) -> dict:
    """Load hyperparameters from checkpoint['args'] or sibling hparams.json."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        return ckpt["args"]

    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    hp_path = os.path.join(run_dir, "hparams.json")
    if os.path.exists(hp_path):
        with open(hp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    if args.output_dir is None:
        run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.output_dir = os.path.join(run_dir, "eval")

    # Load saved split
    split_path = os.path.join(args.split_dir, "test_split.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cannot find internal test split at: {split_path}")

    df_test = pd.read_csv(split_path, dtype={"guid": str})
    print(f"Loaded internal test split: {len(df_test)} samples")

    run_hparams = _load_run_hparams(args.checkpoint)

    clip_model_name = run_hparams.get("clip_model", args.clip_model)
    hidden_dim = int(run_hparams.get("hidden_dim", 512))
    hidden_dim2 = int(run_hparams.get("hidden_dim2", 128))
    dropout = float(run_hparams.get("dropout", 0.3))
    freeze_clip = bool(run_hparams.get("freeze_clip", False))
    fusion = run_hparams.get("fusion", "concat")

    # Tokenizer
    tok_wrapper = TokenizerWrapper(
        model_name_or_path=clip_model_name,
        max_length=args.max_text_len,
        device=device
    )

    # Image transform
    image_transform = build_image_transforms(image_size=args.image_size, train=False)

    # Dataset & DataLoader
    test_dataset = MultiModalDataset(
        df=df_test,
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
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Load model
    model = LateFusionClassifier(
        clip_model_name=clip_model_name,
        embed_dim=None,
        hidden_dims=(hidden_dim, hidden_dim2),
        dropout=dropout,
        freeze_clip=freeze_clip,
        num_classes=3,
        fusion=fusion
    )
    model.to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    # Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().tolist()

            logits = model(image=images, input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, labels=[0, 1, 2])
    print("\n=== Internal Test Evaluation ===")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Precision:     {metrics['precision_macro']:.4f}")
    print(f"Recall:        {metrics['recall_macro']:.4f}")
    print(f"Per-class F1:  {metrics['per_class_f1']}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    # Save results
    ensure_dir(args.output_dir)
    save_json(metrics, os.path.join(args.output_dir, "internal_test_metrics.json"))
    print(f"\nSaved metrics to {args.output_dir}/internal_test_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on internal test split")
    parser.add_argument("--split_dir", type=str, required=True,
                        help="Directory containing test_split.csv saved during training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save metrics. Default: <checkpoint_dir>/eval")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
