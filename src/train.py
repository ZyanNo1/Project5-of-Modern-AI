# src/train.py
import os
import argparse
import json
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import CLIPTokenizerFast

from datasets import get_dataloaders, LABEL2ID, ID2LABEL
from transform import build_image_transforms, TokenizerWrapper
from model import LateFusionClassifier
from utils import set_seed, get_device, setup_logger, compute_metrics, save_checkpoint, plot_training_curves, append_metrics_to_csv, ensure_dir

# -------------------------
# Training / Validation Step
# -------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        # forward
        logits = model(image=images, input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, labels=list(range(len(LABEL2ID))))
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(image=images, input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, labels=list(range(len(LABEL2ID))))
    return avg_loss, metrics


# -------------------------
# Main training loop
# -------------------------
def main(args):
    # prepare output dirs & logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    ensure_dir(out_dir)
    logger = setup_logger(log_file=os.path.join(out_dir, "train.log"))
    logger.info("Output dir: %s", out_dir)

    with open(os.path.join(out_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    logger = setup_logger(log_file=os.path.join(out_dir, "train.log"))
    logger.info("Output dir: %s", out_dir)
    logger.info("Hyperparameters saved to %s", os.path.join(out_dir, "hparams.json"))

    # reproducibility & device
    set_seed(args.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    # tokenizer wrapper
    tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_model)
    tok_wrapper = TokenizerWrapper(tokenizer_obj=tokenizer, max_length=args.max_text_len, device=device)

    # transforms
    image_transform_train = build_image_transforms(image_size=args.image_size, train=True)
    image_transform_eval = build_image_transforms(image_size=args.image_size, train=False)

    # dataloaders
    train_loader, val_loader, test_loader, splits = get_dataloaders(
        train_txt_path=args.train_txt,
        data_dir=args.data_dir,
        tokenizer=tok_wrapper.tokenizer,  # datasets expects HF tokenizer directly
        image_transform_train=image_transform_train,
        image_transform_eval=image_transform_eval,
        batch_size=args.batch_size,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers
    )
    train_df, val_df, test_df = splits
    logger.info("Dataset sizes - train: %d, val: %d, test(split): %d", len(train_df), len(val_df), len(test_df))
    
    split_dir = os.path.join(out_dir, "splits") 
    ensure_dir(split_dir) 
    train_df.to_csv(os.path.join(split_dir, "train_split.csv"), index=False) 
    val_df.to_csv(os.path.join(split_dir, "val_split.csv"), index=False) 
    test_df.to_csv(os.path.join(split_dir, "test_split.csv"), index=False) 
    logger.info("Saved dataset splits to %s", split_dir)
    

    # model
    model = LateFusionClassifier(clip_model_name=args.clip_model,
                                 embed_dim=None,
                                 hidden_dims=(args.hidden_dim, args.hidden_dim2),
                                 dropout=args.dropout,
                                 freeze_clip=args.freeze_clip,
                                 num_classes=len(LABEL2ID))
    model.to(device)
    logger.info("Model initialized. Freeze CLIP: %s", args.freeze_clip)


    # loss, optimizer, scheduler
    if args.class_weights:
        counts = train_df["tag"].map(LABEL2ID).value_counts().sort_index().values.astype(float)

        inv = counts.sum() / (counts + 1e-12)  # inverse frequency
        inv = inv / inv.mean()                 # normalize: mean weight = 1
        inv = inv ** 0.5                       # soften extremes (sqrt)

        weights = torch.tensor(inv, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info("Using class weights for loss (smoothed): %s", weights.tolist())
    else:
        criterion = nn.CrossEntropyLoss()


    # optimizer: only train parameters that require grad
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head, weight_decay=args.weight_decay)

    # scheduler: simple ReduceLROnPlateau or CosineAnnealingWarmRestarts can be used; use ReduceLROnPlateau here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # training loop variables
    best_val_macro = -1.0
    best_epoch = -1
    history = defaultdict(list)
    patience_counter = 0

    # training epochs
    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # logging
        logger.info("Train loss: %.4f | Train macro_f1: %.4f | Val loss: %.4f | Val macro_f1: %.4f",
                    train_loss, train_metrics["macro_f1"], val_loss, val_metrics["macro_f1"])

        # record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        # scheduler step (ReduceLROnPlateau uses metric)
        scheduler.step(val_metrics["macro_f1"])

        # save checkpoint if improved
        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(out_dir, "best_checkpoint.pth")
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_macro_f1": val_metrics["macro_f1"],
                "args": vars(args)
            }, checkpoint_dir=out_dir, filename="best_checkpoint.pth")
            logger.info("Saved new best checkpoint (epoch %d, val_macro_f1 %.4f)", epoch, val_metrics["macro_f1"])
        else:
            patience_counter += 1
            logger.info("No improvement. Patience %d/%d", patience_counter, args.early_stop_patience)

        # early stopping
        if patience_counter >= args.early_stop_patience:
            logger.info("Early stopping triggered. Best epoch: %d (val_macro_f1=%.4f)", best_epoch, best_val_macro)
            break

    # finalize: save history and plot curves
    plot_path = os.path.join(out_dir, "training_curves.png")
    plot_training_curves(history, plot_path)
    logger.info("Saved training curves to %s", plot_path)

    # save metrics CSV
    metrics_csv = os.path.join(out_dir, "val_metrics.csv")
    append_metrics_to_csv(metrics_csv, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_split_size": len(test_df),
        "seed": args.seed
    }, fieldnames=["best_epoch", "best_val_macro_f1", "train_size", "val_size", "test_split_size", "seed"])
    logger.info("Saved summary metrics to %s", metrics_csv)

    logger.info("Training finished. Best epoch: %d, Best val macro_f1: %.4f", best_epoch, best_val_macro)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP late-fusion classifier")
    parser.add_argument("--train_txt", type=str, default="./data/train.txt", help="Path to train.txt (guid,tag)")
    parser.add_argument("--data_dir", type=str, default="./data/data", help="Directory with guid.jpg and guid.txt files")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="HF CLIP model id")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and logs")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for CLIP transforms")
    parser.add_argument("--max_text_len", type=int, default=77, help="Max token length for text")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for classification head")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dim for head first layer")
    parser.add_argument("--hidden_dim2", type=int, default=128, help="Hidden dim for head second layer")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout in head")
    parser.add_argument("--freeze_clip", action="store_true", help="Freeze CLIP backbone initially")
    parser.add_argument("--class_weights", action="store_true", help="Use class weights in loss")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test split fraction (internal split)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Early stopping patience (epochs)")
    args = parser.parse_args()

    main(args)
