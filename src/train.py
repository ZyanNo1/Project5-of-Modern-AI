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

def _build_optimizer(model: LateFusionClassifier, lr_head: float, lr_clip: float, weight_decay: float):
    """Build AdamW with param groups: clip vs head (robust grouping by object id)."""
    clip_param_ids = {id(p) for p in model.clip.parameters()}
    clip_params = [p for p in model.clip.parameters() if p.requires_grad]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in clip_param_ids]

    param_groups = []
    if clip_params:
        param_groups.append({"params": clip_params, "lr": lr_clip})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})

    return AdamW(param_groups, weight_decay=weight_decay)

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

    
    # ---- stage1 optimizer/scheduler ----
    optimizer = _build_optimizer(model, lr_head=args.lr_head, lr_clip=args.lr_clip, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_macro = -1.0
    best_epoch = -1
    history = defaultdict(list)
    patience_counter = 0

    def _run_epochs(start_epoch: int, max_epoch: int, stage_name: str):
        nonlocal best_val_macro, best_epoch, patience_counter, optimizer, scheduler

        for epoch in range(start_epoch, max_epoch + 1):
            logger.info("[%s] Epoch %d/%d", stage_name, epoch, max_epoch)

            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            logger.info("[%s] Train loss: %.4f | Train macro_f1: %.4f | Val loss: %.4f | Val macro_f1: %.4f",
                        stage_name, train_loss, train_metrics["macro_f1"], val_loss, val_metrics["macro_f1"])

            history[f"{stage_name}_train_loss"].append(train_loss)
            history[f"{stage_name}_val_loss"].append(val_loss)
            history[f"{stage_name}_train_macro_f1"].append(train_metrics["macro_f1"])
            history[f"{stage_name}_val_macro_f1"].append(val_metrics["macro_f1"])

            scheduler.step(val_metrics["macro_f1"])

            if val_metrics["macro_f1"] > best_val_macro:
                best_val_macro = val_metrics["macro_f1"]
                best_epoch = epoch
                patience_counter = 0

                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_macro_f1": val_metrics["macro_f1"],
                    "args": vars(args)
                }, checkpoint_dir=out_dir, filename="best_checkpoint.pth")
                logger.info("[%s] Saved new best checkpoint (epoch %d, val_macro_f1 %.4f)",
                            stage_name, epoch, val_metrics["macro_f1"])
            else:
                patience_counter += 1
                logger.info("[%s] No improvement. Patience %d/%d", stage_name, patience_counter, args.early_stop_patience)

            if patience_counter >= args.early_stop_patience:
                logger.info("[%s] Early stopping triggered. Best epoch so far: %d (val_macro_f1=%.4f)",
                            stage_name, best_epoch, best_val_macro)
                break

    # ----------------
    # Stage 1 (freeze)
    # ----------------
    if args.two_stage:
        logger.info("Two-stage training enabled.")
        logger.info("Stage1: freeze CLIP, train head only for %d epoch(s).", args.stage1_epochs)
        _run_epochs(start_epoch=1, max_epoch=args.stage1_epochs, stage_name="stage1")

        # Save stage1 best separately (optional convenience)
        # (best_checkpoint already saved; we copy it for clarity)
        stage1_best = os.path.join(out_dir, "stage1_best_checkpoint.pth")
        try:
            import shutil
            shutil.copyfile(os.path.join(out_dir, "best_checkpoint.pth"), stage1_best)
            logger.info("Copied stage1 best checkpoint to %s", stage1_best)
        except Exception as e:
            logger.warning("Could not copy stage1 best checkpoint: %s", str(e))

        # ----------------
        # Stage 2 (unfreeze)
        # ----------------
        # Reset early-stopping counter for stage2 to avoid stopping immediately
        patience_counter = 0

        # Load best weights before finetuning further
        ckpt = torch.load(os.path.join(out_dir, "best_checkpoint.pth"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint before Stage2 finetune.")

        # Ensure CLIP is frozen then selectively unfreeze
        model.freeze_clip()
        if args.unfreeze_text:
            model.unfreeze_text_encoder(n_last_layers=args.unfreeze_text_layers)
            logger.info("Stage2: Unfroze CLIP text encoder last %d layer(s).", args.unfreeze_text_layers)
        if args.unfreeze_vision:
            model.unfreeze_vision_encoder(n_last_layers=args.unfreeze_vision_layers)
            logger.info("Stage2: Unfroze CLIP vision encoder last %d layer(s).", args.unfreeze_vision_layers)

        # Rebuild optimizer/scheduler for new trainable set
        optimizer = _build_optimizer(model, lr_head=args.lr_head, lr_clip=args.lr_clip, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

        logger.info("Stage2: finetune for %d epoch(s). (total epochs=%d)",
                    args.stage2_epochs, args.stage1_epochs + args.stage2_epochs)
        _run_epochs(
            start_epoch=args.stage1_epochs + 1,
            max_epoch=args.stage1_epochs + args.stage2_epochs,
            stage_name="stage2"
        )
    else:
        # original single-stage behavior
        if args.unfreeze_text:
            model.unfreeze_text_encoder(n_last_layers=args.unfreeze_text_layers)
            logger.info("Unfroze CLIP text encoder last %d layer(s).", args.unfreeze_text_layers)
        if args.unfreeze_vision:
            model.unfreeze_vision_encoder(n_last_layers=args.unfreeze_vision_layers)
            logger.info("Unfroze CLIP vision encoder last %d layer(s).", args.unfreeze_vision_layers)

        _run_epochs(start_epoch=1, max_epoch=args.epochs, stage_name="train")

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
    parser.add_argument("--lr_clip", type=float, default=1e-5, help="Learning rate for CLIP backbone (when unfrozen)")
    parser.add_argument("--unfreeze_text", action="store_true", help="Unfreeze CLIP text encoder for light finetune")
    parser.add_argument("--unfreeze_vision", action="store_true", help="Unfreeze CLIP vision encoder for light finetune")
    parser.add_argument("--unfreeze_text_layers", type=int, default=1,
                        help="How many last text transformer layers to unfreeze (ignored if --unfreeze_text not set)")
    parser.add_argument("--unfreeze_vision_layers", type=int, default=1,
                        help="How many last vision transformer layers to unfreeze (ignored if --unfreeze_vision not set)")
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
    parser.add_argument("--two_stage", action="store_true", help="Enable two-stage training: stage1 freeze CLIP, stage2 light finetune.")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Epochs for stage1 (freeze CLIP, train head).")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="Epochs for stage2 (light finetune from best stage1).")
    args = parser.parse_args()

    main(args)
