"""
Utility functions for training and evaluation.

Provides:
- set_seed(seed): reproducible runs
- get_device(): returns torch.device
- Metric functions: compute_metrics(y_true, y_pred)
- Checkpoint helpers: save_checkpoint, load_checkpoint
- Simple AverageMeter for tracking losses/metrics
- Simple logger setup and CSV log writer
- plot_training_curves (matplotlib) to save loss/metric curves
"""

import os
import random
import json
import csv
import logging
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


# -------------------------
# Reproducibility & device
# -------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for python, numpy and torch for reproducibility.
    If deterministic=True, also set cudnn to deterministic mode (may slow down).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return torch.device: cuda if available and prefer_gpu True, else cpu.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Metrics
# -------------------------
def compute_metrics(y_true: List[int], y_pred: List[int], labels: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Compute classification metrics.
    Returns dict with: accuracy, macro_f1, per_class_f1 (dict), precision_macro, recall_macro, confusion_matrix (ndarray).
    y_true, y_pred: lists or 1D arrays of ints.
    labels: optional list of label indices to pass to sklearn (ensures consistent ordering).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(list(set(y_true.tolist() + y_pred.tolist())))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm
    }
    return metrics


# -------------------------
# Checkpoint helpers
# -------------------------
def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str, filename: str = "checkpoint.pth"):
    """
    Save training checkpoint.
    state: dict containing at least {'epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict' (opt)}.
    checkpoint_dir: directory to save into (created if not exists).
    filename: file name for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(path: str, model: Optional[nn.Module] = None, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load checkpoint and optionally restore model/optimizer/scheduler states.
    Returns the checkpoint dict.
    If model/optimizer/scheduler provided, their states will be loaded in-place.
    """
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            # scheduler state may be incompatible across versions; ignore if fails
            pass
    return checkpoint


# -------------------------
# AverageMeter for tracking
# -------------------------
class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss/metric during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


# -------------------------
# Simple logger & CSV writer
# -------------------------
def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger that prints to stdout and optionally writes to a file.
    Returns the logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def append_metrics_to_csv(csv_path: str, row: Dict[str, Any], fieldnames: Optional[List[str]] = None):
    """
    Append a row (dict) to a CSV file. If file doesn't exist, create it and write header.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    if fieldnames is None:
        fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -------------------------
# Plotting training curves
# -------------------------
def plot_training_curves(history: Dict[str, List[float]], out_path: str):
    """
    history: dict with keys like 'train_loss','val_loss','train_macro_f1','val_macro_f1' mapping to lists per epoch.
    Saves a PNG with loss and metric curves side by side.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    epochs = range(1, len(next(iter(history.values()))) + 1)

    plt.figure(figsize=(10, 4))

    # Loss subplot
    plt.subplot(1, 2, 1)
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="train_loss", marker='o')
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="val_loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Metric subplot (Macro-F1)
    plt.subplot(1, 2, 2)
    if "train_macro_f1" in history:
        plt.plot(epochs, history["train_macro_f1"], label="train_macro_f1", marker='o')
    if "val_macro_f1" in history:
        plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------
# Small helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    def _default(o: Any):
        # numpy
        try:
            import numpy as np  # local import to avoid hard dependency
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except Exception:
            pass
        # torch
        try:
            import torch
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
        except Exception:
            pass

        # fallback: let json raise TypeError for unknown objects
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Example usage (not executed on import)
# -------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    logger = setup_logger()
    logger.info(f"Using device: {device}")

    # fake predictions
    y_true = [0, 1, 2, 1, 0, 2, 2]
    y_pred = [0, 1, 1, 1, 0, 2, 2]
    metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
    logger.info("Metrics example: %s", metrics)
