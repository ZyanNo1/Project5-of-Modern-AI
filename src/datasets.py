# src/datasets.py
import os
import random
import re
from typing import List, Tuple, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Config / Label mapping
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Text cleaning util

def clean_text(text: str) -> str:
    """Basic social-text cleaning: remove urls, mentions, split hashtags, normalize whitespace."""
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("\n", " ")
    # remove urls
    text = re.sub(r"http\S+", "", text)
    # remove @mentions
    text = re.sub(r"@\w+", "", text)
    # split hashtags: #HappyDay -> HappyDay (you may further split camelcase if desired)
    text = re.sub(r"#(\w+)", r"\1", text)
    # collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text



# Data loading & splitting
def read_metadata(train_txt_path: str) -> pd.DataFrame:
    """
    Read train.txt which has columns: guid,tag
    Returns DataFrame with columns ['guid','tag'].
    """
    df = pd.read_csv(train_txt_path, dtype={"guid": str})
    if "tag" not in df.columns:
        raise ValueError("train.txt must contain a 'tag' column.")
    df = df[["guid", "tag"]].dropna().reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame,
                     train_size: float = 0.8,
                     val_size: float = 0.1,
                     test_size: float = 0.1,
                     seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train/val/test by 'tag'.
    Returns (train_df, val_df, test_df).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    # first split train vs temp
    train_df, temp_df = train_test_split(df,
                                         stratify=df["tag"],
                                         train_size=train_size,
                                         random_state=seed)
    if test_size == 0.0:
        # Only split train/val, no test
        val_df = temp_df
        test_df = pd.DataFrame(columns=df.columns)
    else:
        # split temp into val and test
        relative_val = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df,
                                           stratify=temp_df["tag"],
                                           train_size=relative_val,
                                           random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def check_data_consistency(df: pd.DataFrame, data_dir: str, img_exts: List[str] = [".jpg", ".png", ".jpeg"]) -> pd.DataFrame:
    """
    Verify that for each guid there exists an image file and a text file.
    Returns a filtered DataFrame containing only consistent rows.
    """
    ok_rows = []
    missing = []
    for _, row in df.iterrows():
        guid = str(row["guid"])
        img_found = False
        for ext in img_exts:
            if os.path.exists(os.path.join(data_dir, guid + ext)):
                img_found = True
                break
        txt_path = os.path.join(data_dir, guid + ".txt")
        if img_found and os.path.exists(txt_path):
            ok_rows.append(row)
        else:
            missing.append(guid)
    if missing:
        print(f"[Warning] {len(missing)} guids missing image or text. Example missing: {missing[:5]}")
    return pd.DataFrame(ok_rows).reset_index(drop=True)



# Dataset classes
class MultiModalDataset(Dataset):
    """
    Dataset for train/val. Expects:
      - data_dir/{guid}.jpg (or .png)
      - data_dir/{guid}.txt
    tokenizer: a callable that accepts text and returns dict with input_ids & attention_mask (torch tensors)
    image_transform: torchvision transforms to apply to PIL image
    """
    def __init__(self,
                 df: pd.DataFrame,
                 data_dir: str,
                 tokenizer,
                 image_transform,
                 label2id: dict = LABEL2ID,
                 max_text_len: int = 77):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.label2id = label2id
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def _load_image(self, guid: str) -> Image.Image:
        # try common extensions
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(self.data_dir, guid + ext)
            if os.path.exists(p):
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"No image found for guid {guid} in {self.data_dir}")

    def _load_text(self, guid: str) -> str:
        p = os.path.join(self.data_dir, guid + ".txt")
        if not os.path.exists(p):
            return ""
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(p, "r", encoding="gbk") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(p, "r", encoding="latin1") as f:
                    text = f.read()
        return clean_text(text)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = str(row["guid"])
        label_str = row["tag"]
        label = self.label2id[label_str]

        # image
        image = self._load_image(guid)
        if self.image_transform is not None:
            image = self.image_transform(image)

        # text
        text = self._load_text(guid)
        # tokenizer should return tensors; we squeeze batch dim later in collate
        text_inputs = self.tokenizer(text,
                                     padding="max_length",
                                     truncation=True,
                                     max_length=self.max_text_len,
                                     return_tensors="pt")
        item = {
            "guid": guid,
            "image": image,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
        return item


class TestDataset(Dataset):
    """
    Dataset for test_without_label.txt (tag is null). Returns same fields except label.
    """
    def __init__(self, guid_list: List[str], data_dir: str, tokenizer, image_transform, max_text_len: int = 77):
        self.guid_list = [str(g) for g in guid_list]
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.guid_list)

    def _load_image(self, guid: str) -> Image.Image:
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(self.data_dir, guid + ext)
            if os.path.exists(p):
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"No image found for guid {guid} in {self.data_dir}")

    def _load_text(self, guid: str) -> str:
        p = os.path.join(self.data_dir, guid + ".txt")
        if not os.path.exists(p):
            return ""
        
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(p, "r", encoding=encoding) as f:
                    text = f.read()
                return clean_text(text)
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        print(f"Warning: Cannot decode {p} with any encoding, using empty text")
        return ""

    def __getitem__(self, idx):
        guid = self.guid_list[idx]
        image = self._load_image(guid)
        if self.image_transform is not None:
            image = self.image_transform(image)
        text = self._load_text(guid)
        text_inputs = self.tokenizer(text,
                                     padding="max_length",
                                     truncation=True,
                                     max_length=self.max_text_len,
                                     return_tensors="pt")
        return {
            "guid": guid,
            "image": image,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0)
        }


# Collate fn for DataLoader
def collate_fn(batch):
    """
    Batch is a list of dicts from MultiModalDataset.
    This collate stacks tensors and returns a dict.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    guids = [b["guid"] for b in batch]
    return {
        "guids": guids,
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels
    }


def collate_fn_infer(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    guids = [b["guid"] for b in batch]
    return {
        "guids": guids,
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_dataloaders(train_txt_path: str,
                    data_dir: str,
                    tokenizer,
                    image_transform_train,
                    image_transform_eval,
                    batch_size: int = 16,
                    seed: int = 42,
                    val_size: float = 0.2,
                    test_size: float = 0.0,
                    num_workers: int = 4):
    """
    Read metadata, split, check consistency, and return train/val/test DataLoaders.
    """
    df = read_metadata(train_txt_path)
    train_df, val_df, test_df = stratified_split(df, train_size=1 - val_size - test_size,
                                                 val_size=val_size, test_size=test_size, seed=seed)
    # check files exist
    train_df = check_data_consistency(train_df, data_dir)
    val_df = check_data_consistency(val_df, data_dir)
    test_df = check_data_consistency(test_df, data_dir)

    train_ds = MultiModalDataset(train_df, data_dir, tokenizer, image_transform_train)
    val_ds = MultiModalDataset(val_df, data_dir, tokenizer, image_transform_eval)
    test_ds = MultiModalDataset(test_df, data_dir, tokenizer, image_transform_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)



if __name__ == "__main__":
    # Test example (not executed during import)
    from torchvision import transforms
    from transformers import CLIPTokenizerFast

    data_dir = "./data/data"
    train_txt = "./data/train.txt"
    seed = 42

    # image transforms (example)
    image_transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        # normalize with CLIP mean/std if using CLIP; replace with actual values in training script
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    image_transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # tokenizer: pass the CLIP tokenizer or other tokenizer consistent with your text encoder
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    train_loader, val_loader, test_loader, splits = get_dataloaders(
        train_txt_path=train_txt,
        data_dir=data_dir,
        tokenizer=tokenizer,
        image_transform_train=image_transform_train,
        image_transform_eval=image_transform_eval,
        batch_size=16,
        seed=seed,
        num_workers=4
    )

    print("Train/Val/Test sizes:", [len(s) for s in splits])
