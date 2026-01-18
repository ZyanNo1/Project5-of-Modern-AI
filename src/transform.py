"""
Image & text transforms and tokenizer wrapper for CLIP-based late-fusion pipeline.

Provides:
- build_image_transforms(image_size, train=True)
- TokenizerWrapper: unify CLIP / BERT tokenizers to return torch tensors
- generate_caption_placeholder(image): BLIP caption placeholder (to be replaced by real BLIP call)
- tta_transforms(image_size): simple TTA set (center + horizontal flip)
"""

from typing import Callable, Dict, Optional, Tuple, List
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# If using Hugging Face CLIP tokenizer
try:
    from transformers import CLIPTokenizerFast, AutoTokenizer
except Exception:
    CLIPTokenizerFast = None
    AutoTokenizer = None

# CLIP default normalization values (openai/clip-vit-base-patch32)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_image_transforms(image_size: int = 224, train: bool = True) -> Callable:
    """
    Return torchvision transform pipeline for CLIP-compatible images.
    - image_size: target short side / crop size (CLIP typically uses 224 or 336)
    - train: if True, include augmentations
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # slightly larger for random crop
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
    return transform


def tta_transforms(image_size: int = 224) -> List[Callable]:
    """
    Return a small list of deterministic transforms for simple TTA (center + hflip).
    Use by applying each transform and averaging logits/probs.
    """
    base = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    hflip = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),  # deterministic flip
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    return [base, hflip]


class TokenizerWrapper:
    """
    Wrapper to unify tokenizer interface for CLIP or other HF tokenizers.

    Usage:
      tok = TokenizerWrapper("openai/clip-vit-base-patch32", max_length=77)
      enc = tok.encode("some text")  # returns dict of tensors: input_ids, attention_mask
    """
    def __init__(self, model_name_or_path: Optional[str] = None, tokenizer_obj: Optional[object] = None,
                 max_length: int = 77, device: Optional[torch.device] = None):
        self.max_length = max_length
        self.device = device or torch.device("cpu")
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            if model_name_or_path is None:
                raise ValueError("Provide model_name_or_path or tokenizer_obj")
            # prefer CLIP tokenizer if available
            if CLIPTokenizerFast is not None and "clip" in model_name_or_path:
                self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name_or_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single text string and return tensors on configured device.
        Returns: {"input_ids": Tensor( L ), "attention_mask": Tensor( L )}
        """
        # tokenizer returns batch dim; squeeze later in Dataset
        enc = self.tokenizer(text,
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_length,
                             return_tensors="pt")
        enc = {k: v.squeeze(0).to(self.device) for k, v in enc.items()}
        return enc

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(texts,
                             padding="longest",
                             truncation=True,
                             max_length=self.max_length,
                             return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return enc


# -------------------------
# Text cleaning helper (re-export)
# -------------------------
import re


def clean_text(text: Optional[str]) -> str:
    """
    Basic social-text cleaning: remove urls, mentions, split hashtags, normalize whitespace.
    Keep punctuation like ! and ?.
    """
    if text is None:
        return ""
    text = text.replace("\n", " ").strip()
    # remove urls
    text = re.sub(r"http\S+", "", text)
    # remove @mentions
    text = re.sub(r"@\w+", "", text)
    # split hashtags: #HappyDay -> HappyDay
    text = re.sub(r"#(\w+)", r"\1", text)
    # collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


# -------------------------
# BLIP caption placeholder
# -------------------------
def generate_caption_placeholder(image: Image.Image) -> str:
    """
    Placeholder for BLIP caption generation.
    Replace this function with a real BLIP call when integrating BLIP.
    For now returns a short heuristic caption (size + mean color) to avoid breaking pipeline.
    """
    try:
        img = image.convert("RGB").resize((64, 64))
        arr = np.array(img).astype(np.float32) / 255.0
        mean_color = arr.mean(axis=(0, 1))
        caption = f"image with dominant color r{mean_color[0]:.2f} g{mean_color[1]:.2f} b{mean_color[2]:.2f}"
        return caption
    except Exception:
        return "photo"


# -------------------------
# Utility: apply tokenizer in collate if needed
# -------------------------
def collate_tokenize_texts(texts: List[str], tokenizer_wrapper: TokenizerWrapper) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of texts into batched tensors (input_ids, attention_mask).
    Useful if Dataset returns raw text and you want to tokenize in collate_fn to speed up IO.
    """
    enc = tokenizer_wrapper.batch_encode(texts)
    return enc


# -------------------------
# Example quick test (not executed on import)
# -------------------------
if __name__ == "__main__":
    # quick smoke test for transforms
    img_size = 224
    train_tf = build_image_transforms(img_size, train=True)
    eval_tf = build_image_transforms(img_size, train=False)
    tta = tta_transforms(img_size)
    print("Transforms ready. TTA variants:", len(tta))
