# src/model.py
"""
CLIP late-fusion classifier.

"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPConfig


class LateFusionClassifier(nn.Module):
    """
    Late-fusion classifier using CLIP image/text embeddings.

    Args:
      clip_model_name: pretrained CLIP model id (Hugging Face) or local path.
      embed_dim: embedding dimension of CLIP outputs (if None, inferred from model).
      hidden_dims: list of hidden dims for MLP head (e.g., [512, 128]).
      dropout: dropout probability in head.
      freeze_clip: if True, freeze CLIP backbone parameters initially.
      fusion: fusion type for image/text embeddings: "concat" or "gated".
             - concat: x = [img; txt], head input dim = 2*D
             - gated:  x = gate * txt + (1-gate) * img, head input dim = D
    """

    def __init__(self,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 embed_dim: Optional[int] = None,
                 hidden_dims: Tuple[int, ...] = (512, 128),
                 dropout: float = 0.3,
                 freeze_clip: bool = True,
                 num_classes: int = 3,
                 fusion: str = "concat"):
        super().__init__()

        self.fusion = fusion.lower().strip()

        # Load CLIP model (image + text encoders)
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        # Determine embedding dim
        if embed_dim is None:
            # CLIPModel config has projection dims
            cfg: CLIPConfig = self.clip.config
            # text_config and vision_config may differ; CLIPModel projects to text_config.projection_dim
            # Use text_projection if available, else vision_projection
            embed_dim = getattr(cfg, "text_config", None) and getattr(cfg.text_config, "projection_dim", None)
            if embed_dim is None:
                # fallback to vision projection dim
                embed_dim = getattr(cfg, "vision_config", None) and getattr(cfg.vision_config, "projection_dim", None)
            if embed_dim is None:
                # last resort: use hidden_size of text model
                embed_dim = getattr(cfg, "text_config", None) and getattr(cfg.text_config, "hidden_size", 512)
            if embed_dim is None:
                embed_dim = 512

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Freeze CLIP if requested
        if freeze_clip:
            self.freeze_clip()

        # Gated fusion
        self.image_norm = nn.LayerNorm(self.embed_dim)
        self.text_norm = nn.LayerNorm(self.embed_dim)

        if self.fusion == "gated":
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.embed_dim * 2, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            head_input_dim = self.embed_dim
        elif self.fusion == "concat":
            self.gate_mlp = None
            head_input_dim = self.embed_dim * 2
        else:
            raise ValueError(f"Unknown fusion type: {fusion}. Use 'concat' or 'gated'.")

        # Build classification head
        layers = []
        in_dim = head_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Optional layernorm on embeddings (stabilizes training)
        self.image_norm = nn.LayerNorm(self.embed_dim)
        self.text_norm = nn.LayerNorm(self.embed_dim)

    def freeze_clip(self):
        """Freeze all parameters of the CLIP backbone."""
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_clip(self):
        """Unfreeze all CLIP parameters."""
        for p in self.clip.parameters():
            p.requires_grad = True

    def unfreeze_text_encoder(self, n_last_layers: int = 0) -> None:
        """
        Unfreeze text encoder parameters.
        If n_last_layers > 0, only unfreeze last N transformer layers (and final layer norm).
        If n_last_layers <= 0, unfreeze the whole text encoder.
        """
        if n_last_layers <= 0:
            for name, p in self.clip.named_parameters():
                if name.startswith("text_model"):
                    p.requires_grad = True
            return

        # freeze all text encoder params first
        for name, p in self.clip.named_parameters():
            if name.startswith("text_model"):
                p.requires_grad = False

        # unfreeze last N layers
        layers = self.clip.text_model.encoder.layers
        n = min(n_last_layers, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # also unfreeze final layer norm (helps adaptation)
        if hasattr(self.clip.text_model, "final_layer_norm"):
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True

    def unfreeze_vision_encoder(self, n_last_layers: int = 0) -> None:
        """
        Unfreeze vision encoder parameters.
        If n_last_layers > 0, only unfreeze last N transformer layers (and post layer norm).
        If n_last_layers <= 0, unfreeze the whole vision encoder.
        """
        if n_last_layers <= 0:
            for name, p in self.clip.named_parameters():
                if name.startswith("vision_model"):
                    p.requires_grad = True
            return

        # freeze all vision encoder params first
        for name, p in self.clip.named_parameters():
            if name.startswith("vision_model"):
                p.requires_grad = False

        # unfreeze last N layers
        layers = self.clip.vision_model.encoder.layers
        n = min(n_last_layers, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # also unfreeze post layer norm (helps adaptation)
        if hasattr(self.clip.vision_model, "post_layernorm"):
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True

    def get_image_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get CLIP image embedding.
        Input: pixel_values shape (B, C, H, W) already normalized as CLIP expects.
        Returns: (B, embed_dim)
        """
        # CLIPModel.get_image_features handles projection and pooling
        img_emb = self.clip.get_image_features(pixel_values=pixel_values)
        # normalize / optional layernorm
        img_emb = self.image_norm(img_emb)
        return img_emb

    def get_text_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get CLIP text embedding.
        Inputs: input_ids, attention_mask (B, L)
        Returns: (B, embed_dim)
        """
        txt_emb = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        txt_emb = self.text_norm(txt_emb)
        return txt_emb

    def forward(self,
                image: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Provide image and text inputs. For text-only or image-only modes, pass the other as None.
        If return_embeddings=True, returns (logits, image_emb, text_emb).
        Otherwise returns logits.

        Note: If CLIP backbone is frozen, get_image_features/get_text_features are called under no_grad
        to save memory; but we still return embeddings as tensors (detached).
        """
        # Validate inputs
        if image is None and (input_ids is None or attention_mask is None):
            raise ValueError("At least one of image or text inputs must be provided.")

        # Compute embeddings
        # If backbone frozen, compute under torch.no_grad to save memory
        if not any(p.requires_grad for p in self.clip.parameters()):
            # backbone frozen
            with torch.no_grad():
                image_emb = self.get_image_embedding(image) if image is not None else None
                text_emb = self.get_text_embedding(input_ids, attention_mask) if input_ids is not None else None
        else:
            image_emb = self.get_image_embedding(image) if image is not None else None
            text_emb = self.get_text_embedding(input_ids, attention_mask) if input_ids is not None else None

        # If one modality missing, use zeros for the other (so classifier shape consistent)
        if image_emb is None:
            image_emb = torch.zeros((text_emb.size(0), self.embed_dim), device=text_emb.device, dtype=text_emb.dtype)
        if text_emb is None:
            text_emb = torch.zeros((image_emb.size(0), self.embed_dim), device=image_emb.device, dtype=image_emb.dtype)

        # Concatenate and classify
        if self.fusion == "concat":
            x = torch.cat([image_emb, text_emb], dim=1)  # (B, 2*D)
        else:
            # gated: scalar gate per sample
            gate = self.gate_mlp(torch.cat([image_emb, text_emb], dim=1))  # (B, 1)
            x = gate * text_emb + (1.0 - gate) * image_emb  # (B, D)
            
        logits = self.classifier(x)  # (B, num_classes)

        if return_embeddings:
            return logits, image_emb, text_emb
        return logits

    # Convenience wrappers for single-modality inference
    def predict_text_only(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(image=None, input_ids=input_ids, attention_mask=attention_mask)
        return logits

    def predict_image_only(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.forward(image=image, input_ids=None, attention_mask=None)
        return logits


# Example usage (not executed on import)
if __name__ == "__main__":
    import torch
    from transformers import CLIPProcessor

    # quick smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LateFusionClassifier(clip_model_name="openai/clip-vit-base-patch32", freeze_clip=True)
    model.to(device)

    # dummy inputs
    B = 2
    # CLIP expects pixel_values normalized; here we create random tensors for smoke test
    dummy_images = torch.randn(B, 3, 224, 224).to(device)
    # dummy token ids / attention mask (length 16)
    dummy_input_ids = torch.randint(0, 1000, (B, 16)).to(device)
    dummy_attention_mask = torch.ones(B, 16).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image=dummy_images, input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    print("Logits shape:", logits.shape)  # expect (B, 3)
