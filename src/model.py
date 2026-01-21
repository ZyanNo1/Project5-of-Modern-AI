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
             - film:  x = (1 + gamma) * txt + beta + img   (vector FiLM, with residual img)
             - film_concat: x = [ (1+gamma_v)*img + beta_v ; (1+gamma_t)*txt + beta_t ]
    """

    def __init__(self,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 embed_dim: Optional[int] = None,
                 hidden_dims: Tuple[int, ...] = (512, 128),
                 dropout: float = 0.3,
                 freeze_clip: bool = True,
                 num_classes: int = 3,
                 fusion: str = "concat",
                 cross_attn_heads: int = 8):
        super().__init__()

        self.fusion = fusion.lower().strip()

        # Load CLIP model (image + text encoders)
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        cfg: CLIPConfig = self.clip.config
        self.text_hidden = cfg.text_config.hidden_size
        self.vision_hidden = cfg.vision_config.hidden_size

        # projection dim used by get_*_features
        if embed_dim is None:
            embed_dim = getattr(cfg.text_config, "projection_dim", 512)
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
        elif self.fusion in ("film", "film_concat"):
            # Produce gamma/beta vectors from concatenated embeddings
            # Keep it small to avoid overfitting.
            out_dim = (2 * self.embed_dim) if self.fusion == "film" else (4 * self.embed_dim)
            self.film_mlp = nn.Sequential(
                nn.Linear(self.embed_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, out_dim)
            )
            self.gate_mlp = None
            head_input_dim = self.embed_dim if self.fusion == "film" else (2 * self.embed_dim)
        elif self.fusion == "concat":
            self.gate_mlp = None
            head_input_dim = self.embed_dim * 2
        elif self.fusion == "cross_attn":
            self.gate_mlp = None
            self.film_mlp = None

            attn_dim = self.embed_dim  # keep it small/stable
            self.q_proj = nn.Linear(self.text_hidden, attn_dim)
            self.kv_proj = nn.Linear(self.vision_hidden, attn_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=attn_dim, num_heads=cross_attn_heads, dropout=dropout, batch_first=True
            )
            self.out_proj = nn.Sequential(
                nn.LayerNorm(attn_dim),
                nn.Dropout(p=dropout),
                nn.Linear(attn_dim, self.embed_dim)
            )
            self.attn_gate = nn.Parameter(torch.tensor(-4.0))  # sigmoid(-4)≈0.018, starts almost off
            # head will see [cross_attended_text ; pooled_text ; pooled_image]
            head_input_dim = self.embed_dim * 3
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

        backbone_frozen = not any(p.requires_grad for p in self.clip.parameters())

        if self.fusion != "cross_attn":
            # ---- existing pooled-embedding paths ----
            if backbone_frozen:
                with torch.no_grad():
                    image_emb = self.get_image_embedding(image) if image is not None else None
                    text_emb = self.get_text_embedding(input_ids, attention_mask) if input_ids is not None else None
            else:
                image_emb = self.get_image_embedding(image) if image is not None else None
                text_emb = self.get_text_embedding(input_ids, attention_mask) if input_ids is not None else None

            if image_emb is None:
                image_emb = torch.zeros((text_emb.size(0), self.embed_dim), device=text_emb.device, dtype=text_emb.dtype)
            if text_emb is None:
                text_emb = torch.zeros((image_emb.size(0), self.embed_dim), device=image_emb.device, dtype=image_emb.dtype)

            if self.fusion == "concat":
                x = torch.cat([image_emb, text_emb], dim=1)
            elif self.fusion == "gated":
                gate = self.gate_mlp(torch.cat([image_emb, text_emb], dim=1))
                x = gate * text_emb + (1.0 - gate) * image_emb
            elif self.fusion == "film":
                params = self.film_mlp(torch.cat([image_emb, text_emb], dim=1))
                gamma, beta = params.chunk(2, dim=1)
                x = (1.0 + torch.tanh(gamma)) * text_emb + beta + image_emb
            elif self.fusion == "film_concat":
                params = self.film_mlp(torch.cat([image_emb, text_emb], dim=1))
                gamma_v, beta_v, gamma_t, beta_t = params.chunk(4, dim=1)
                beta_scale = 0.1
                v = (1.0 + torch.tanh(gamma_v)) * image_emb + beta_scale * beta_v
                t = (1.0 + torch.tanh(gamma_t)) * text_emb + beta_scale * beta_t
                x = torch.cat([v, t], dim=1)
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion}")

            logits = self.classifier(x)
            if return_embeddings:
                return logits, image_emb, text_emb
            return logits

        # ---- cross_attn path  ----
        if image is None or input_ids is None or attention_mask is None:
            raise ValueError("fusion=cross_attn requires both image and text inputs.")

        # get pooled embeddings (still useful as global features)
        if backbone_frozen:
            with torch.no_grad():
                pooled_img = self.get_image_embedding(image)
                pooled_txt = self.get_text_embedding(input_ids, attention_mask)

                v_hid = self.clip.vision_model(pixel_values=image).last_hidden_state  # (B, Nv, Hv)
                t_hid = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, Lt, Ht)
        else:
            pooled_img = self.get_image_embedding(image)
            pooled_txt = self.get_text_embedding(input_ids, attention_mask)
            v_hid = self.clip.vision_model(pixel_values=image).last_hidden_state
            t_hid = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # projections to attn_dim
        q = self.q_proj(t_hid)         # (B, Lt, D)
        kv = self.kv_proj(v_hid)       # (B, Nv, D)

        # key padding: vision has no padding; for text we can mask queries implicitly by using attention_mask and pooling later
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)  # (B, Lt, D)

        # masked mean pool over text length
        m = attention_mask.unsqueeze(-1).to(attn_out.dtype)  # (B, Lt, 1)
        attn_pooled = (attn_out * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))  # (B, D)
        attn_feat = self.out_proj(attn_pooled)  # (B, embed_dim)
        
        gate = torch.sigmoid(self.attn_gate)  # scalar in (0,1)
        attn_feat = gate * attn_feat

        x = torch.cat([attn_feat, pooled_txt, pooled_img], dim=1)  # (B, 3*embed_dim)
        logits = self.classifier(x)
        if return_embeddings:
            return logits, pooled_img, pooled_txt
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
