"""
PatchTST Masked Autoencoder (`patchtst_mae.py`)

Self-supervised representation learning on irregular light curves:
75% of contiguous time patches are masked and a lightweight Transformer
decoder reconstructs the normalized per-band magnitudes. Reconstruction
error over masked patches + latent-space density then drive anomaly scoring.

Padding awareness: light curves shorter than `seq_len` are zero-padded by the
dataset; the per-epoch observation mask is turned into per-patch weights so
padded epochs never contribute to the training objective.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

try:
    import lightning as L
    _BASE_CLASS = L.LightningModule
except ImportError:
    try:
        import pytorch_lightning as L
        _BASE_CLASS = L.LightningModule
    except ImportError:
        _BASE_CLASS = nn.Module

from .layers import TimePatchEmbedding, TransformerEncoderBlock, TransformerDecoderBlock, PositionalEncoding


class PatchTSTMaskedAutoencoder(_BASE_CLASS):
    """Masked autoencoder over patch tokens of multi-band light curves."""

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 256,
        patch_len: int = 16,
        d_model: int = 256,
        enc_layers: int = 4,
        enc_heads: int = 8,
        dec_dim: int = 128,
        dec_layers: int = 2,
        dec_heads: int = 4,
        mask_ratio: float = 0.75,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 50,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len}).")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}.")
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.d_model = d_model
        self.dec_dim = dec_dim
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # 1. Patch tokenizer with continuous-time modulation.
        self.patch_embed = TimePatchEmbedding(
            in_channels=in_channels, seq_len=seq_len, patch_len=patch_len, d_model=d_model
        )

        # 2. Encoder over visible patches + [CLS].
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, nhead=enc_heads, dim_feedforward=d_model * 4)
            for _ in range(enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # 3. Encoder->decoder projection and learned mask token.
        self.enc_to_dec = nn.Linear(d_model, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.dec_pos_enc = PositionalEncoding(d_model=dec_dim, max_len=self.num_patches + 1)

        # 4. Lightweight decoder.
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model=dec_dim, nhead=dec_heads, dim_feedforward=dec_dim * 4)
            for _ in range(dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(dec_dim)

        # 5. Projection back to flattened patch space.
        self.pred_proj = nn.Linear(dec_dim, in_channels * patch_len)

    # ------------------------------------------------------------------ #
    # Masking / encoding / decoding
    # ------------------------------------------------------------------ #
    def random_structured_masking(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random per-sample patch masking.

        Returns:
            mask:        [B, N] binary, 1 = masked, 0 = visible
            ids_keep:    [B, num_visible] indices of visible patches
            ids_restore: [B, N] permutation restoring original patch order
        """
        num_keep = max(int(self.num_patches * (1.0 - self.mask_ratio)), 1)
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        mask = torch.ones(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids_keep, 0)
        return mask, ids_keep, ids_restore

    def forward_encoder(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, x_flat = self.patch_embed(x, t)
        batch_size = tokens.size(0)

        cls_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]

        mask, ids_keep, ids_restore = self.random_structured_masking(batch_size, tokens.device)
        ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, self.d_model)
        visible = torch.gather(patch_tokens, dim=1, index=ids_keep_exp)

        enc_tokens = torch.cat([cls_token, visible], dim=1)
        for block in self.encoder_blocks:
            enc_tokens = block(enc_tokens)
        enc_tokens = self.enc_norm(enc_tokens)

        return enc_tokens, mask, ids_restore, x_flat

    def forward_decoder(self, enc_tokens: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        batch_size = enc_tokens.size(0)
        dec_tokens = self.enc_to_dec(enc_tokens)

        cls_dec = dec_tokens[:, :1, :]
        visible_dec = dec_tokens[:, 1:, :]

        num_masked = self.num_patches - visible_dec.size(1)
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)

        full = torch.cat([visible_dec, mask_tokens], dim=1)
        ids_restore_exp = ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_dim)
        full = torch.gather(full, dim=1, index=ids_restore_exp)

        dec_input = torch.cat([cls_dec, full], dim=1)
        dec_input = self.dec_pos_enc(dec_input)
        for block in self.decoder_blocks:
            dec_input = block(dec_input)
        dec_input = self.dec_norm(dec_input)

        return self.pred_proj(dec_input[:, 1:, :])  # [B, N, C * P]

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            patch_preds: [B, N, in_channels * patch_len] reconstructions
            x_flat:      [B, N, in_channels * patch_len] targets
            mask:        [B, N] binary patch mask (1 = was masked)
        """
        enc_tokens, mask, ids_restore, x_flat = self.forward_encoder(x, t)
        patch_preds = self.forward_decoder(enc_tokens, ids_restore)
        return patch_preds, x_flat, mask

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_cls_latent(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[CLS] embedding with no masking, for density-based anomaly scoring. [B, d_model]"""
        tokens, _ = self.patch_embed(x, t)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.enc_norm(tokens)
        return tokens[:, 0, :]

    def patch_observation_weights(self, obs_mask: Optional[torch.Tensor], batch_size: int, device) -> torch.Tensor:
        """Per-patch fraction of real (non-padded) epochs. [B, N]"""
        if obs_mask is None:
            return torch.ones(batch_size, self.num_patches, device=device)
        return obs_mask.reshape(batch_size, self.num_patches, self.patch_len).mean(dim=-1)

    @torch.no_grad()
    def per_sample_recon_error(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        n_repeats: int = 4,
    ) -> torch.Tensor:
        """
        Masked-patch reconstruction MSE per sample, averaged over `n_repeats`
        independent random maskings to reduce mask-sampling variance. [B]
        """
        batch_size = x.size(0)
        obs_w = self.patch_observation_weights(obs_mask, batch_size, x.device)
        totals = torch.zeros(batch_size, device=x.device)

        for _ in range(max(n_repeats, 1)):
            preds, targets, mask = self.forward(x, t)
            per_patch = ((preds - targets) ** 2).mean(dim=-1)          # [B, N]
            # Same per-sample variance normalization as the training loss, so
            # errors are comparable across objects of different amplitudes.
            target_var = targets.var(dim=-1).mean(dim=-1, keepdim=True) + 1e-6
            per_patch = per_patch / target_var
            w = mask * obs_w                                           # masked AND observed
            totals += (per_patch * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)

        return totals / max(n_repeats, 1)

    @torch.no_grad()
    def reconstruct(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-sequence reconstruction [B, C, seq_len] plus the patch mask [B, N]."""
        patch_preds, _, mask = self.forward(x, t)
        batch_size = patch_preds.size(0)
        # Inverse of the tokenizer flattening: [B, N, P*C] -> [B, N, P, C] -> [B, L, C] -> [B, C, L]
        preds = patch_preds.reshape(batch_size, self.num_patches, self.patch_len, self.in_channels)
        preds = preds.reshape(batch_size, self.seq_len, self.in_channels).transpose(1, 2)
        return preds, mask

    # ------------------------------------------------------------------ #
    # Loss / Lightning plumbing
    # ------------------------------------------------------------------ #
    def compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        patch_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE over masked patches only, normalized by per-sample target variance
        (keeps high-amplitude variables from dominating the objective), and
        weighted by the fraction of real observations in each patch.
        """
        per_patch = ((preds - targets) ** 2).mean(dim=-1)              # [B, N]
        target_var = targets.var(dim=-1).mean(dim=-1, keepdim=True) + 1e-6  # [B, 1]
        per_patch = per_patch / target_var

        weights = mask if patch_weights is None else mask * patch_weights
        return (per_patch * weights).sum() / (weights.sum() + 1e-6)

    def _shared_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["x"]
        t = batch.get("t")
        obs_mask = batch.get("obs_mask")
        preds, targets, mask = self.forward(x, t)
        obs_w = self.patch_observation_weights(obs_mask, x.size(0), x.device)
        return self.compute_loss(preds, targets, mask, patch_weights=obs_w)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=batch["x"].size(0))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=batch["x"].size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_epochs, 1), eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
