"""
Core Neural Network Layers for PatchTST Masked Autoencoder (`layers.py`)
Includes Time-Aware Patch Tokenization (`TimePatchEmbedding`), Positional Encodings,
and Transformer Encoder/Decoder Blocks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for patch sequences and [CLS] token."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1), :]


class TimePatchEmbedding(nn.Module):
    """
    Splits multi-band irregular time-series [Batch, Channels=2, SeqLen=256]
    into contiguous patches [Batch, Num_Patches=16, Dim=256] and appends a [CLS] token.
    Incorporates learnable continuous time positional modulation if time timestamps `t` are provided.
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 256,
        patch_len: int = 16,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.d_model = d_model

        # Patch projection: flattens (in_channels * patch_len) -> d_model
        self.patch_proj = nn.Linear(in_channels * patch_len, d_model)
        
        # Time projection: projects average time difference per patch into d_model embedding
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model)
        )

        # [CLS] token for global anomaly representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=self.num_patches + 1)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [Batch, Channels=2, SeqLen=256] or [Batch, SeqLen=256, Channels=2]
            t: Optional observation timestamps [Batch, SeqLen=256]
        Returns:
            tokens: [Batch, Num_Patches + 1, d_model]
            patches: Raw patch tensor [Batch, Num_Patches, in_channels * patch_len]
        """
        if x.dim() == 3 and x.size(-1) == self.seq_len:
            # Shape is [Batch, Channels, SeqLen], transpose to [Batch, SeqLen, Channels]
            x = x.transpose(1, 2)
        
        batch_size = x.size(0)
        
        # Reshape to patches: [Batch, Num_Patches, Patch_Len, Channels]
        x_patches = x.reshape(batch_size, self.num_patches, self.patch_len, self.in_channels)
        
        # Flatten patch dimension: [Batch, Num_Patches, Patch_Len * Channels]
        x_flat = x_patches.reshape(batch_size, self.num_patches, self.patch_len * self.in_channels)
        
        # Linear projection to embedding dim
        tokens = self.patch_proj(x_flat)  # [Batch, Num_Patches, d_model]

        # Inject continuous time modulation if available
        if t is not None:
            if t.dim() == 3:
                # If times is [Batch, Channels, SeqLen] or [Batch, SeqLen, Channels], average across channels
                t = t.float().mean(dim=1 if t.size(1) == self.in_channels else 2)
            t_patches = t.reshape(batch_size, self.num_patches, self.patch_len)
            # Use mean timestamp across the patch
            t_mean = t_patches.mean(dim=-1, keepdim=True)  # [Batch, Num_Patches, 1]
            time_embed = self.time_proj(t_mean)            # [Batch, Num_Patches, d_model]
            tokens = tokens + time_embed

        # Append [CLS] token to the front
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [Batch, 1, d_model]
        tokens = torch.cat([cls_tokens, tokens], dim=1)         # [Batch, Num_Patches + 1, d_model]

        # Add positional encoding & layer norm
        tokens = self.pos_enc(tokens)
        tokens = self.norm(self.dropout(tokens))

        return tokens, x_flat


class TransformerEncoderBlock(nn.Module):
    """Standard multi-head self-attention + feed-forward encoder layer."""
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerDecoderBlock(nn.Module):
    """Lightweight self-attention decoder layer for reconstruction."""
    def __init__(self, d_model: int = 128, nhead: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
