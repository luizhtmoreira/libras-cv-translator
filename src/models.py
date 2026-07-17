"""
models.py — Arquiteturas do estudo de ablação de Libras
========================================================

Duas famílias, consumindo variações do tensor (30, 86) — expandido on-the-fly
pelo loader a partir do (30, 66) em disco:

    MLPStatic           → baseline: lê 1 frame expandido (86,), ignora tempo
    TransformerTemporal → temporal: consome a janela inteira (30, 86)

Ambos têm .config() serializável — usado por infer.py para reconstruir a rede
sem hardcode dos hiperparams.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.constants import FEATURE_DIM_EXTENDED, FRAMES


# ─────────────────────────────────────────────────────────────
# MLP ESTÁTICO (baseline)
# ─────────────────────────────────────────────────────────────
class MLPStatic(nn.Module):
    """
    MLP simples sobre um único frame de features (86 dims por padrão).

    Entrada:  (B, input_dim)
    Saída:    (B, num_classes)  — logits (sem softmax; use CrossEntropyLoss)

    Arquitetura: [Linear→ReLU→Dropout] × N → Linear
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = FEATURE_DIM_EXTENDED,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.num_classes = num_classes
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict:
        """Serializável — infer.py reconstrói a rede a partir daqui."""
        return {
            "name": "mlp_static",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }


# ─────────────────────────────────────────────────────────────
# TRANSFORMER TEMPORAL
# ─────────────────────────────────────────────────────────────
def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Positional encoding senoidal padrão (Vaswani et al 2017), pré-computado.
    Retorna (1, seq_len, d_model) — broadcast direto no batch.
    """
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class TransformerTemporal(nn.Module):
    """
    Classificador temporal para letras dinâmicas (h/j/k/x/y/z).

    Entrada:  (B, seq_len, input_dim)  ex: (B, 30, 86)
    Saída:    (B, num_classes)         — logits

    Pipeline:
        Linear(input_dim → d_model)
        + positional encoding senoidal
        → N × TransformerEncoderLayer (self-attn + FFN)
        → LayerNorm
        → mean pooling temporal
        → Linear(d_model → num_classes)

    Escolha do mean pooling em vez de CLS token: dataset pequeno, mean pooling
    dá regularização implícita e evita o token virar bottleneck.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = FEATURE_DIM_EXTENDED,
        seq_len: int = FRAMES,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, d_model)
        # PE pré-computado como buffer — não é treinável.
        self.register_buffer(
            "pos_enc",
            _sinusoidal_positional_encoding(seq_len, d_model),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN — mais estável em treinos curtos
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        h = self.input_proj(x) + self.pos_enc[:, : x.size(1)]
        h = self.encoder(h)          # (B, T, d_model)
        h = self.norm(h)
        h = h.mean(dim=1)            # (B, d_model) — mean pooling temporal
        return self.head(h)

    def config(self) -> dict:
        return {
            "name": "transformer_temporal",
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp_static": MLPStatic,
    "transformer_temporal": TransformerTemporal,
}


# Modo do dataset que cada modelo consome. Usado pelo train.py para não obrigar
# o usuário a lembrar dessa combinação.
MODEL_DATASET_MODE: dict[str, str] = {
    "mlp_static": "static",
    "transformer_temporal": "sequence",
}
