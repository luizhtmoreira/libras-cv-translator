"""
report_figures.py — Figuras para o notebook/relatório final (Fase 6)
====================================================================

Gera, a partir dos artefatos de treino em models/:
    reports/{modelo}_confusion_matrix.png   — matriz de confusão (val split)
    reports/{modelo}_curves.png             — loss + F1 macro por época

A matriz de confusão é recomputada rodando o melhor checkpoint sobre o mesmo
split de validação do treino (mesma seed/val_ratio dos HPARAMS persistidos no
train_log — reprodutível por construção).

Uso:
    python -m src.report_figures --model mlp_static --data-dir data/processed/alphabet
    python -m src.report_figures --model transformer_temporal --data-dir data/luiz_split/dynamic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import LibrasDataset, load_label_map, stratified_split
from src.models import MODEL_DATASET_MODE, MODEL_REGISTRY


def load_model(model_name: str, models_dir: Path, device: torch.device):
    with open(models_dir / f"{model_name}_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if "hidden_dims" in kwargs:
        kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
    model = MODEL_REGISTRY[cfg["name"]](**kwargs)
    model.load_state_dict(
        torch.load(models_dir / f"{model_name}.pth", map_location=device)
    )
    model.to(device).eval()
    return model


def plot_curves(history: list[dict], model_name: str, out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["val_f1_macro"] for h in history], label="F1 macro (val)")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="acc (train)", alpha=0.6)
    ax2.axhline(0.90, color="gray", ls="--", lw=1, label="alvo 0.90")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Métrica")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"{model_name} — F1 macro")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Curvas salvas em {out_path}")


def plot_confusion(model, model_name: str, data_roots: list[Path],
                   models_dir: Path, hparams: dict, device, out_path: Path,
                   no_cap_dirs: list[str]) -> None:
    label_map = load_label_map(models_dir / f"{model_name}_label_map.json")
    mode = MODEL_DATASET_MODE[model_name]

    # Mesma subamostragem do treino (max_per_class/seed vêm do train_log):
    # listagem de arquivos diferente = split de validação diferente = leakage.
    ds = LibrasDataset(
        data_roots, mode=mode, label_map=label_map, augment=False,
        max_per_class=hparams.get("max_per_class"),
        subsample_seed=hparams["seed"],
        no_cap_dirs=no_cap_dirs,
    )
    _, val_subset = stratified_split(
        ds, val_ratio=hparams["val_ratio"], seed=hparams["seed"], val_dataset=ds
    )
    loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())

    labels_sorted = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_map))))

    size = max(7, 0.5 * len(labels_sorted) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=labels_sorted).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45,
        values_format="d",
    )
    ax.set_title(f"{model_name} — matriz de confusão (validação, n={len(y_true)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Gera figuras de relatório a partir de models/")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--data-dir", required=True,
                   help="Mesmo --data-dir usado no treino (para recompor o split de val)")
    p.add_argument("--extra-dir", type=str, nargs="+", default=None,
                   help="Mesmos --extra-dir usados no treino, se houver")
    p.add_argument("--no-cap-dir", type=str, nargs="+",
                   default=["data/raw", "data/raw/dynamic"],
                   help="Mesmos roots protegidos do max_per_class usados no treino")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--out-dir", type=str, default="reports")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(models_dir / f"{args.model}_train_log.json", encoding="utf-8") as f:
        train_log = json.load(f)

    plot_curves(train_log["history"], args.model, out_dir / f"{args.model}_curves.png")

    data_roots = [Path(args.data_dir)]
    if args.extra_dir:
        data_roots.extend(Path(d) for d in args.extra_dir)
    model = load_model(args.model, models_dir, device)
    plot_confusion(model, args.model, data_roots, models_dir,
                   train_log["hparams"], device,
                   out_dir / f"{args.model}_confusion_matrix.png",
                   no_cap_dirs=args.no_cap_dir)


if __name__ == "__main__":
    main()
