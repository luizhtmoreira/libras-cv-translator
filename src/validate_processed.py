"""
validate_processed.py — Auditoria de qualidade dos .npy processados
===================================================================

Verifica um diretório de dados canônicos (30, 66) e reporta:

  - shape/dtype errados (erro fatal — quebra o contrato do pipeline)
  - amostras degeneradas: pulso praticamente parado em ≥ N dos 30 frames
    (mão estática numa amostra supostamente dinâmica — captura ruim)
  - NaN/Inf

Com --delete-degenerate, remove as amostras degeneradas (pede o flag
explícito; por padrão só reporta).

Uso:
    python -m src.validate_processed --data-dir data/processed/alphabet
    python -m src.validate_processed --data-dir data/luiz_split/dynamic --delete-degenerate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.constants import FEATURE_DIM, FRAMES


def audit_sample(path: Path, still_eps: float, max_still_frames: int) -> list[str]:
    """Retorna a lista de problemas da amostra (vazia = ok)."""
    problems: list[str] = []
    tensor = np.load(path)

    if tensor.shape != (FRAMES, FEATURE_DIM):
        return [f"shape {tensor.shape} != {(FRAMES, FEATURE_DIM)}"]
    if not np.isfinite(tensor).all():
        problems.append("NaN/Inf")

    # Delta do pulso (3 últimas dims) ≈ 0 em quase todos os frames → mão parada.
    # O primeiro frame tem delta 0 por construção, por isso o limiar é sobre
    # os 29 seguintes.
    deltas = np.linalg.norm(tensor[1:, 63:66], axis=1)
    still = int((deltas < still_eps).sum())
    if still >= max_still_frames:
        problems.append(f"degenerada: pulso parado em {still}/29 frames")

    return problems


def main() -> None:
    p = argparse.ArgumentParser(description="Audita .npy (30,66) processados")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--still-eps", type=float, default=1e-4,
                   help="Norma do delta do pulso abaixo disso = frame parado")
    p.add_argument("--max-still-frames", type=int, default=25,
                   help="Frames parados (de 29) a partir do qual a amostra é degenerada")
    p.add_argument("--delete-degenerate", action="store_true",
                   help="Remove amostras degeneradas em vez de só reportar")
    args = p.parse_args()

    root = Path(args.data_dir)
    files = sorted(root.rglob("*.npy"))
    if not files:
        raise SystemExit(f"Nenhum .npy em {root}")

    per_class: dict[str, int] = {}
    bad: list[tuple[Path, list[str]]] = []
    for f in files:
        label = f.parent.name
        per_class[label] = per_class.get(label, 0) + 1
        problems = audit_sample(f, args.still_eps, args.max_still_frames)
        if problems:
            bad.append((f, problems))

    print(f"{len(files)} amostras em {len(per_class)} classes:")
    for label in sorted(per_class):
        print(f"  {label:12s} {per_class[label]}")

    if not bad:
        print("\nNenhum problema encontrado.")
        return

    print(f"\n{len(bad)} amostras com problema:")
    for f, problems in bad:
        print(f"  {f.relative_to(root)}: {'; '.join(problems)}")

    fatal = [f for f, probs in bad if any("shape" in p or "NaN" in p for p in probs)]
    if fatal:
        raise SystemExit(f"\n{len(fatal)} amostras com erro fatal (shape/NaN) — investigue o preprocess.")

    if args.delete_degenerate:
        for f, _ in bad:
            f.unlink()
        print(f"\n{len(bad)} amostras degeneradas removidas.")
    else:
        print("\nRode com --delete-degenerate para removê-las.")


if __name__ == "__main__":
    main()
