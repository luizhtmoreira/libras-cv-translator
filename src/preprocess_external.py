"""
preprocess_external.py — Converte datasets externos em .npy (30, 66)
=====================================================================

Alimenta o mesmo pipeline dos scripts de captura, mas em batch sobre
imagens já baixadas. Elimina a necessidade da equipe gravar dataset.

Sources suportados:
    --source alphabet  → Brazilian Sign Language Alphabet (biankatpas, imagens)
                         https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset
                         Cada imagem vira um tensor (30, 66) replicando o frame
                         estático 30x com delta zero.

Layout de entrada esperado (por convenção):
    <input>/<label>/*.jpg|*.png   (alphabet)

Layout de saída (canônico):
    <output>/<label>/<i>.npy       shape (30, 66) float32

Uso:
    python -m src.preprocess_external --source alphabet \
        --input data/external/brazilian-alphabet \
        --output data/processed/alphabet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from tqdm import tqdm

from src.constants import FRAMES, MODEL_PATH
from src.features import build_feature_vector, extract_landmarks, normalize_landmarks

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────
# MEDIAPIPE HELPERS
# ─────────────────────────────────────────────────────────────
def make_landmarker(mode: RunningMode):
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def frame_to_mp_image(bgr_frame: np.ndarray) -> mp.Image:
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
    )


# ─────────────────────────────────────────────────────────────
# ALFABETO — IMAGENS ESTÁTICAS
# ─────────────────────────────────────────────────────────────
def process_alphabet(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Cada imagem gera 1 tensor (30, 66):
        - 63 primeiras dims: landmarks normalizados
        - 3 últimas dims: delta = 0 (sinal estático)
        - Replicado 30x pelo eixo temporal
    """
    landmarker = make_landmarker(RunningMode.IMAGE)
    stats: dict[str, int] = {}
    dropped = 0

    class_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"Nenhuma subpasta de classe em {input_dir}")

    for cls_dir in class_dirs:
        label = cls_dir.name.lower()
        out_cls_dir = output_dir / label
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        # Continua do próximo índice se já existirem arquivos
        existing = list(out_cls_dir.glob("*.npy"))
        next_idx = (
            max((int(p.stem) for p in existing if p.stem.isdigit()), default=-1) + 1
        )

        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        for img_path in tqdm(images, desc=f"[{label}]", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                dropped += 1
                continue

            result = landmarker.detect(frame_to_mp_image(img))
            if not result.hand_landmarks:
                dropped += 1
                continue

            raw = extract_landmarks(result.hand_landmarks[0])
            normalized, wrist_raw = normalize_landmarks(raw)
            # Sinal estático: delta é sempre zero
            vec = build_feature_vector(normalized, wrist_raw, wrist_raw)  # (66,)
            tensor = np.tile(vec, (FRAMES, 1)).astype(np.float32)         # (30, 66)

            np.save(out_cls_dir / f"{next_idx}.npy", tensor)
            next_idx += 1

        stats[label] = next_idx
        print(f"  {label}: {next_idx} amostras totais em {out_cls_dir}")

    landmarker.close()
    print(f"\nImagens descartadas (sem mão detectada / falha de leitura): {dropped}")
    return stats


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocessa datasets externos de Libras")
    p.add_argument("--source", choices=["alphabet"], required=True)
    p.add_argument("--input", type=str, required=True,
                   help="Raiz do dataset externo (subpastas por classe)")
    p.add_argument("--output", type=str, required=True,
                   help="Onde salvar os .npy processados")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input não existe: {input_dir}")

    print(f"Fonte: {args.source}  |  input={input_dir}  →  output={output_dir}")
    t0 = time.time()

    stats = process_alphabet(input_dir, output_dir)

    dt = time.time() - t0
    total = sum(stats.values())
    print(f"\nConcluído em {dt:.1f}s.  Total: {total} tensores em {len(stats)} classes.")


if __name__ == "__main__":
    main()
