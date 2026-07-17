"""
infer.py — Inferência ao vivo (webcam → MediaPipe → MLP → letra)
=================================================================

Fluxo:
    frame webcam
      → MediaPipe extrai 21 landmarks
      → features.py normaliza + delta        → vetor (66,)
      → MLPStatic.forward                    → logits
      → softmax                              → prob
      → filtro temporal (N frames iguais)    → letra estável

Uso:
    python -m src.infer
    python -m src.infer --model models/mlp_static.pth --min-conf 0.7 --stable-frames 5

Teclas:
    q → sai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

from src.constants import (
    COR_AMARELO,
    COR_BRANCO,
    COR_CINZA,
    COR_VERDE,
    COR_VERMELHO,
    FINGERTIP_INDICES,
    HAND_CONNECTIONS,
    MODEL_PATH,
)
from src.dataset import load_label_map
from src.features import (
    build_feature_vector,
    expand_frame_for_model,
    extract_landmarks,
    normalize_landmarks,
)
from src.models import MODEL_REGISTRY


# ─────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO
# ─────────────────────────────────────────────────────────────
def load_model(weights_path: Path, config_path: Path, device: torch.device):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg["name"] != "mlp_static":
        raise SystemExit(
            f"infer.py só suporta o MLP estático (frame a frame).\n"
            f"O model_config.json aponta para '{cfg['name']}' — inferência ao "
            f"vivo temporal ainda não implementada.\n"
            f"Retreine o MLP com: python -m src.train --model mlp_static"
        )

    model_cls = MODEL_REGISTRY[cfg["name"]]
    model = model_cls(
        num_classes=cfg["num_classes"],
        input_dim=cfg["input_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────
# DESENHO
# ─────────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], COR_CINZA, 2)
    for i, (px, py) in enumerate(points):
        color = COR_VERMELHO if i in FINGERTIP_INDICES else COR_VERDE
        cv2.circle(frame, (px, py), 5, color, -1)


def draw_prediction(frame, landmarks, label: str, prob: float, stable: bool):
    """Overlay do label acima da mão (ou canto se não há mão)."""
    h, w, _ = frame.shape

    if landmarks is not None:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        cx = int(sum(xs) / len(xs) * w)
        cy = max(int(min(ys) * h) - 30, 40)
    else:
        cx, cy = w // 2, 60

    color = COR_VERDE if stable else COR_AMARELO
    text = f"{label.upper()}  {prob*100:.0f}%"
    cv2.putText(frame, text, (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def draw_hud(frame, fps: float, num_classes: int):
    h, w, _ = frame.shape
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Classes: {num_classes}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_BRANCO, 1)
    cv2.putText(frame, "[q] sair", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CINZA, 1)


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = Path(args.model)
    config_path = Path(args.config)
    label_map_path = Path(args.label_map)

    model, cfg = load_model(weights_path, config_path, device)
    label_map = load_label_map(label_map_path)
    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"Modelo carregado: {cfg['name']}  |  {cfg['num_classes']} classes")

    # ── MediaPipe ────────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    # VIDEO: detecta a palma uma vez e rastreia entre frames (IMAGE redetecta
    # do zero a cada frame). Limiares alinhados ao preprocess do treino.
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Estado do filtro temporal
    recent_preds: deque[int] = deque(maxlen=args.stable_frames)
    # Média móvel de probabilidades — suaviza flicker frame a frame.
    # Guardamos as últimas N distribuições softmax e usamos a média para decidir.
    prob_history: deque[np.ndarray] = deque(maxlen=args.stable_frames)
    prev_wrist: np.ndarray | None = None
    t_inicio = time.monotonic()
    last_time = time.monotonic()
    fps = 0.0

    print("Rodando. Pressione 'q' para sair.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = int((time.monotonic() - t_inicio) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks_for_draw = None
        label_str = "-"
        prob = 0.0
        stable = False

        if result.hand_landmarks:
            raw_landmarks = result.hand_landmarks[0]
            landmarks_for_draw = raw_landmarks
            draw_landmarks(frame, raw_landmarks)

            raw_array = extract_landmarks(raw_landmarks)
            normalized, wrist_raw = normalize_landmarks(raw_array)

            if prev_wrist is None:
                prev_wrist = wrist_raw.copy()
            feature_vector = build_feature_vector(normalized, wrist_raw, prev_wrist)  # (66,)
            prev_wrist = wrist_raw.copy()

            # Expansão para 86 dims (63 landmarks + 3 delta + 20 geométricas)
            # — mesma transformação usada pelo loader no treino (ADR-007).
            # augment=False sempre no infer: perturbar landmarks ao vivo só ia adicionar ruído.
            feature_vector = expand_frame_for_model(feature_vector, augment=False)

            # Inferência
            x = torch.from_numpy(feature_vector).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # Suavização: média das últimas N distribuições. Reduz flicker
            # quando duas classes competem frame a frame.
            prob_history.append(probs)
            smoothed = np.mean(np.stack(prob_history), axis=0)
            top_idx = int(np.argmax(smoothed))
            prob = float(smoothed[top_idx])

            recent_preds.append(top_idx)

            # Guardrails: só mostra label se:
            #   (a) confiança suavizada acima do limiar (default 0.85), E
            #   (b) últimos N frames concordam na mesma classe.
            # Caso contrário mostra "?" — melhor omitir do que sugerir errado.
            consensus = (
                len(recent_preds) == recent_preds.maxlen
                and len(set(recent_preds)) == 1
            )
            confident = prob >= args.min_conf

            if consensus and confident:
                stable = True
                label_str = idx_to_label[top_idx]
            elif confident:
                # Confiante mas ainda sem consenso temporal — amarelo
                label_str = idx_to_label[top_idx]
            else:
                # Abaixo do threshold — não arrisca palpite
                label_str = "?"

        else:
            recent_preds.clear()
            prob_history.clear()
            prev_wrist = None

        # FPS suavizado
        now = time.monotonic()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        draw_hud(frame, fps, cfg["num_classes"])
        draw_prediction(frame, landmarks_for_draw, label_str, prob, stable)

        cv2.imshow("Libras — Inferencia (MLP)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência ao vivo de Libras")
    p.add_argument("--model", type=str, default="models/mlp_static.pth")
    p.add_argument("--config", type=str, default="models/model_config.json")
    p.add_argument("--label-map", type=str, default="models/label_map.json")
    p.add_argument("--min-conf", type=float, default=0.85,
                   help="Confiança mínima (softmax suavizado) para mostrar label. "
                        "Abaixo disso mostra '?' em vez de arriscar palpite.")
    p.add_argument("--stable-frames", type=int, default=7,
                   help="Janela do filtro temporal (também usado para suavizar "
                        "as probabilidades softmax via média móvel).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
