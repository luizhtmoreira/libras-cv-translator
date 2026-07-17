"""
infer_temporal.py — Inferência ao vivo dual: letra ↔ letra dinâmica (ADR-010)
==============================================================================

Dois pipelines coexistem, alternados por tecla (arquitetura dual, ADR-012):

    MODO LETRA (MLP estático — 21 letras)
        frame → landmarks → vec (66,) → expand (86,) → MLPStatic → letra
        Mesmos guardrails do infer.py: softmax suavizado + consenso temporal.

    MODO LETRA DINÂMICA (Transformer temporal — h/j/k/x/y/z)
        frame → landmarks → vec (66,) → buffer circular de 30 frames
        A cada N frames (default 5 ≈ 6 predições/s), o buffer vira (30, 86)
        e passa pelo TransformerTemporal. Predição só dispara quando:
          - buffer cheio (30 frames), E
          - mão detectada há ≥ 10 frames contínuos (evita transições), E
          - ≥ N frames desde a última predição (sem spam).
        A letra dinâmica vira "estável" quando duas predições consecutivas
        concordam acima do limiar de confiança.

O buffer é alimentado nos DOIS modos — trocar para LETRA DINÂMICA já encontra
a janela quente, sem esperar 1 segundo de warm-up.

Uso:
    python -m src.infer_temporal
    python -m src.infer_temporal --model models/transformer_temporal.pth \
        --config models/transformer_temporal_config.json

Teclas:
    espaço → alterna modo letra/letra dinâmica
    q      → sai
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
    COR_ROXO,
    COR_VERDE,
    FRAMES,
    MODEL_PATH,
)
from src.dataset import load_label_map
from src.features import (
    build_feature_vector,
    expand_frame_for_model,
    expand_sequence_for_model,
    extract_landmarks,
    normalize_landmarks,
)
from src.infer import draw_landmarks
from src.models import MODEL_REGISTRY

MODE_LETTER = "letra"
MODE_DYNAMIC = "dinamica"

MODE_DISPLAY = {
    MODE_LETTER: "LETRA",
    MODE_DYNAMIC: "LETRA DINAMICA",
}


# ─────────────────────────────────────────────────────────────
# CARREGAMENTO GENÉRICO DE MODELO
# ─────────────────────────────────────────────────────────────
def load_model_from_config(
    weights_path: Path,
    config_path: Path,
    expected_name: str,
    device: torch.device,
):
    """Reconstrói qualquer modelo do MODEL_REGISTRY a partir do seu config()."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg["name"] != expected_name:
        raise SystemExit(
            f"Config {config_path} aponta para '{cfg['name']}', esperado "
            f"'{expected_name}'. Retreine com: python -m src.train --model {expected_name}"
        )

    model_cls = MODEL_REGISTRY[cfg["name"]]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if "hidden_dims" in kwargs:
        kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
    model = model_cls(**kwargs)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, cfg


# ─────────────────────────────────────────────────────────────
# PIPELINE ESTÁTICO (mesma lógica do infer.py, encapsulada)
# ─────────────────────────────────────────────────────────────
class StaticRecognizer:
    """MLP frame a frame com suavização de softmax + consenso temporal."""

    def __init__(self, model, idx_to_label: dict[int, str], device,
                 min_conf: float = 0.85, stable_frames: int = 7):
        self.model = model
        self.idx_to_label = idx_to_label
        self.device = device
        self.min_conf = min_conf
        self.recent_preds: deque[int] = deque(maxlen=stable_frames)
        self.prob_history: deque[np.ndarray] = deque(maxlen=stable_frames)

    def reset(self) -> None:
        self.recent_preds.clear()
        self.prob_history.clear()

    def process(self, vec66: np.ndarray) -> tuple[str, float, bool]:
        """vec66 (66,) → (label, prob suavizada, estável?)."""
        feat = expand_frame_for_model(vec66, augment=False)  # (86,)
        x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        self.prob_history.append(probs)
        smoothed = np.mean(np.stack(self.prob_history), axis=0)
        top_idx = int(np.argmax(smoothed))
        prob = float(smoothed[top_idx])
        self.recent_preds.append(top_idx)

        consensus = (
            len(self.recent_preds) == self.recent_preds.maxlen
            and len(set(self.recent_preds)) == 1
        )
        confident = prob >= self.min_conf

        if confident:
            return self.idx_to_label[top_idx], prob, consensus and confident
        return "?", prob, False


# ─────────────────────────────────────────────────────────────
# PIPELINE TEMPORAL (buffer de 30 frames → Transformer)
# ─────────────────────────────────────────────────────────────
class DynamicRecognizer:
    """
    Consome vetores (66,) frame a frame e roda o Transformer sobre a janela
    (30, 86) quando o trigger permite. Ver docstring do módulo para as regras.
    """

    def __init__(self, model, idx_to_label: dict[int, str], device,
                 min_conf: float = 0.60, pred_every: int = 5,
                 min_hand_frames: int = 10, history_size: int = 3):
        self.model = model
        self.idx_to_label = idx_to_label
        self.device = device
        self.min_conf = min_conf
        self.pred_every = pred_every
        self.min_hand_frames = min_hand_frames

        self.buffer: deque[np.ndarray] = deque(maxlen=FRAMES)
        self.frames_since_hand = 0
        self.frames_since_pred = 0
        self._last_candidate: int | None = None

        self.last_word: str | None = None
        self.last_prob = 0.0
        self.history: deque[str] = deque(maxlen=history_size)

    def push(self, vec66: np.ndarray | None) -> None:
        """Alimenta o buffer. vec66=None significa frame sem mão detectada."""
        if vec66 is not None:
            self.buffer.append(vec66)
            self.frames_since_hand += 1
            self.frames_since_pred += 1
        else:
            # Perdeu a mão: não zera o buffer (o gesto pode continuar), mas o
            # contador de estabilidade recomeça — só prediz com mão firme.
            self.frames_since_hand = 0

    def should_predict(self) -> bool:
        return (
            len(self.buffer) == FRAMES
            and self.frames_since_hand >= self.min_hand_frames
            and self.frames_since_pred >= self.pred_every
        )

    def predict(self) -> tuple[str, float, bool]:
        """Roda o Transformer no buffer. Retorna (letra dinâmica, prob, estável?)."""
        seq = np.stack(list(self.buffer))                       # (30, 66)
        seq86 = expand_sequence_for_model(seq, augment=False)   # (30, 86)
        x = torch.from_numpy(seq86).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self.frames_since_pred = 0

        top_idx = int(np.argmax(probs))
        prob = float(probs[top_idx])

        if prob < self.min_conf:
            self._last_candidate = None
            return "?", prob, False

        # Estável = duas predições consecutivas concordando acima do limiar.
        stable = top_idx == self._last_candidate
        self._last_candidate = top_idx
        word = self.idx_to_label[top_idx]

        if stable:
            self.last_word = word
            self.last_prob = prob
            if not self.history or self.history[-1] != word:
                self.history.append(word)
        return word, prob, stable


# ─────────────────────────────────────────────────────────────
# DESENHO
# ─────────────────────────────────────────────────────────────
def draw_mode_hud(frame, mode: str, fps: float) -> None:
    h, w, _ = frame.shape
    color = COR_VERDE if mode == MODE_LETTER else COR_ROXO
    cv2.rectangle(frame, (0, 0), (w, 38), (30, 30, 30), -1)
    cv2.putText(frame, f"MODO: {MODE_DISPLAY[mode]}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_BRANCO, 1)
    cv2.putText(frame, "[espaco] troca modo   [q] sair", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_CINZA, 1)


def draw_letter_overlay(frame, landmarks, label: str, prob: float, stable: bool) -> None:
    h, w, _ = frame.shape
    if landmarks is not None:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        cx = int(sum(xs) / len(xs) * w)
        cy = max(int(min(ys) * h) - 30, 70)
    else:
        cx, cy = w // 2, 90
    color = COR_VERDE if stable else COR_AMARELO
    cv2.putText(frame, f"{label.upper()}  {prob*100:.0f}%", (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def draw_dynamic_overlay(frame, rec: DynamicRecognizer,
                         candidate: str | None, prob: float, stable: bool) -> None:
    h, w, _ = frame.shape

    # Barra de preenchimento do buffer (canto superior, abaixo do HUD)
    frac = len(rec.buffer) / FRAMES
    bar_w = int(0.35 * w)
    cv2.rectangle(frame, (10, 48), (10 + bar_w, 60), COR_CINZA, 1)
    cv2.rectangle(frame, (10, 48), (10 + int(bar_w * frac), 60), COR_ROXO, -1)
    cv2.putText(frame, f"buffer {len(rec.buffer)}/{FRAMES}", (18 + bar_w, 59),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_CINZA, 1)

    # Última letra dinâmica estável, em destaque
    if rec.last_word is not None:
        cv2.putText(frame, rec.last_word.upper().replace("_", " "),
                    (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, COR_VERDE, 3)
        cv2.putText(frame, f"{rec.last_prob*100:.0f}%", (10, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)

    # Candidata ainda não estável (amarelo, menor)
    if candidate is not None and candidate != "?" and not stable:
        cv2.putText(frame, f"~ {candidate} {prob*100:.0f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_AMARELO, 2)

    # Histórico das últimas letras dinâmicas reconhecidas
    if rec.history:
        hist = "  >  ".join(rec.history)
        cv2.putText(frame, hist, (10, h - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_BRANCO, 1)


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transformer (obrigatório — é o motivo deste módulo existir)
    tf_model, tf_cfg = load_model_from_config(
        Path(args.model), Path(args.config), "transformer_temporal", device
    )
    tf_labels = load_label_map(Path(args.label_map))
    dynamic = DynamicRecognizer(
        tf_model,
        idx_to_label={v: k for k, v in tf_labels.items()},
        device=device,
        min_conf=args.min_conf_dynamic,
        pred_every=args.pred_every,
        min_hand_frames=args.min_hand_frames,
    )
    print(f"Transformer: {tf_cfg['num_classes']} classes — {sorted(tf_labels)}")

    # MLP (opcional — sem ele o modo letra fica indisponível)
    static: StaticRecognizer | None = None
    if Path(args.static_model).exists() and Path(args.static_config).exists():
        mlp_model, mlp_cfg = load_model_from_config(
            Path(args.static_model), Path(args.static_config), "mlp_static", device
        )
        mlp_labels = load_label_map(Path(args.static_label_map))
        static = StaticRecognizer(
            mlp_model,
            idx_to_label={v: k for k, v in mlp_labels.items()},
            device=device,
            min_conf=args.min_conf_letter,
            stable_frames=args.stable_frames,
        )
        print(f"MLP: {mlp_cfg['num_classes']} classes — {sorted(mlp_labels)}")
    else:
        print("MLP não encontrado — modo letra desabilitado (só letra dinâmica).")

    mode = MODE_LETTER if (static is not None and args.start_mode == MODE_LETTER) else MODE_DYNAMIC

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

    prev_wrist: np.ndarray | None = None
    t_inicio = time.monotonic()
    last_time = time.monotonic()
    fps = 0.0

    print(f"Rodando em modo {MODE_DISPLAY[mode]}. [espaço] troca modo, [q] sai.")

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
        vec66: np.ndarray | None = None

        if result.hand_landmarks:
            raw_landmarks = result.hand_landmarks[0]
            landmarks_for_draw = raw_landmarks
            draw_landmarks(frame, raw_landmarks)

            raw_array = extract_landmarks(raw_landmarks)
            normalized, wrist_raw = normalize_landmarks(raw_array)
            if prev_wrist is None:
                prev_wrist = wrist_raw.copy()
            vec66 = build_feature_vector(normalized, wrist_raw, prev_wrist)
            prev_wrist = wrist_raw.copy()
        else:
            prev_wrist = None
            if static is not None:
                static.reset()

        # Buffer temporal é alimentado SEMPRE — trocar de modo acha janela quente.
        dynamic.push(vec66)

        if mode == MODE_LETTER and static is not None:
            if vec66 is not None:
                label, prob, stable = static.process(vec66)
                draw_letter_overlay(frame, landmarks_for_draw, label, prob, stable)
        else:
            candidate, prob, stable = None, 0.0, False
            if dynamic.should_predict():
                candidate, prob, stable = dynamic.predict()
            draw_dynamic_overlay(frame, dynamic, candidate, prob, stable)

        now = time.monotonic()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        draw_mode_hud(frame, mode, fps)

        cv2.imshow("Libras — Inferencia (letra/letra dinamica)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if static is None:
                print("Modo letra indisponível — MLP não carregado.")
            else:
                mode = MODE_DYNAMIC if mode == MODE_LETTER else MODE_LETTER
                static.reset()
                print(f"Modo: {MODE_DISPLAY[mode]}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferência ao vivo de Libras: letra estática (MLP) / letra dinâmica (Transformer)"
    )
    # Transformer (nomes seguem o comando de aceitação do roadmap)
    p.add_argument("--model", type=str, default="models/transformer_temporal.pth")
    p.add_argument("--config", type=str, default="models/transformer_temporal_config.json")
    p.add_argument("--label-map", type=str, default="models/transformer_temporal_label_map.json")
    # MLP estático (modo letra)
    p.add_argument("--static-model", type=str, default="models/mlp_static.pth")
    p.add_argument("--static-config", type=str, default="models/mlp_static_config.json")
    p.add_argument("--static-label-map", type=str, default="models/mlp_static_label_map.json")
    # Guardrails
    p.add_argument("--min-conf-letter", type=float, default=0.85,
                   help="Confiança mínima do MLP no modo letra")
    p.add_argument("--min-conf-dynamic", type=float, default=0.60,
                   help="Confiança mínima do Transformer no modo letra dinâmica")
    p.add_argument("--stable-frames", type=int, default=7,
                   help="Janela de consenso do modo letra")
    p.add_argument("--pred-every", type=int, default=5,
                   help="Frames entre predições do Transformer (~6 pred/s a 30fps)")
    p.add_argument("--min-hand-frames", type=int, default=10,
                   help="Frames contínuos com mão antes de predizer letra dinâmica")
    p.add_argument("--start-mode", type=str, choices=[MODE_LETTER, MODE_DYNAMIC],
                   default=MODE_LETTER)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
