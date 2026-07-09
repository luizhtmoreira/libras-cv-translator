"""
capture_dynamic.py — Pipeline de Captura para Sinais Dinâmicos (Palavras/Expressões)
======================================================================================
Diferenças em relação ao capture.py (letras):

1. RunningMode.VIDEO  → MediaPipe mantém um tracker entre frames.
   Em movimentos rápidos (ex: "Obrigado") a mão não é perdida, pois o
   detector usa o frame anterior como âncora — muito mais robusto que IMAGE.

2. Grace period (3 frames) → se a mão desaparecer brevemente durante a
   gravação, o script congela o último frame válido em vez de cancelar.
   Evita descartar amostras por oclusão momentânea no meio do sinal.

3. Saída em data/raw/words/{sinal}/ — separado das letras para não
   misturar os datasets (letras e palavras terão classes distintas no modelo).

Feature vector por frame (66 valores) — idêntico ao capture.py:
  [0:63]  → 21 landmarks normalizados (translação + escala)
  [63:66] → Δ(x, y, z) do pulso em relação ao frame anterior

Tensor salvo: shape (30, 66), dtype float32

Como rodar:
    .venv/bin/python src/capture_dynamic.py --sinal obrigado --amostras 50

Teclas durante a captura:
    ESPAÇO → inicia gravação de 1 amostra (30 frames)
    u      → desfaz a última amostra salva (apaga o arquivo)
    q      → encerra o programa
"""

import argparse
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "models/hand_landmarker.task"
DATA_DIR    = "data/raw/words"   # separado das letras
FRAMES      = 30                 # janela temporal (igual às letras → mesmo modelo)
COORDS      = 21 * 3             # 21 landmarks × 3 coords = 63
DELTA       = 3                  # Δx, Δy, Δz do pulso
FEATURE_DIM = COORDS + DELTA     # 66 valores por frame
GRACE_PERIOD = 3                 # frames de tolerância para perda momentânea da mão

WRIST_IDX = 0
SCALE_IDX = 9

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Cores (BGR)
COR_VERDE    = (0, 255, 0)
COR_VERMELHO = (0, 0, 255)
COR_AMARELO  = (0, 255, 255)
COR_BRANCO   = (255, 255, 255)
COR_CINZA    = (180, 180, 180)
COR_ROXO     = (255, 100, 100)   # indica grace period ativo


# ─────────────────────────────────────────────────────────────
# ARGPARSE
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Captura de sinais dinâmicos para Libras")
parser.add_argument("--sinal",    required=True, help="Nome do sinal (ex: obrigado, desculpa)")
parser.add_argument("--amostras", type=int, default=50, help="Número de amostras a coletar")
args = parser.parse_args()

SINAL        = args.sinal.lower()
MAX_AMOSTRAS = args.amostras

output_dir = os.path.join(DATA_DIR, SINAL)
os.makedirs(output_dir, exist_ok=True)

existing = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
existing_count = len(existing)
next_idx = max((int(f[:-4]) for f in existing), default=-1) + 1

print(f"Sinal: '{SINAL}' | Amostras já coletadas: {existing_count} | Meta: {MAX_AMOSTRAS}")
print(f"Modo: VIDEO (tracking entre frames) | Grace period: {GRACE_PERIOD} frames")
print("Pressione ESPAÇO para gravar uma amostra. 'q' para sair.")


# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO MEDIAPIPE — VIDEO MODE
# ─────────────────────────────────────────────────────────────
# RunningMode.VIDEO difere do IMAGE porque:
# - Recebe timestamps em ms (landmarker.detect_for_video)
# - Mantém estado interno entre chamadas (tracker Kalman)
# - Muito mais robusto para gestos rápidos/dinâmicos
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.4,  # mais permissivo que IMAGE — tracker ajuda
    min_tracking_confidence=0.4,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────
# FUNÇÕES DE PROCESSAMENTO (idênticas ao capture.py)
# ─────────────────────────────────────────────────────────────
def extract_landmarks(landmarks) -> np.ndarray:
    """Converte 21 landmarks em array (63,) com [x0,y0,z0, x1,y1,z1, ...]."""
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normaliza os landmarks e retorna também as coordenadas brutas do pulso.

    Retorna:
        normalized: array (63,) com coords normalizadas (translação + escala)
        wrist_raw:  array (3,) com [x, y, z] do pulso ANTES da normalização
    """
    points = raw.reshape(21, 3)
    wrist_raw = points[WRIST_IDX].copy()
    points = points - wrist_raw
    scale = np.linalg.norm(points[SCALE_IDX])
    if scale > 1e-6:
        points = points / scale
    return points.flatten(), wrist_raw


def build_feature_vector(normalized: np.ndarray, wrist_raw: np.ndarray,
                         prev_wrist: np.ndarray) -> np.ndarray:
    """Monta o vetor de features final de 66 valores para um frame."""
    delta = wrist_raw - prev_wrist
    return np.concatenate([normalized, delta])


# ─────────────────────────────────────────────────────────────
# FUNÇÕES DE DESENHO
# ─────────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks, grace_ativo: bool = False):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    cor_conexao = COR_ROXO if grace_ativo else COR_CINZA
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], cor_conexao, 2)
    for i, (px, py) in enumerate(points):
        color = COR_VERMELHO if i in [4, 8, 12, 16, 20] else COR_VERDE
        cv2.circle(frame, (px, py), 5, color, -1)


def draw_hud(frame, gravando: bool, frames_gravados: int,
             amostras_salvas: int, ultima_amostra: str | None,
             grace_frames: int):
    h, w, _ = frame.shape

    # ── Barra de progresso ───────────────────────────────────
    if gravando:
        progresso = frames_gravados / FRAMES
        bar_w = int(w * 0.6)
        bar_x = w // 2 - bar_w // 2
        bar_y = h - 50

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), COR_CINZA, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progresso), bar_y + 20), COR_VERDE, -1)

        txt = f"GRAVANDO... {frames_gravados}/{FRAMES} frames"
        cv2.putText(frame, txt, (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)

        # Aviso de grace period ativo
        if grace_frames > 0:
            grace_txt = f"[sem mao — congelando {grace_frames}/{GRACE_PERIOD}]"
            cv2.putText(frame, grace_txt, (bar_x, bar_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_ROXO, 1)
    else:
        instrucao = "Execute o sinal e pressione ESPACO para gravar"
        cv2.putText(frame, instrucao, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COR_AMARELO, 1)

    # ── Info superior ────────────────────────────────────────
    cv2.putText(frame, f"Sinal: {SINAL.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_BRANCO, 2)
    cv2.putText(frame, f"Amostras: {amostras_salvas}/{MAX_AMOSTRAS}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_BRANCO, 1)
    cv2.putText(frame, "Modo: VIDEO", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_AMARELO, 1)

    # ── Undo ─────────────────────────────────────────────────
    if ultima_amostra and not gravando:
        cv2.putText(frame, "[u] Desfazer ultima amostra", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # ── Meta atingida ────────────────────────────────────────
    if amostras_salvas >= MAX_AMOSTRAS:
        msg = "META ATINGIDA! Pressione 'q' para sair."
        cv2.putText(frame, msg, (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_AMARELO, 2)


# ─────────────────────────────────────────────────────────────
# ESTADO DA GRAVAÇÃO
# ─────────────────────────────────────────────────────────────
gravando        = False
buffer          = []           # acumula arrays (66,)
prev_wrist      = None
ultimo_fv       = None         # último feature vector válido (para grace period)
ultimos_lm      = None         # últimos landmarks válidos (para desenho no grace)
grace_contador  = 0            # quantos frames consecutivos sem mão durante gravação
amostras_salvas = existing_count
next_file_idx   = next_idx
ultima_amostra  = None

# ─────────────────────────────────────────────────────────────
# WEBCAM
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam.")

# Timestamp de início para o modo VIDEO (MediaPipe exige ms monotônicos)
t_inicio = time.monotonic()


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Timestamp em ms para o RunningMode.VIDEO
    timestamp_ms = int((time.monotonic() - t_inicio) * 1000)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    # detect_for_video em vez de detect — usa o tracker interno
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    mao_detectada = bool(result.hand_landmarks)

    if mao_detectada:
        grace_contador = 0   # mão voltou — reseta grace
        raw_landmarks  = result.hand_landmarks[0]
        ultimos_lm     = raw_landmarks
        draw_landmarks(frame, raw_landmarks, grace_ativo=False)

        raw_array             = extract_landmarks(raw_landmarks)
        normalized, wrist_raw = normalize_landmarks(raw_array)

        if prev_wrist is None:
            prev_wrist = wrist_raw.copy()

        feature_vector = build_feature_vector(normalized, wrist_raw, prev_wrist)
        prev_wrist     = wrist_raw.copy()
        ultimo_fv      = feature_vector

        if gravando:
            buffer.append(feature_vector)

    else:
        # ── Mão não detectada ────────────────────────────────
        if gravando and ultimo_fv is not None and grace_contador < GRACE_PERIOD:
            # Grace period: congela o último frame válido
            grace_contador += 1
            buffer.append(ultimo_fv)   # repete o último frame
            if ultimos_lm:
                draw_landmarks(frame, ultimos_lm, grace_ativo=True)
        elif gravando:
            # Grace esgotado → descarta
            print(f"  Mao perdida por mais de {GRACE_PERIOD} frames — amostra descartada.")
            buffer        = []
            gravando      = False
            prev_wrist    = None
            ultimo_fv     = None
            ultimos_lm    = None
            grace_contador = 0
        else:
            # Fora de gravação — apenas reseta o delta
            prev_wrist = None

    # ── Verifica se o buffer encheu ──────────────────────────
    if gravando and len(buffer) >= FRAMES:
        tensor = np.stack(buffer)   # (30, 66)
        filepath = os.path.join(output_dir, f"{next_file_idx}.npy")
        np.save(filepath, tensor)

        next_file_idx   += 1
        amostras_salvas += 1
        ultima_amostra   = filepath
        print(f"  Salvo: {filepath} | shape={tensor.shape} | dtype={tensor.dtype}")
        print(f"  (Pressione 'u' para desfazer esta amostra)")

        buffer        = []
        gravando      = False
        prev_wrist    = None
        ultimo_fv     = None
        ultimos_lm    = None
        grace_contador = 0

    # ── HUD ──────────────────────────────────────────────────
    draw_hud(frame, gravando, len(buffer), amostras_salvas, ultima_amostra, grace_contador)

    if not mao_detectada and grace_contador == 0 and not gravando:
        cv2.putText(frame, "MAO NAO DETECTADA", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERMELHO, 2)

    cv2.imshow(f"Captura Libras DINAMICO — {SINAL.upper()}", frame)

    # ── Teclado ───────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Encerrando...")
        break

    elif key == ord("u"):
        if gravando:
            print("  Nao e possivel desfazer durante a gravacao.")
        elif ultima_amostra is None:
            print("  Nenhuma amostra para desfazer.")
        elif not os.path.exists(ultima_amostra):
            print("  Arquivo ja foi removido.")
        else:
            os.remove(ultima_amostra)
            amostras_salvas -= 1
            print(f"  DESFEITO: {ultima_amostra} removido. Contador: {amostras_salvas}.")
            ultima_amostra = None

    elif key == 32:  # ESPAÇO
        if not mao_detectada and grace_contador == 0:
            print("  Mao nao detectada. Posicione a mao antes de gravar.")
        elif gravando:
            print("  Ja esta gravando...")
        elif amostras_salvas >= MAX_AMOSTRAS:
            print("  Meta atingida! Pressione 'q' para sair.")
        else:
            print(f"  Iniciando gravacao da amostra {amostras_salvas}...")
            gravando      = True
            buffer        = []
            prev_wrist    = None
            ultimo_fv     = None
            ultimos_lm    = None
            grace_contador = 0
            ultima_amostra = None


# ─────────────────────────────────────────────────────────────
# LIMPEZA
# ─────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
landmarker.close()
print(f"\nSessão encerrada. Amostras salvas nesta sessão: {amostras_salvas - existing_count} "
      f"(total no disco: {amostras_salvas})")
