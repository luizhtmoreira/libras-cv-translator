"""
capture.py — Pipeline de Captura de Landmarks para Libras
==========================================================
Fase 3 (completa): webcam → MediaPipe → normalização → buffer 30 frames → .npy

Feature vector por frame (66 valores):
  [0:63]  → 21 landmarks normalizados (translação + escala)
  [63:66] → Δ(x, y, z) do pulso em relação ao frame anterior
              captura a trajetória do movimento da mão

Tensor salvo: shape (30, 66), dtype float32

Como rodar:
    .venv/bin/python src/capture.py --sinal A --amostras 30

Teclas durante a captura:
    ESPAÇO → inicia gravação de 1 amostra (30 frames)
    u      → desfaz a última amostra salva (apaga o arquivo)
    q      → encerra o programa
"""

import argparse
import os
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
DATA_DIR    = "data/raw"
FRAMES      = 30          # tamanho da janela temporal
COORDS      = 21 * 3      # 21 landmarks × 3 coords (x, y, z) = 63
DELTA       = 3           # Δx, Δy, Δz do pulso = 3
FEATURE_DIM = COORDS + DELTA  # 66 valores por frame

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
COR_VERDE   = (0, 255, 0)
COR_VERMELHO = (0, 0, 255)
COR_AMARELO = (0, 255, 255)
COR_BRANCO  = (255, 255, 255)
COR_CINZA   = (180, 180, 180)


# ─────────────────────────────────────────────────────────────
# ARGPARSE — lê argumentos da linha de comando
# ─────────────────────────────────────────────────────────────
# argparse é a biblioteca padrão do Python para argumentos CLI.
# Quando você roda: python capture.py --sinal A --amostras 30
# ele lê "--sinal A" e "--amostras 30" e disponibiliza como atributos.
parser = argparse.ArgumentParser(description="Captura de landmarks para Libras")
parser.add_argument("--sinal",    required=True, help="Nome do sinal (ex: A, B, obrigado)")
parser.add_argument("--amostras", type=int, default=30, help="Número de amostras a coletar")
args = parser.parse_args()

SINAL        = args.sinal.lower()
MAX_AMOSTRAS = args.amostras

# Cria o diretório de saída se não existir
# exist_ok=True → não dá erro se a pasta já existir
output_dir = os.path.join(DATA_DIR, SINAL)
os.makedirs(output_dir, exist_ok=True)

# Descobre qual é o próximo índice de amostra (continua de onde parou)
# Usa o maior índice existente + 1 para evitar sobrescrever arquivos
# quando alguns foram deletados no meio (ex: deletou 0.npy e 1.npy,
# mas ainda existem 2.npy...49.npy → próximo deve ser 50, não 47)
existing = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
existing_count = len(existing)           # quantas amostras já existem de fato
if existing:
    next_idx = max(int(f[:-4]) for f in existing) + 1
else:
    next_idx = 0

print(f"Sinal: '{SINAL}' | Amostras já coletadas: {existing_count} | Meta: {MAX_AMOSTRAS}")
print("Pressione ESPAÇO para gravar uma amostra. 'q' para sair.")


# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO MEDIAPIPE
# ─────────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────
# FUNÇÕES DE PROCESSAMENTO
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
                    → usado para calcular o delta de movimento entre frames
    """
    points = raw.reshape(21, 3)

    # Guarda posição bruta do pulso ANTES de normalizar
    # (precisamos disso para calcular o quanto ele se moveu)
    wrist_raw = points[WRIST_IDX].copy()

    # Translação: centraliza no pulso
    points = points - wrist_raw

    # Escala: normaliza pela distância pulso → base do médio
    scale = np.linalg.norm(points[SCALE_IDX])
    if scale > 1e-6:
        points = points / scale

    return points.flatten(), wrist_raw


def build_feature_vector(normalized: np.ndarray, wrist_raw: np.ndarray,
                         prev_wrist: np.ndarray) -> np.ndarray:
    """
    Monta o vetor de features final de 66 valores para um frame.

    Args:
        normalized:  array (63,) com landmarks normalizados
        wrist_raw:   array (3,) com posição bruta do pulso neste frame
        prev_wrist:  array (3,) com posição bruta do pulso no frame anterior

    Returns:
        array (66,) = [63 coords normalizadas | Δx, Δy, Δz do pulso]
    """
    # Calcula o delta: quanto o pulso se moveu desde o frame anterior
    # No frame 0, prev_wrist == wrist_raw, então delta = [0, 0, 0]
    delta = wrist_raw - prev_wrist  # array (3,)

    # np.concatenate junta dois arrays em sequência
    return np.concatenate([normalized, delta])  # shape (66,)


# ─────────────────────────────────────────────────────────────
# FUNÇÕES DE DESENHO
# ─────────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], COR_CINZA, 2)
    for i, (px, py) in enumerate(points):
        color = COR_VERMELHO if i in [4, 8, 12, 16, 20] else COR_VERDE
        cv2.circle(frame, (px, py), 5, color, -1)


def draw_hud(frame, gravando: bool, frames_gravados: int, amostras_salvas: int, ultima_amostra: str | None):
    """
    HUD = Heads-Up Display. Informações sobrepostas na tela.
    Mostra o status atual da gravação.
    """
    h, w, _ = frame.shape

    # ── Barra de progresso da gravação ───────────────────────
    if gravando:
        progresso = frames_gravados / FRAMES
        bar_w     = int(w * 0.6)
        bar_x     = w // 2 - bar_w // 2
        bar_y     = h - 50

        # Fundo da barra (cinza)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), COR_CINZA, -1)
        # Preenchimento (verde proporcional ao progresso)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progresso), bar_y + 20), COR_VERDE, -1)

        # Texto de status
        txt = f"GRAVANDO... {frames_gravados}/{FRAMES} frames"
        cv2.putText(frame, txt, (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERDE, 2)
    else:
        instrucao = "Posicione a mao e pressione ESPACO para gravar"
        cv2.putText(frame, instrucao, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COR_AMARELO, 1)

    # ── Info no canto superior esquerdo ──────────────────────
    cv2.putText(frame, f"Sinal: {SINAL.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_BRANCO, 2)
    cv2.putText(frame, f"Amostras: {amostras_salvas}/{MAX_AMOSTRAS}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_BRANCO, 1)

    # ── Dica de undo ─────────────────────────────────────────
    if ultima_amostra and not gravando:
        cv2.putText(frame, "[u] Desfazer ultima amostra", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # ── Aviso de conclusão ────────────────────────────────────
    if amostras_salvas >= MAX_AMOSTRAS:
        msg = "META ATINGIDA! Pressione 'q' para sair."
        cv2.putText(frame, msg, (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_AMARELO, 2)


# ─────────────────────────────────────────────────────────────
# ESTADO DA GRAVAÇÃO
# ─────────────────────────────────────────────────────────────
gravando        = False          # True enquanto estamos acumulando frames
buffer          = []             # lista de arrays (66,) — acumula os 30 frames
prev_wrist      = None           # posição do pulso no frame anterior
amostras_salvas = existing_count # conta real de arquivos no disco
next_file_idx   = next_idx       # índice usado para nomear o próximo arquivo
ultima_amostra  = None           # caminho do último .npy salvo (para undo)


# ─────────────────────────────────────────────────────────────
# INICIALIZAÇÃO DA WEBCAM
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam.")


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(mp_image)

    mao_detectada = bool(result.hand_landmarks)

    if mao_detectada:
        raw_landmarks = result.hand_landmarks[0]
        draw_landmarks(frame, raw_landmarks)

        raw_array               = extract_landmarks(raw_landmarks)
        normalized, wrist_raw   = normalize_landmarks(raw_array)

        # No primeiro frame com mão visível, inicializa prev_wrist
        if prev_wrist is None:
            prev_wrist = wrist_raw.copy()

        feature_vector = build_feature_vector(normalized, wrist_raw, prev_wrist)
        prev_wrist     = wrist_raw.copy()  # atualiza para o próximo frame

        # ── Acumula frames se estiver gravando ───────────────
        if gravando:
            buffer.append(feature_vector)  # adiciona array (66,) ao buffer

            # Quando o buffer enche (30 frames), salva o tensor
            if len(buffer) >= FRAMES:
                # np.stack transforma lista de 30 arrays (66,) em tensor (30, 66)
                tensor = np.stack(buffer)  # shape (30, 66)

                # Salva em data/raw/{sinal}/{idx}.npy
                filepath = os.path.join(output_dir, f"{next_file_idx}.npy")
                np.save(filepath, tensor)

                next_file_idx   += 1
                amostras_salvas += 1
                ultima_amostra   = filepath  # guarda para possível undo
                print(f"  Salvo: {filepath} | shape={tensor.shape} | dtype={tensor.dtype}")
                print(f"  (Pressione 'u' para desfazer esta amostra)")

                # Reseta o estado de gravação
                buffer     = []
                gravando   = False
                prev_wrist = None  # reseta delta para a próxima amostra

    else:
        # Se a mão sumiu, reseta o delta (evita delta espúrio ao reaparecer)
        prev_wrist = None
        if gravando:
            # Se a mão sumiu durante a gravação, descarta e avisa
            print("  Mao perdida durante gravacao — amostra descartada.")
            buffer   = []
            gravando = False

    # ── HUD ──────────────────────────────────────────────────
    draw_hud(frame, gravando, len(buffer), amostras_salvas, ultima_amostra)

    # Indica se a mão NÃO está na tela (aviso vermelho)
    if not mao_detectada and not gravando:
        cv2.putText(frame, "MAO NAO DETECTADA", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_VERMELHO, 2)

    cv2.imshow(f"Captura Libras — {SINAL.upper()}", frame)

    # ── Teclado ───────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Encerrando...")
        break

    # ESPAÇO (ASCII 32) → inicia gravação se não estiver gravando
    # e se a mão estiver visível e ainda não atingiu a meta
    elif key == ord("u"):
        # ── Undo: apaga a última amostra salva ───────────────
        if gravando:
            print("  Nao e possivel desfazer durante a gravacao.")
        elif ultima_amostra is None:
            print("  Nenhuma amostra para desfazer.")
        elif not os.path.exists(ultima_amostra):
            print("  Arquivo ja foi removido.")
        else:
            os.remove(ultima_amostra)
            amostras_salvas -= 1
            print(f"  DESFEITO: {ultima_amostra} removido. Contador voltou para {amostras_salvas}.")
            ultima_amostra = None  # só permite desfazer uma vez

    elif key == 32:
        if not mao_detectada:
            print("  Mao nao detectada. Posicione a mao antes de gravar.")
        elif gravando:
            print("  Ja esta gravando...")
        elif amostras_salvas >= MAX_AMOSTRAS:
            print("  Meta atingida! Pressione 'q' para sair.")
        else:
            print(f"  Iniciando gravacao da amostra {amostras_salvas}...")
            gravando       = True
            buffer         = []
            prev_wrist     = None
            ultima_amostra = None  # nova gravação limpa o undo anterior


# ─────────────────────────────────────────────────────────────
# LIMPEZA
# ─────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
landmarker.close()
print(f"\nSessão encerrada. Amostras salvas nesta sessão: {amostras_salvas - existing_count} (total no disco: {amostras_salvas})")
