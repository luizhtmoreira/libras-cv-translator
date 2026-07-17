"""
capture.py — Pipeline de Captura de Landmarks para Libras
==========================================================
Fase 3 (completa): webcam → MediaPipe → normalização → buffer 30 frames → .npy

Feature vector por frame (66 valores):
  [0:63]  → 21 landmarks normalizados (translação + escala)
  [63:66] → Δ(x, y, z) do pulso em relação ao frame anterior
              captura a trajetória do movimento da mão

Tensor salvo: shape (30, 66), dtype float32

Dois modos de captura (--modo):

  burst (default) — otimizado para letras ESTÁTICAS: ESPAÇO dispara uma rajada
      de N fotos (default 5) com intervalo de 1s entre elas. Cada foto vira uma
      amostra (30, 66) com o frame replicado 30x e delta zero — exatamente o
      formato que o preprocess do Brazilian Alphabet gera (imagem estática).
      Fluxo recomendado: uma rajada por pose; entre rajadas, varie ângulo,
      distância e iluminação.

  janela — modo legado: ESPAÇO grava 30 frames contínuos (~1s) em uma amostra.

Como rodar:
    python -m src.capture --sinal A --amostras 40
    python -m src.capture --sinal A --amostras 40 --fotos 5 --intervalo 1.0
    python -m src.capture --sinal A --modo janela

Teclas durante a captura:
    ESPAÇO → dispara a rajada (burst) ou grava 30 frames (janela)
    u      → desfaz a última rajada/amostra salva (apaga os arquivos)
    q      → encerra o programa
"""

import argparse
import os
import sys
import time

# Permite rodar como `python src/capture.py` (o diretório do script vai para o sys.path,
# mas precisamos do repo root para importar `src.features` etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

from src.constants import (
    COR_AMARELO,
    COR_BRANCO,
    COR_CINZA,
    COR_VERDE,
    COR_VERMELHO,
    DATA_DIR_LETTERS as DATA_DIR,
    FINGERTIP_INDICES,
    FRAMES,
    HAND_CONNECTIONS,
    MODEL_PATH,
)
from src.features import build_feature_vector, extract_landmarks, normalize_landmarks


# ─────────────────────────────────────────────────────────────
# ARGPARSE — lê argumentos da linha de comando
# ─────────────────────────────────────────────────────────────
# argparse é a biblioteca padrão do Python para argumentos CLI.
# Quando você roda: python capture.py --sinal A --amostras 30
# ele lê "--sinal A" e "--amostras 30" e disponibiliza como atributos.
parser = argparse.ArgumentParser(description="Captura de landmarks para Libras")
parser.add_argument("--sinal",    required=True, help="Nome do sinal (ex: A, B, obrigado)")
parser.add_argument("--amostras", type=int, default=30, help="Número de amostras a coletar")
parser.add_argument("--modo", choices=["burst", "janela"], default="burst",
                    help="burst: ESPAÇO tira N fotos espaçadas (letras estáticas). "
                         "janela: ESPAÇO grava 30 frames contínuos (modo legado)")
parser.add_argument("--fotos", type=int, default=5,
                    help="Fotos por rajada no modo burst")
parser.add_argument("--intervalo", type=float, default=1.0,
                    help="Segundos entre fotos da rajada")
args = parser.parse_args()

SINAL        = args.sinal.lower()
MAX_AMOSTRAS = args.amostras
MODO         = args.modo
BURST_FOTOS  = args.fotos
BURST_INTERVALO = args.intervalo

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
if MODO == "burst":
    print(f"Modo BURST: ESPAÇO tira {BURST_FOTOS} fotos com {BURST_INTERVALO:.0f}s de "
          f"intervalo. Varie ângulo/iluminação entre rajadas. 'q' para sair.")
else:
    print("Pressione ESPAÇO para gravar uma amostra (30 frames). 'q' para sair.")


# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO MEDIAPIPE
# ─────────────────────────────────────────────────────────────
# VIDEO: detecta a palma uma vez e rastreia entre frames (IMAGE redetecta
# do zero a cada frame e ignora min_tracking_confidence). Mesmos limiares
# do capture_dynamic/preprocess_external — captura e treino consistentes.
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────
# FUNÇÕES DE DESENHO
# ─────────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], COR_CINZA, 2)
    for i, (px, py) in enumerate(points):
        color = COR_VERMELHO if i in FINGERTIP_INDICES else COR_VERDE
        cv2.circle(frame, (px, py), 5, color, -1)


def draw_burst_hud(frame, fotos_tiradas: int, proxima_em: float, flash: bool):
    """HUD do modo burst: contagem da rajada + countdown até a próxima foto."""
    h, w, _ = frame.shape

    # Flash branco por ~0.15s logo após cada foto — feedback de "clique"
    if flash:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COR_BRANCO, 18)

    txt = f"FOTO {fotos_tiradas}/{BURST_FOTOS}"
    cv2.putText(frame, txt, (w // 2 - 90, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COR_VERDE, 3)

    # Countdown grande até a próxima foto (segundos restantes)
    if proxima_em > 0:
        cv2.putText(frame, f"{proxima_em:.1f}s", (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, COR_AMARELO, 3)

    # Barra de progresso da rajada
    bar_w = int(w * 0.6)
    bar_x = w // 2 - bar_w // 2
    bar_y = h - 50
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), COR_CINZA, -1)
    fill = int(bar_w * fotos_tiradas / BURST_FOTOS)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 20), COR_VERDE, -1)


def draw_hud(frame, gravando: bool, frames_gravados: int, amostras_salvas: int, ultima_amostra: str | None):
    """
    HUD = Heads-Up Display. Informações sobrepostas na tela.
    Mostra o status atual da gravação.
    """
    h, w, _ = frame.shape

    # ── Barra de progresso da gravação (modo janela) ─────────
    if gravando and MODO == "janela":
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
    elif not gravando:
        if MODO == "burst":
            instrucao = f"ESPACO dispara {BURST_FOTOS} fotos ({BURST_INTERVALO:.0f}s entre elas) — varie a pose entre rajadas"
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
        rotulo = "rajada" if (MODO == "burst" and len(ultima_amostra) > 1) else "amostra"
        cv2.putText(frame, f"[u] Desfazer ultima {rotulo}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # ── Aviso de conclusão ────────────────────────────────────
    if amostras_salvas >= MAX_AMOSTRAS:
        msg = "META ATINGIDA! Pressione 'q' para sair."
        cv2.putText(frame, msg, (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_AMARELO, 2)


# ─────────────────────────────────────────────────────────────
# ESTADO DA GRAVAÇÃO
# ─────────────────────────────────────────────────────────────
gravando        = False          # True enquanto estamos acumulando frames (modo janela)
buffer          = []             # lista de arrays (66,) — acumula os 30 frames
prev_wrist      = None           # posição do pulso no frame anterior
amostras_salvas = existing_count # conta real de arquivos no disco
next_file_idx   = next_idx       # índice usado para nomear o próximo arquivo
ultima_amostra  = None           # lista de caminhos salvos no último disparo (para undo)

# Estado do modo burst
burst_restantes  = 0             # fotos que ainda faltam na rajada atual
proxima_foto_em  = 0.0           # time.monotonic() da próxima foto
flash_ate        = 0.0           # feedback visual de "clique" até este instante


def salvar_amostra_estatica(feature_vector) -> str:
    """
    Salva UMA foto como amostra estática (30, 66): frame replicado 30x.
    O delta do pulso já vem zerado (passamos wrist == prev_wrist na chamada) —
    mesmo formato que o preprocess do Brazilian Alphabet gera para imagens.
    """
    global next_file_idx, amostras_salvas
    tensor = np.tile(feature_vector, (FRAMES, 1)).astype(np.float32)  # (30, 66)
    filepath = os.path.join(output_dir, f"{next_file_idx}.npy")
    np.save(filepath, tensor)
    next_file_idx   += 1
    amostras_salvas += 1
    return filepath


# ─────────────────────────────────────────────────────────────
# INICIALIZAÇÃO DA WEBCAM
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
t_inicio = time.monotonic()


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
    timestamp_ms = int((time.monotonic() - t_inicio) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

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

        # ── Modo burst: tira a foto quando chega a hora ──────
        if burst_restantes > 0 and time.monotonic() >= proxima_foto_em:
            # Foto = frame atual com delta ZERO (sinal estático) — recalcula
            # o vetor com prev == atual em vez de reusar o delta do stream.
            vec_estatico = build_feature_vector(normalized, wrist_raw, wrist_raw)
            filepath = salvar_amostra_estatica(vec_estatico)
            ultima_amostra.append(filepath)
            burst_restantes -= 1
            flash_ate = time.monotonic() + 0.15
            proxima_foto_em = time.monotonic() + BURST_INTERVALO
            n = len(ultima_amostra)
            print(f"  Foto {n}/{BURST_FOTOS} salva: {filepath}")
            if burst_restantes == 0:
                print(f"  Rajada completa ({n} amostras). Mude a pose e pressione ESPACO. [u] desfaz a rajada.")

        # ── Acumula frames se estiver gravando (modo janela) ─
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
                ultima_amostra   = [filepath]  # guarda para possível undo
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
        if burst_restantes > 0:
            # Rajada em pausa: empurra o timer para frente até a mão voltar,
            # em vez de fotografar um frame sem mão.
            proxima_foto_em = time.monotonic() + BURST_INTERVALO

    # ── HUD ──────────────────────────────────────────────────
    draw_hud(frame, gravando, len(buffer), amostras_salvas, ultima_amostra)
    if burst_restantes > 0:
        fotos_tiradas = BURST_FOTOS - burst_restantes
        draw_burst_hud(
            frame,
            fotos_tiradas,
            max(0.0, proxima_foto_em - time.monotonic()),
            flash=time.monotonic() < flash_ate,
        )
    elif time.monotonic() < flash_ate:
        # Flash da última foto da rajada ainda visível
        draw_burst_hud(frame, BURST_FOTOS, 0.0, flash=True)

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
        # ── Undo: apaga a última rajada/amostra salva ─────────
        if gravando or burst_restantes > 0:
            print("  Nao e possivel desfazer durante a gravacao.")
        elif not ultima_amostra:
            print("  Nenhuma amostra para desfazer.")
        else:
            removidos = 0
            for path in ultima_amostra:
                if os.path.exists(path):
                    os.remove(path)
                    removidos += 1
            amostras_salvas -= removidos
            print(f"  DESFEITO: {removidos} arquivo(s) removido(s). Contador voltou para {amostras_salvas}.")
            ultima_amostra = None  # só permite desfazer uma vez

    elif key == 32:
        if not mao_detectada:
            print("  Mao nao detectada. Posicione a mao antes de gravar.")
        elif gravando or burst_restantes > 0:
            print("  Ja esta gravando...")
        elif amostras_salvas >= MAX_AMOSTRAS:
            print("  Meta atingida! Pressione 'q' para sair.")
        elif MODO == "burst":
            print(f"  Rajada iniciada: {BURST_FOTOS} fotos, {BURST_INTERVALO:.0f}s entre elas. Segure a pose!")
            burst_restantes = BURST_FOTOS
            proxima_foto_em = time.monotonic()  # primeira foto imediata
            ultima_amostra  = []                # acumula os paths da rajada
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
