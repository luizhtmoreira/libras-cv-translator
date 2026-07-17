"""
constants.py — Constantes compartilhadas pelo pipeline de Libras
================================================================
Centraliza os valores usados por capture.py, capture_dynamic.py,
preprocess_external.py e infer.py. Se algum desses valores mudar,
todo o pipeline (dados novos + modelo + inferência) precisa acompanhar.
"""

# ── Caminhos padrão ──────────────────────────────────────────
MODEL_PATH = "models/hand_landmarker.task"
DATA_DIR_LETTERS = "data/raw"
DATA_DIR_DYNAMIC = "data/raw/dynamic"  # capturas de letras dinâmicas (h/j/k/x/y/z)

# ── Formato do tensor por amostra ────────────────────────────
FRAMES = 30              # janela temporal (1 segundo a ~30fps)
COORDS = 21 * 3          # 21 landmarks × (x, y, z) = 63
DELTA = 3                # Δx, Δy, Δz do pulso entre frames
FEATURE_DIM = COORDS + DELTA  # 66 valores por frame

# Frame usado pelo modelo estático (meio da janela — ver ADR-004)
STATIC_FRAME_IDX = FRAMES // 2  # 15

# ── Features geométricas extras (rotação-invariantes) ────────
# Calculadas on-the-fly em features.build_geometric_features:
#   10 distâncias par-a-par entre 5 pontas de dedos
#    5 distâncias ponta-de-dedo → pulso
#    5 métricas de flexão (ponta → base do dedo)
GEOMETRIC_DIM = 20
FEATURE_DIM_EXTENDED = FEATURE_DIM + GEOMETRIC_DIM  # 86 — o que o MLP consome

# ── Índices de landmarks do MediaPipe ────────────────────────
WRIST_IDX = 0     # pulso — origem da translação
SCALE_IDX = 9     # base do dedo médio — referência de escala
FINGERTIP_INDICES_ORDERED = (4, 8, 12, 16, 20)  # polegar, indicador, médio, anelar, mínimo
MCP_INDICES_ORDERED = (2, 5, 9, 13, 17)         # base de cada dedo (para métrica de flexão)

# ── Conexões para desenhar o esqueleto da mão ────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # médio
    (0, 13), (13, 14), (14, 15), (15, 16), # anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # mínimo
    (5, 9), (9, 13), (13, 17),             # palma
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}

# ── Cores BGR (OpenCV) ───────────────────────────────────────
COR_VERDE    = (0, 255, 0)
COR_VERMELHO = (0, 0, 255)
COR_AMARELO  = (0, 255, 255)
COR_BRANCO   = (255, 255, 255)
COR_CINZA    = (180, 180, 180)
COR_ROXO     = (255, 100, 100)
