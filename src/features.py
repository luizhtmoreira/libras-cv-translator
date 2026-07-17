"""
features.py — Extração, normalização e augmentation de landmarks
==================================================================
Fonte única de verdade para o pipeline de features.

Camadas:
    1. extract_landmarks       — MediaPipe → (63,) cru
    2. normalize_landmarks     — translação no pulso + escala pela palma (ADR-002)
    3. build_feature_vector    — 63 normalizados + 3 delta do pulso (ADR-003)     → (66,)
    4. augment_landmarks       — perturbações geométricas (ADR-006)                — TREINO
    5. build_geometric_features — features rotação-invariantes (ADR-007)          → (20,)
    6. expand_frame_for_model  — (66,) → (86,) [augment opcional + geom features]

O disco continua armazenando (30, 66). A expansão para 86 acontece no loader
(dataset.py) e no infer.py, mantendo compatibilidade com todos os .npy existentes.
"""

import numpy as np

from src.constants import (
    FINGERTIP_INDICES_ORDERED,
    MCP_INDICES_ORDERED,
    SCALE_IDX,
    WRIST_IDX,
)


# ─────────────────────────────────────────────────────────────
# CAMADA 1-3 — EXTRAÇÃO + NORMALIZAÇÃO (ADR-002 / ADR-003)
# ─────────────────────────────────────────────────────────────
def extract_landmarks(landmarks) -> np.ndarray:
    """Converte 21 landmarks do MediaPipe em array (63,) → [x0,y0,z0, x1,y1,z1, ...]."""
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normaliza landmarks e devolve o pulso bruto (para cálculo do delta).

    Retorna:
        normalized: (63,) coords normalizadas (translação + escala)
        wrist_raw:  (3,)  posição bruta do pulso antes da normalização
    """
    points = raw.reshape(21, 3)
    wrist_raw = points[WRIST_IDX].copy()

    # Translação: pulso vira origem
    points = points - wrist_raw

    # Escala: distância pulso → base do médio vira 1.0
    scale = np.linalg.norm(points[SCALE_IDX])
    if scale > 1e-6:
        points = points / scale

    return points.flatten(), wrist_raw


def build_feature_vector(
    normalized: np.ndarray,
    wrist_raw: np.ndarray,
    prev_wrist: np.ndarray,
) -> np.ndarray:
    """
    Concatena landmarks normalizados com o delta bruto do pulso.

    No primeiro frame de uma amostra, `prev_wrist == wrist_raw` → delta = [0, 0, 0].

    Retorna:
        (66,) = [63 coords normalizadas | Δx, Δy, Δz do pulso]
    """
    delta = wrist_raw - prev_wrist
    return np.concatenate([normalized, delta])


# ─────────────────────────────────────────────────────────────
# CAMADA 4 — AUGMENTATION (ADR-006)
# ─────────────────────────────────────────────────────────────
def _rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matriz de rotação 3D a partir de ângulos de Euler (radianos)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def sample_augment_params(
    rng: np.random.Generator,
    rot_range_deg: float = 15.0,
    scale_range: float = 0.10,
    translation_range: float = 0.05,
    noise_std: float = 0.01,
) -> dict:
    """
    Sorteia UMA vez os parâmetros de rotação/escala/translação da augmentation.

    Para sinais estáticos, chamamos isto por amostra (1 frame).
    Para sinais dinâmicos (sequência), chamamos isto por SEQUÊNCIA — todos os
    30 frames recebem a MESMA rotação/escala/translação. Aplicar aug diferente
    por frame destruiria a trajetória do movimento (que é o próprio sinal).

    O ruído gaussiano é mantido como parâmetro para ser aplicado por-frame no
    apply_augment_params (esse sim é observação-a-observação).
    """
    angles = np.deg2rad(rng.uniform(-rot_range_deg, rot_range_deg, size=3))
    R = _rotation_matrix_3d(float(angles[0]), float(angles[1]), float(angles[2]))
    scale = float(1.0 + rng.uniform(-scale_range, scale_range))
    trans = rng.uniform(-translation_range, translation_range, size=3).astype(np.float32)
    return {"R": R, "scale": scale, "trans": trans, "noise_std": float(noise_std)}


def apply_augment_params(
    points: np.ndarray,           # (21, 3) — já normalizados (pulso na origem)
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Aplica um conjunto de parâmetros já sorteados a um único frame de landmarks.
    Ordem: rotação → escala → translação → ruído (ruído é sempre fresco por chamada).
    """
    pts = points @ params["R"].T
    pts = pts * params["scale"]
    pts = pts + params["trans"]
    pts = pts + rng.normal(0, params["noise_std"], size=pts.shape).astype(np.float32)
    return pts.astype(np.float32)


def augment_landmarks(
    points: np.ndarray,          # (21, 3) — já normalizados (pulso na origem)
    rot_range_deg: float = 15.0,
    scale_range: float = 0.10,
    translation_range: float = 0.05,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Perturba landmarks normalizados (um frame só). Wrapper conveniente sobre
    sample_augment_params + apply_augment_params — mantido para o modo estático
    onde augment é sempre "1 sorteio por chamada".
    """
    if rng is None:
        rng = np.random.default_rng()
    params = sample_augment_params(rng, rot_range_deg, scale_range, translation_range, noise_std)
    return apply_augment_params(points, params, rng)


# ─────────────────────────────────────────────────────────────
# CAMADA 5 — FEATURES GEOMÉTRICAS (ADR-007)
# ─────────────────────────────────────────────────────────────
_TIP_PAIR_INDICES = [
    (i, j)
    for i in range(len(FINGERTIP_INDICES_ORDERED))
    for j in range(i + 1, len(FINGERTIP_INDICES_ORDERED))
]  # 10 pares


def build_geometric_features(points: np.ndarray) -> np.ndarray:
    """
    Features rotação-invariantes derivadas dos landmarks normalizados.

    Retorna (20,):
        [ 0:10] → distâncias par-a-par entre 5 pontas de dedos (discrimina V/U/D/R)
        [10:15] → distâncias ponta-de-dedo → pulso (mão aberta vs fechada)
        [15:20] → distância ponta → base do dedo (flexão de cada dedo)

    Todas as distâncias são invariantes a rotação da mão — atacam diretamente as
    confusões que features baseadas em coordenadas absolutas sofrem quando a mão
    é inclinada.
    """
    tips = points[list(FINGERTIP_INDICES_ORDERED)]   # (5, 3)
    mcps = points[list(MCP_INDICES_ORDERED)]         # (5, 3) — base de cada dedo
    wrist = points[WRIST_IDX]                        # (3,) — origem (≈0 após normalização)

    tip_pairs = np.array(
        [np.linalg.norm(tips[i] - tips[j]) for i, j in _TIP_PAIR_INDICES],
        dtype=np.float32,
    )  # (10,)
    tips_to_wrist = np.linalg.norm(tips - wrist, axis=1).astype(np.float32)  # (5,)
    flexions = np.linalg.norm(tips - mcps, axis=1).astype(np.float32)        # (5,)

    return np.concatenate([tip_pairs, tips_to_wrist, flexions])


# ─────────────────────────────────────────────────────────────
# CAMADA 6 — EXPANSÃO PARA O MODELO (juntando tudo)
# ─────────────────────────────────────────────────────────────
def expand_frame_for_model(
    frame: np.ndarray,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    aug_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Ponto único de entrada entre o feature vector (66,) armazenado em disco e
    o feature vector (86,) que o MLP consome.

    Entrada:
        frame — (66,) = [63 coords normalizadas | 3 delta do pulso]

    Fluxo:
        1. Se augment=True, perturba os 63 landmarks (rotação/escala/etc).
           O delta do pulso não sofre augmentation (é bruto, tem semântica temporal).
        2. Calcula 20 features geométricas dos landmarks (possivelmente aumentados).
        3. Concatena tudo.

    Saída:
        (86,) = [63 coords | 3 delta | 20 geométricas]
    """
    coords = frame[:63].reshape(21, 3).astype(np.float32)
    delta = frame[63:66].astype(np.float32)

    if augment:
        coords = augment_landmarks(coords, rng=rng, **(aug_kwargs or {}))

    geom = build_geometric_features(coords)
    return np.concatenate([coords.flatten(), delta, geom]).astype(np.float32)


def expand_sequence_for_model(
    tensor: np.ndarray,           # (30, 66)
    augment: bool = False,
    rng: np.random.Generator | None = None,
    aug_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Análogo ao expand_frame_for_model, mas para o tensor completo (30, 66).
    Sorteia UM único conjunto de aug params e aplica em TODOS os 30 frames
    — preservando a trajetória do movimento.

    Retorna (30, 86) = frames de [63 coords | 3 delta | 20 geométricas].
    """
    if rng is None:
        rng = np.random.default_rng()

    aug_params = None
    if augment:
        aug_params = sample_augment_params(rng, **(aug_kwargs or {}))

    T = tensor.shape[0]
    out = np.empty((T, 86), dtype=np.float32)
    for t in range(T):
        coords = tensor[t, :63].reshape(21, 3).astype(np.float32)
        delta = tensor[t, 63:66].astype(np.float32)
        if aug_params is not None:
            coords = apply_augment_params(coords, aug_params, rng)
        geom = build_geometric_features(coords)
        out[t] = np.concatenate([coords.flatten(), delta, geom])
    return out
