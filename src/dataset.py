"""
dataset.py — Loader PyTorch para tensores (30, 66) de Libras
============================================================

Layout de disco esperado (um ou mais roots):

    <root_dir>/
        <label_1>/
            0.npy   → array (30, 66) float32
            1.npy
            ...
        <label_2>/
            ...

Suporta múltiplos roots (`root_dirs=[...]`) para mesclar dataset externo (Bianka)
com capturas próprias do usuário (data/raw/). Ver ADR-005.

Dois modos de leitura, alinhados com o estudo de ablação (ADR-004):
    - mode="static":   frame do meio → (66,) → EXPANDIDO para (86,) via features.expand_frame_for_model
    - mode="sequence": tensor completo → (30, 66)  (sem expansão — modelo temporal ainda pendente)

Augmentation (ADR-006) só é aplicada quando `augment=True` (usa-se no split de treino,
NÃO no de validação — mesmo que compartilhem o mesmo arquivo em disco).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from src.constants import FEATURE_DIM, FRAMES, STATIC_FRAME_IDX
from src.features import expand_frame_for_model, expand_sequence_for_model

Mode = Literal["static", "sequence"]


def build_label_map(root_dirs: str | os.PathLike | list) -> dict[str, int]:
    """
    Mapeia nomes de classe (subpastas) para índices inteiros, em ordem lexicográfica.
    Aceita 1 root ou uma lista — a união das subpastas define as classes.
    """
    if isinstance(root_dirs, (str, os.PathLike)):
        root_dirs = [root_dirs]
    labels: set[str] = set()
    for root in root_dirs:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Diretório de dados não existe: {root_path}")
        labels.update(p.name for p in root_path.iterdir() if p.is_dir())
    if not labels:
        raise ValueError(f"Nenhuma subpasta de classe encontrada em {root_dirs}")
    return {label: idx for idx, label in enumerate(sorted(labels))}


def save_label_map(label_map: dict[str, int], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def load_label_map(path: str | os.PathLike) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class LibrasDataset(Dataset):
    """
    Dataset de tensores (30, 66) salvos em .npy.

    Args:
        root_dirs:  raiz (str/Path) ou lista de raízes com subpastas por classe.
                    Múltiplos roots são mesclados por label (Bianka + data/raw/, etc).
        mode:       "static" → (86,) do frame do meio (63 landmarks + 3 delta + 20 geom)
                    "sequence" → (30, 66) sem expansão
        label_map:  se None, é construído varrendo os discos. Passe explicitamente
                    para garantir que treino e validação/inferência usem os mesmos índices.
        augment:    se True, aplica augment_landmarks nos 63 landmarks antes de calcular
                    features geométricas. Só faz sentido no split de treino.
        aug_kwargs: dict passado para augment_landmarks (rot_range_deg, scale_range, etc).
        rng_seed:   opcional; se dado, o RNG interno usa essa seed para reprodutibilidade.
        max_per_class: teto de arquivos por classe. Classes acima do teto são
                    subamostradas com seed própria (subsample_seed) — determinístico,
                    para que dois datasets sobre os mesmos roots vejam a MESMA
                    listagem (requisito do stratified_split). None = sem teto.
        subsample_seed: seed do RNG da subamostragem (independente do de augment).
        no_cap_dirs: roots cujas amostras NUNCA são subamostradas (ex: data/raw —
                    domínio da webcam alvo). O teto vale só para os demais roots;
                    o total da classe pode exceder o teto pelo tanto protegido.
    """

    def __init__(
        self,
        root_dirs: str | os.PathLike | list,
        mode: Mode = "static",
        label_map: dict[str, int] | None = None,
        augment: bool = False,
        aug_kwargs: dict | None = None,
        rng_seed: int | None = None,
        oversample_factor: int = 1,
        max_per_class: int | None = None,
        subsample_seed: int = 42,
        no_cap_dirs: list | None = None,
    ):
        if mode not in ("static", "sequence"):
            raise ValueError(f"mode deve ser 'static' ou 'sequence', recebido: {mode!r}")
        if oversample_factor < 1:
            raise ValueError(f"oversample_factor deve ser ≥ 1, recebido: {oversample_factor}")
        if max_per_class is not None and max_per_class < 1:
            raise ValueError(f"max_per_class deve ser ≥ 1 ou None, recebido: {max_per_class}")

        if isinstance(root_dirs, (str, os.PathLike)):
            root_dirs = [root_dirs]
        self.root_dirs = [Path(r) for r in root_dirs]
        self.mode = mode
        self.augment = augment
        self.aug_kwargs = aug_kwargs or {}
        self._rng = np.random.default_rng(rng_seed)
        self.label_map = label_map or build_label_map(self.root_dirs)
        # Oversample só faz sentido com augment=True: replica o dataset lógica
        # (não física) e cada visita gera aug diferente.
        self.oversample_factor = oversample_factor if augment else 1

        self.max_per_class = max_per_class
        no_cap_resolved = {Path(d).resolve() for d in (no_cap_dirs or [])}
        # RNG dedicado da subamostragem: consumido na ordem do label_map (fixa),
        # logo dois datasets com mesmos roots/label_map/seed selecionam os
        # MESMOS arquivos — pré-condição do stratified_split(val_dataset=...).
        cap_rng = np.random.default_rng(subsample_seed)

        # Varre todos os roots e concatena — mantém só paths + label idx.
        self.samples: list[tuple[Path, int]] = []
        for label, idx in self.label_map.items():
            protected: list[Path] = []   # roots em no_cap_dirs — nunca subamostrados
            cappable: list[Path] = []
            for root in self.root_dirs:
                class_dir = root / label
                if not class_dir.is_dir():
                    continue
                bucket = protected if root.resolve() in no_cap_resolved else cappable
                bucket.extend(sorted(class_dir.glob("*.npy")))
            if max_per_class is not None and len(cappable) > max_per_class:
                keep = cap_rng.choice(len(cappable), size=max_per_class, replace=False)
                cappable = [cappable[i] for i in sorted(keep)]
            for npy in protected + cappable:
                self.samples.append((npy, idx))

        if not self.samples:
            raise ValueError(f"Nenhum .npy encontrado sob {self.root_dirs}")

    def __len__(self) -> int:
        # Comprimento efetivo — oversample aumenta o número de "amostras" que o
        # DataLoader vê por epoch. Cada índice além do base mapeia de volta pro
        # mesmo arquivo, mas com augment aplicado com RNG independente.
        return len(self.samples) * self.oversample_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Reduz idx expandido → idx de arquivo (oversample)
        base_idx = idx % len(self.samples)
        path, label = self.samples[base_idx]
        tensor = np.load(path)

        if tensor.shape != (FRAMES, FEATURE_DIM):
            raise ValueError(
                f"Shape inesperado em {path}: {tensor.shape}, esperado {(FRAMES, FEATURE_DIM)}"
            )

        if self.mode == "static":
            frame66 = tensor[STATIC_FRAME_IDX]  # (66,)
            data = expand_frame_for_model(
                frame66,
                augment=self.augment,
                rng=self._rng,
                aug_kwargs=self.aug_kwargs,
            )  # (86,)
        else:
            # Sequence: expande TODOS os 30 frames com a MESMA aug transform
            # (rotação/escala/translação sorteadas uma vez por sequência).
            # Preserva a trajetória do movimento — que é o sinal em si.
            data = expand_sequence_for_model(
                tensor,
                augment=self.augment,
                rng=self._rng,
                aug_kwargs=self.aug_kwargs,
            )  # (30, 86)

        return torch.from_numpy(data).float(), label

    @property
    def targets(self) -> list[int]:
        """Lista de labels na ordem de self.samples — útil para split estratificado."""
        return [label for _, label in self.samples]

    @property
    def num_classes(self) -> int:
        return len(self.label_map)


def stratified_split(
    dataset: LibrasDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
    val_dataset: LibrasDataset | None = None,
) -> tuple[Subset, Subset]:
    """
    Split estratificado train/val preservando distribuição por classe.

    Faz o split nos ÍNDICES BASE (arquivos únicos), ignorando oversample. Depois:
      - subset de treino: expande cada base_idx nas N visitas (oversample).
      - subset de validação: mantém 1 visita por arquivo, sem aug.

    Se `val_dataset` for passado, ele fornece as amostras de validação (útil para
    ter train com augment=True e val com augment=False sobre os MESMOS arquivos).
    Requer que os dois datasets tenham o mesmo `.samples` (mesma ordem/conteúdo).
    """
    val_ds = val_dataset if val_dataset is not None else dataset
    if val_dataset is not None and len(val_ds.samples) != len(dataset.samples):
        raise ValueError(
            "val_dataset e dataset devem ter a mesma listagem de arquivos (.samples)."
        )

    n_base = len(dataset.samples)
    base_indices = list(range(n_base))
    base_targets = [label for _, label in dataset.samples]

    train_base_idx, val_base_idx = train_test_split(
        base_indices,
        test_size=val_ratio,
        stratify=base_targets,
        random_state=seed,
    )

    # Expande índices de treino para cobrir as K visitas do oversample.
    # __getitem__ do dataset faz `idx % len(self.samples)`, então qualquer offset
    # múltiplo de n_base cai no mesmo arquivo mas com aug diferente.
    factor = dataset.oversample_factor
    train_idx = [b + k * n_base for k in range(factor) for b in train_base_idx]

    return Subset(dataset, train_idx), Subset(val_ds, val_base_idx)
