"""
train.py — Treino do baseline MLP para classificação de Libras estático
========================================================================

Fluxo:
    dataset .npy (30, 66) → frame do meio (66,) → MLPStatic → label

Métrica alvo: F1-Score macro ≥ 0.90 no split de validação.

Uso típico:

    # Treino real (dados no disco):
    python -m src.train --data-dir data/processed/alphabet

    # Smoke test sem dados reais (gera tensores sintéticos e treina poucas épocas):
    python -m src.train --dry-run

Artefatos salvos em --out-dir (default: models/):
    mlp_static.pth       state_dict do melhor modelo (maior F1 macro em validação)
    label_map.json       {label_str: idx}
    model_config.json    hyperparams + arquitetura para o infer.py reconstruir
    train_log.json       histórico de loss/acc/F1 por época + relatório final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.constants import FEATURE_DIM, FEATURE_DIM_EXTENDED, FRAMES
from src.dataset import (
    LibrasDataset,
    build_label_map,
    save_label_map,
    stratified_split,
)
from src.models import MODEL_DATASET_MODE, MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# HIPERPARÂMETROS — ajuste tudo aqui.
# Estes valores são a fonte única de verdade; a CLI só existe para overrides
# pontuais (ex: --dry-run, --data-dir). Mexer aqui é o caminho recomendado.
# ═══════════════════════════════════════════════════════════════════════════
HPARAMS: dict = {
    # ── Arquitetura ──────────────────────────────────────────────────────
    # Camadas ocultas do MLP. Mais fundo/largo = mais capacidade (e overfit).
    # Para 21 classes e ~63 features normalizadas, (256,128,64) costuma bater
    # F1 macro > 0.90 sem overfittar.
    "hidden_dims": [256, 128, 64],
    "dropout": 0.2,

    # ── Otimização ───────────────────────────────────────────────────────
    "lr": 1e-3,
    "weight_decay": 1e-4,     # L2 leve — ajuda com desbalanceamento
    "batch_size": 64,

    # ── Loop de treino ───────────────────────────────────────────────────
    "epochs": 150,
    "patience": 20,           # early stopping (épocas sem melhora de F1 macro)
    "val_ratio": 0.2,

    # ── Balanceamento ────────────────────────────────────────────────────
    # Class weights inversos à frequência. Alfabeto brasileiro é desbalanceado
    # (ex: 'n' com 31 amostras vs 'b' com 563). Sem isso o MLP colapsa.
    "class_weight": True,

    # Teto de arquivos por classe (subamostragem com seed fixa). O teto de 80
    # existia para o desbalanceamento de 200× do dataset unificado (ADR-011,
    # revertida). Com escopo letras-only o desbalanceamento é moderado e o
    # class_weight dá conta; um teto cortaria o alphabet (~256/classe) sem
    # necessidade. Roots em --no-cap-dir nunca são subamostrados.
    "max_per_class": None,

    # ── Scheduler (ReduceLROnPlateau em cima do F1 macro de validação) ──
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 6,
    "lr_min": 1e-6,

    # ── Augmentation de landmarks (ADR-006 / ADR-008) — só no split de treino
    # Perturbações geométricas aplicadas aos 21 landmarks (21,3) já normalizados.
    # Ataca o domain shift entre dataset de estúdio (Bianka) e webcam do usuário.
    # Faixas fortes (ADR-008): dataset é pequeno, webcam é hostil — precisamos
    # forçar o modelo a ver o máximo de variações possíveis.
    "augment": True,
    "augmentation": {
        "rot_range_deg": 25.0,        # ↑ de 15 → 25 (mão bem inclinada)
        "scale_range": 0.15,          # ↑ de 0.10 → 0.15 (perto/longe da câmera)
        "translation_range": 0.08,    # ↑ de 0.05 → 0.08
        "noise_std": 0.015,           # ↑ de 0.01 → 0.015 (ruído do detector real)
    },

    # Oversample: cada amostra passa N vezes por epoch, cada vez com aug diferente.
    # Multiplica a diversidade efetiva sem inflar disco. 4x já é ganho grande
    # e mantém o tempo de epoch aceitável em CPU.
    "oversample_factor": 4,

    # ── Reprodutibilidade ────────────────────────────────────────────────
    "seed": 42,

    # ── TransformerTemporal (só usado quando --model transformer_temporal) ──
    # Padrões conservadores: dataset dinâmico de Libras é pequeno (~1000 vídeos
    # para MINDS-Libras), rede grande demais overfitta na hora.
    # Ablação 2026-07-14 (MINDS 3 sinalizadores, 288 amostras, 20 classes):
    #   run 1: n_layers=3, dropout=0.1 → F1 val 0.8474 (train acc 1.0 — overfit)
    #   run 2: n_layers=2, dropout=0.2 → F1 val 0.8242 (regularizar custou mais
    #          que rendeu com dataset deste tamanho)
    # Mantida a config da run 1. Próximo ganho esperável vem de DADOS (mais
    # sinalizadores ou capturas próprias), não de tuning.
    "transformer": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 256,
        "dropout": 0.1,
    },
}


# ─────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────
def make_synthetic_dataset(
    root: Path,
    classes: list[str],
    samples_per_class: int = 60,
    seed: int = 0,
    dynamic: bool = False,
) -> None:
    """
    Gera tensores (30, 66) sintéticos para smoke-test do pipeline.

    Se dynamic=False (padrão), cada classe tem 1 centro estático + ruído — todos
    os 30 frames são variações desse centro. Testa o MLP baseline.

    Se dynamic=True, cada classe tem TRAJETÓRIA distinta (linear no espaço de
    features com direção característica). Frame t = start + t/(T-1) * direction.
    Testa o Transformer — o discriminante é a evolução temporal, não o estado
    médio (que fica ~= centro do dataset para todas as classes).
    """
    rng = np.random.default_rng(seed)

    if not dynamic:
        centers = rng.normal(0, 1, size=(len(classes), FEATURE_DIM)).astype(np.float32)
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = root / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(samples_per_class):
                noise = rng.normal(0, 0.2, size=(FRAMES, FEATURE_DIM)).astype(np.float32)
                tensor = centers[cls_idx][None, :] + noise
                np.save(cls_dir / f"{i}.npy", tensor)
    else:
        # Direções de trajetória — normais entre classes para separação clara.
        directions = rng.normal(0, 1, size=(len(classes), FEATURE_DIM)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6
        t_axis = np.linspace(0.0, 1.0, FRAMES, dtype=np.float32)[:, None]  # (T,1)
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = root / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            direction = directions[cls_idx][None, :]  # (1, F)
            for i in range(samples_per_class):
                start = rng.normal(0, 0.3, size=(1, FEATURE_DIM)).astype(np.float32)
                # trajetória = start + t * direction + ruído por-frame
                noise = rng.normal(0, 0.05, size=(FRAMES, FEATURE_DIM)).astype(np.float32)
                tensor = start + t_axis * direction + noise
                np.save(cls_dir / f"{i}.npy", tensor.astype(np.float32))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device,
    num_classes: int,
) -> dict:
    """
    Roda avaliação e retorna dict com loss, accuracy, balanced accuracy,
    F1 macro/weighted e vetores brutos de y_true/y_pred (para o report final).
    """
    model.eval()
    total_loss = 0.0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    labels = list(range(num_classes))

    acc = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    # Balanced accuracy = média de recalls por classe (ignora classes vazias)
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    for cls in range(num_classes):
        mask = y_true == cls
        per_class_total[cls] = mask.sum()
        per_class_correct[cls] = (y_pred[mask] == cls).sum()
    seen = per_class_total > 0
    balanced_acc = float((per_class_correct[seen] / per_class_total[seen]).mean()) if seen.any() else 0.0

    return {
        "loss": total_loss / total,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────
# TREINO
# ─────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    # Modo do dataset determinado pelo modelo escolhido — MLP quer 1 frame,
    # Transformer quer a sequência inteira.
    dataset_mode = MODEL_DATASET_MODE[args.model]

    if args.dry_run:
        tmpdir = tempfile.mkdtemp(prefix="libras_dryrun_")
        classes = ["A", "B", "C", "D", "E"]
        # Para o transformer o discriminante precisa estar no eixo temporal;
        # geramos trajetórias distintas por classe.
        dynamic_data = dataset_mode == "sequence"
        print(f"[dry-run] gerando dataset sintético ({'dinâmico' if dynamic_data else 'estático'}) em {tmpdir} → {len(classes)} classes")
        make_synthetic_dataset(Path(tmpdir), classes, samples_per_class=60, dynamic=dynamic_data)
        args.data_dir = tmpdir
        HPARAMS["epochs"] = min(HPARAMS["epochs"], 5)

    # Múltiplos roots: --data-dir (obrigatório) + --extra-dir (opcionais, ex: capturas próprias)
    data_roots: list[Path] = [Path(args.data_dir)]
    if args.extra_dir:
        data_roots.extend(Path(p) for p in args.extra_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset + split ──────────────────────────────────────
    label_map = build_label_map(data_roots)
    # Modelo consome 86 dims: 63 landmarks + 3 delta + 20 geométricas (ADR-007).
    input_dim = FEATURE_DIM_EXTENDED

    # Dois datasets sobre os MESMOS arquivos: um com augment=True + oversample
    # (treino), outro sem nada (validação). Split estratificado por arquivo base.
    # Subamostragem por classe idêntica nos dois datasets: mesmo max_per_class,
    # mesma subsample_seed e mesmos no_cap_dirs → mesma listagem de arquivos.
    cap_kwargs = {
        "max_per_class": HPARAMS.get("max_per_class"),
        "subsample_seed": HPARAMS["seed"],
        "no_cap_dirs": args.no_cap_dir or [],
    }
    train_ds_full = LibrasDataset(
        data_roots,
        mode=dataset_mode,
        label_map=label_map,
        augment=HPARAMS["augment"],
        aug_kwargs=HPARAMS["augmentation"],
        rng_seed=HPARAMS["seed"],
        oversample_factor=HPARAMS.get("oversample_factor", 1),
        **cap_kwargs,
    )
    val_ds_full = LibrasDataset(
        data_roots,
        mode=dataset_mode,
        label_map=label_map,
        augment=False,
        **cap_kwargs,
    )
    n_base = len(train_ds_full.samples)
    print(f"Modelo: {args.model}  |  dataset_mode: {dataset_mode}")
    print(f"Classes ({len(label_map)}): {list(label_map.keys())}")
    print(f"Roots: {[str(r) for r in data_roots]}")
    print(f"Arquivos únicos: {n_base}  |  visitas por epoch (oversample × arquivos): {len(train_ds_full)}")
    if cap_kwargs["max_per_class"] is not None:
        print(f"max_per_class: {cap_kwargs['max_per_class']}  |  roots protegidos: {cap_kwargs['no_cap_dirs']}")
    print(f"Input dim: {input_dim}  |  augment: {HPARAMS['augment']}  |  oversample: {train_ds_full.oversample_factor}x")

    train_subset, val_subset = stratified_split(
        train_ds_full,
        val_ratio=HPARAMS["val_ratio"],
        seed=HPARAMS["seed"],
        val_dataset=val_ds_full,
    )
    print(f"Split: train={len(train_subset)}  val={len(val_subset)}")

    # Ponteiro genérico para consultar targets/num_classes (mesmos em ambos)
    dataset = train_ds_full

    train_loader = DataLoader(
        train_subset, batch_size=HPARAMS["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=HPARAMS["batch_size"], shuffle=False, num_workers=0
    )

    # ── Modelo ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cls = MODEL_REGISTRY[args.model]
    if args.model == "mlp_static":
        model = model_cls(
            num_classes=dataset.num_classes,
            input_dim=input_dim,
            hidden_dims=tuple(HPARAMS["hidden_dims"]),
            dropout=HPARAMS["dropout"],
        ).to(device)
    else:  # transformer_temporal
        tcfg = HPARAMS["transformer"]
        model = model_cls(
            num_classes=dataset.num_classes,
            input_dim=input_dim,
            seq_len=FRAMES,
            d_model=tcfg["d_model"],
            n_heads=tcfg["n_heads"],
            n_layers=tcfg["n_layers"],
            d_ff=tcfg["d_ff"],
            dropout=tcfg["dropout"],
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {args.model}  |  parâmetros: {n_params:,}")

    # ── Class weights (inverso da frequência no split de treino) ──
    # `train_subset.indices` são índices EXPANDIDOS (com oversample). Módulo
    # n_base volta pro arquivo real, que é o que define a distribuição de classe.
    if HPARAMS["class_weight"]:
        base_targets = [label for _, label in dataset.samples]
        train_targets = np.array([base_targets[i % n_base] for i in train_subset.indices])
        counts = np.bincount(train_targets, minlength=dataset.num_classes)
        counts = np.maximum(counts, 1)
        weights = train_targets.size / (dataset.num_classes * counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights: min={weights.min():.2f}  max={weights.max():.2f}")
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=HPARAMS["lr_scheduler_factor"],
        patience=HPARAMS["lr_scheduler_patience"],
        min_lr=HPARAMS["lr_min"],
    )

    # ── Loop com early stopping em F1 macro ──────────────────
    history: list[dict] = []
    best_f1 = -1.0
    patience_left = HPARAMS["patience"]
    ckpt_path = out_dir / f"{args.model}.pth"

    for epoch in range(1, HPARAMS["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += x.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val = evaluate(model, val_loader, criterion, device, dataset.num_classes)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_balanced_acc": val["balanced_acc"],
            "val_f1_macro": val["f1_macro"],
            "val_f1_weighted": val["f1_weighted"],
        })
        print(
            f"Ep {epoch:03d} | lr={current_lr:.1e} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val['loss']:.4f} acc={val['acc']:.4f} "
            f"bal_acc={val['balanced_acc']:.4f} "
            f"F1={val['f1_macro']:.4f}"
        )

        scheduler.step(val["f1_macro"])

        if val["f1_macro"] > best_f1 + 1e-4:
            best_f1 = val["f1_macro"]
            patience_left = HPARAMS["patience"]
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping na época {epoch} (sem melhora de F1 há {HPARAMS['patience']} épocas).")
                break

    # ── Relatório final com o melhor checkpoint ──────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    final = evaluate(model, val_loader, criterion, device, dataset.num_classes)
    idx_to_label = {v: k for k, v in label_map.items()}
    target_names = [idx_to_label[i] for i in range(dataset.num_classes)]

    report_str = classification_report(
        final["y_true"], final["y_pred"],
        labels=list(range(dataset.num_classes)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        final["y_true"], final["y_pred"],
        labels=list(range(dataset.num_classes)),
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print("\n" + "=" * 72)
    print("RELATÓRIO FINAL — melhor checkpoint (maior F1 macro em validação)")
    print("=" * 72)
    print(report_str)
    print(f"F1 macro:    {final['f1_macro']:.4f}")
    print(f"F1 weighted: {final['f1_weighted']:.4f}")
    print(f"Accuracy:    {final['acc']:.4f}")
    if final["f1_macro"] >= 0.90:
        print("Alvo atingido: F1 macro ≥ 0.90")
    else:
        print(f"Abaixo do alvo (0.90). Ajuste HPARAMS no topo de src/train.py.")

    # ── Persistência ─────────────────────────────────────────
    # Artefatos POR MODELO — MLP e Transformer coexistem em models/ e o
    # infer_temporal.py precisa dos dois carregados ao mesmo tempo (ADR-010).
    # Cada modelo tem seu próprio label_map: o espaço de classes do MLP
    # (letras estáticas) quase não intersecta o do Transformer (letras dinâmicas).
    train_log = {
        "hparams": HPARAMS,
        "history": history,
        "final_report": report_dict,
        "final_f1_macro": final["f1_macro"],
        "final_f1_weighted": final["f1_weighted"],
        "final_acc": final["acc"],
    }
    save_label_map(label_map, out_dir / f"{args.model}_label_map.json")
    with open(out_dir / f"{args.model}_config.json", "w", encoding="utf-8") as f:
        json.dump(model.config(), f, indent=2)
    with open(out_dir / f"{args.model}_train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)

    # Aliases legados (label_map.json / model_config.json / train_log.json):
    # só o MLP escreve neles — são os defaults do infer.py. Treinar o
    # Transformer não pode clobberar os artefatos do modelo estático.
    if args.model == "mlp_static":
        save_label_map(label_map, out_dir / "label_map.json")
        with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model.config(), f, indent=2)
        with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2, ensure_ascii=False)

    print(f"\nArtefatos salvos em: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────
# CLI — apenas overrides pontuais; ajustes de tuning vão em HPARAMS.
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino do baseline MLP de Libras")
    p.add_argument("--data-dir", type=str, default="data/processed/alphabet",
                   help="Diretório principal com subpastas por classe contendo .npy")
    p.add_argument("--extra-dir", type=str, nargs="+", default=None,
                   help="Diretórios extras a mesclar (ex: data/raw/ com capturas próprias)")
    p.add_argument("--no-cap-dir", type=str, nargs="+",
                   default=["data/raw", "data/raw/dynamic"],
                   help="Roots nunca subamostrados pelo max_per_class (webcam alvo)")
    p.add_argument("--out-dir", type=str, default="models",
                   help="Onde salvar pesos, label_map e log")
    p.add_argument("--model", type=str, default="mlp_static",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--dry-run", action="store_true",
                   help="Gera dataset sintético e treina poucas épocas — valida pipeline")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(HPARAMS["seed"])
    np.random.seed(HPARAMS["seed"])
    train(args)
