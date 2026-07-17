# ADR-009 — TransformerTemporal para sinais dinâmicos

**Data:** 2026-07-13
**Status:** Aceita (pipeline pronto; treino real depende de dataset dinâmico)
**Contexto:** Diogo

---

## Contexto

O MLP estático (ADR-006/007/008) cobre apenas o alfabeto **estático** (letras
onde a mão fica parada). Palavras como "obrigado", "bom dia", e as letras
dinâmicas J/K/X/Y/Z e H exigem um modelo que consuma a janela temporal inteira
`(30, 66)` — não um único frame.

O placeholder `TransformerTemporal` estava em `src/models.py` desde ADR-003;
faltava a implementação e o pipeline de treino.

## Decisão

Implementar `TransformerTemporal` como classificador temporal padrão:

```
Entrada: (B, 30, 86)  — 30 frames expandidos com features geométricas
  → Linear(86 → d_model=128)
  → + Positional Encoding senoidal (não treinável)
  → N × TransformerEncoderLayer (self-attn + FFN, pre-LN, GELU, dropout=0.1)
  → LayerNorm
  → mean pooling temporal (não CLS token)
  → Linear(d_model → num_classes)
Saída: (B, num_classes) — logits
```

Defaults (`HPARAMS["transformer"]`): `d_model=128, n_heads=4, n_layers=3,
d_ff=256, dropout=0.1` → ~410k parâmetros. Conservador porque MINDS-Libras
tem só ~1000 amostras.

### Integração com o pipeline existente

- **Mesmos `.npy (30, 66)` do MLP.** Zero mudança em `preprocess_external.py`,
  `capture.py`, `capture_dynamic.py`. A expansão pra 86 acontece no loader
  (`expand_sequence_for_model` em `features.py`).
- **`train.py` unificado** — mesmo script treina os dois. O modo do dataset
  (static vs sequence) é derivado do `--model` via `MODEL_DATASET_MODE`.
- **Augmentation temporalmente consistente.** Rotação/escala/translação são
  sorteadas UMA vez por sequência e aplicadas aos 30 frames idênticas —
  preserva a trajetória de movimento que é o próprio sinal. Ruído gaussiano
  é aplicado por-frame (ruído por definição não é correlacionado).
- **Checkpoint por modelo** — `mlp_static.pth` e `transformer_temporal.pth`
  convivem em `models/`. `model_config.json` reflete o último treinado.

### Escolhas de arquitetura

| Decisão | Alternativa considerada | Motivo |
|---|---|---|
| Mean pooling temporal | CLS token | Dataset pequeno; mean é regularizador implícito e evita o CLS virar bottleneck. |
| Positional encoding senoidal | Learnable positional embedding | Menos parâmetros; suficiente para 30 posições. |
| pre-LN (norm_first=True) | post-LN | Treino mais estável em epochs curtos, especialmente com dataset pequeno. |
| Ativação GELU | ReLU | Padrão moderno em Transformers; ganho marginal mas gratuito. |
| Sem attention mask | Padding mask | Todos os tensores têm exatamente 30 frames (ADR-004). |

## Justificativa

- **Ataca o problema real dos dinâmicos** — MLP não vê movimento por definição
  (ADR-003). Só um modelo temporal consegue "obrigado" vs "olá" vs "bom dia".
- **Compatibilidade máxima com o já existente** — a interface `dataset ← npy →
  model` foi preservada. Nenhum ADR anterior foi violado.
- **Custo de treino aceitável em CPU** — ~410k params, batch=64, 30 timesteps.
  Uma epoch em MINDS-Libras (~1000 vídeos) deve levar 1-2 min em CPU decente.
- **Escopo delimitado**: `infer.py` NÃO foi adaptado. Continua exclusivamente
  para o MLP estático (single frame). Inferência ao vivo temporal exige buffer
  de 30 frames + trigger — escopo separado (`infer_temporal.py` futuro).
  A validação temporal fica via métrica de treino/val por enquanto.

## O que exige dataset dinâmico

O smoke-test (`--dry-run --model transformer_temporal`) gera trajetórias
sintéticas com direção característica por classe e valida que o pipeline
converge — F1 macro ≈ 0.84 em 5 epochs sobre dados sintéticos limpos.

Para treino real:

1. **MINDS-Libras** (ADR-005): 20 palavras, ~1000 vídeos, ~5 GB.
2. **Capturas próprias** via `capture_dynamic.py --sinal obrigado --amostras N`.
3. Combinar via `--extra-dir` (ADR-008).

## Referências

- Modelo: `src/models.py::TransformerTemporal`
- Loader temporal: `src/features.py::expand_sequence_for_model`
- ADRs relacionados: [ADR-003](ADR-003-feature-vector-66d.md),
  [ADR-004](ADR-004-30-frames-para-todos.md), [ADR-005](ADR-005-dataset-externo.md),
  [ADR-007](ADR-007-features-geometricas.md).
