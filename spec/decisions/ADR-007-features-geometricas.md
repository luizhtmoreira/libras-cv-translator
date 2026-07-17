# ADR-007 — Features geométricas rotação-invariantes (66 → 86 dims)

**Data:** 2026-07-13
**Status:** Aceita
**Contexto:** Diogo

---

## Contexto

O feature vector definido em ADR-003 (63 landmarks normalizados + 3 delta = 66)
é **sensível a rotação da mão**: inclinar a mão ao lado gira todas as 21
coordenadas, movendo o ponto de operação do MLP para regiões nunca vistas
durante o treino.

Isso explica as confusões observadas ao vivo em pares onde a semântica é a
**relação entre pontas de dedos**, não a posição absoluta delas:

| Par confundido | O que diferencia | Por que 63 coords falham |
|---|---|---|
| U ↔ V | distância entre pontas do indicador e médio | rotação move as pontas juntas |
| D ↔ R | cruzamento indicador/médio | sinal geométrico, não posicional |
| N ↔ M | posição do polegar entre dedos | requer distâncias inter-dedos |
| S ↔ I | mão fechada vs. mínimo pra cima | requer flexão explícita |

Augmentation (ADR-006) mitiga em parte, mas não consegue substituir features
que já são **invariantes por construção**.

## Decisão

Adicionar 20 features geométricas derivadas dos landmarks normalizados,
calculadas on-the-fly no loader e no infer:

- **10 distâncias par-a-par entre as 5 pontas de dedos** (polegar, indicador,
  médio, anelar, mínimo) → discrimina U/V/D/R.
- **5 distâncias ponta-de-dedo → pulso** → discrimina mão aberta vs. fechada.
- **5 métricas de flexão** (distância ponta → base do dedo / MCP) → discrimina
  dedos estendidos vs. flexionados por dedo (S/I/L).

Feature vector expandido: `(63 landmarks | 3 delta | 20 geométricas)` = **86 dims**.

## Justificativa

- **Rotação-invariantes por construção:** distâncias euclidianas entre pontos
  não dependem da orientação da mão no espaço. Ataca a causa raiz das confusões
  reportadas, não só o sintoma.
- **Complementares, não substitutas:** as 63 coords originais + 3 deltas
  continuam presentes. O MLP aprende quando cada tipo de sinal é útil.
- **Zero custo em disco:** os `.npy` continuam `(30, 66)`. A expansão para 86
  acontece em `features.expand_frame_for_model`, chamado tanto por
  `LibrasDataset.__getitem__` quanto por `infer.py`.
- **Compatível com ADR-003:** o vetor de 66 continua sendo a fonte da verdade
  em disco; a expansão é um adaptador de I/O, não uma substituição.

## Ablação

Comparação treinando o mesmo `MLPStatic` no dataset Brazilian Alphabet:

| Configuração | F1 macro val | Accuracy val |
|---|---|---|
| Baseline (66 dims, sem aug) | 0.9830 | 0.9896 |
| **86 dims + aug (ADR-006 + ADR-007)** | **0.9863** | **0.9935** |

Ganho pequeno no validation (dataset já saturado). O impacto principal está na
inferência ao vivo, onde a robustez a rotação é o gargalo.

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| Substituir as 63 coords por só features geométricas | Perde informação de posição relativa dentro da palma; MLP perde capacidade |
| Ângulos de junta (dot product entre segmentos) | Redundante com distâncias par-a-par; mais custo com pouco ganho marginal |
| PCA sobre o vetor de 66 | Reduz dimensionalidade mas não adiciona invariância |
| Aprender essas features via camada convolucional | Overkill para MLP; melhor deixar explícito |

## Referências

- Implementação: `src/features.py::build_geometric_features` e `expand_frame_for_model`
- Constantes: `src/constants.py::GEOMETRIC_DIM`, `FEATURE_DIM_EXTENDED`
