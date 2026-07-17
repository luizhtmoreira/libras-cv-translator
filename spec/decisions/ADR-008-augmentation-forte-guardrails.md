# ADR-008 — Augmentation forte, oversampling e guardrails na inferência

**Data:** 2026-07-13
**Status:** Aceita
**Contexto:** Diogo

---

## Contexto

Após ADR-006 (augmentation) e ADR-007 (features geométricas), o modelo atingiu
F1 macro ≈ 0.986 no split de validação — mas em teste real na webcam ainda
apresentava confusões sistemáticas (N↔M, D↔U↔R, S↔I, V↔U↔D).

Duas causas foram investigadas e descartadas como resolução via mais dados
públicos:

1. **Baixar mais datasets de estúdio não resolve.** MINDS-Libras (5 GB) e
   V-LIBRASIL (10 GB) foram gravados em chroma key. Adicioná-los treina em
   mais estúdio, não na distribuição hostil da webcam do usuário.
2. **Não existe dataset público com o alfabeto Libras estático completo.**
   Bianka cobre 15 das 19 letras estáticas (faltam P, Q, T). J/K/X/Y/Z e H
   são dinâmicas e não cabem no MLP por definição.

## Decisão

Três mudanças combinadas, todas visando robustez à inferência real, não
métrica de validation:

### 1. Augmentation mais forte (`HPARAMS["augmentation"]`)

| Parâmetro | Antes (ADR-006) | Agora | Racional |
|---|---|---|---|
| `rot_range_deg` | 15° | **25°** | mão inclinada durante gesto natural pode ir bem além de 15° |
| `scale_range` | 0.10 | **0.15** | distância à webcam varia mais que ±10% no uso real |
| `translation_range` | 0.05 | **0.08** | pulso não é rigidamente fixo |
| `noise_std` | 0.01 | **0.015** | detector do MediaPipe é mais ruidoso ao vivo do que em fotos de estúdio |

### 2. Oversampling online (`HPARAMS["oversample_factor"] = 4`)

Cada arquivo `.npy` é visitado **4 vezes por epoch**, cada visita gerando uma
augmentation independente. Efeitos:

- Dataset efetivo cresce 4x sem custo em disco.
- Modelo vê 4 variações diferentes do mesmo sinal por epoch (contra 1).
- Class weighting continua correto: computado sobre `train_subset.indices %
  n_base` (o oversample não deve inflar contagens per-classe).
- Split estratificado é feito sobre índices BASE (não expandidos) para não
  vazar amostras do mesmo arquivo entre treino e validação.

### 3. Guardrails na inferência ao vivo (`src/infer.py`)

- **`--min-conf` default 0.85** (era 0.70). Abaixo disso, a UI mostra `?` em
  vez de um palpite ruim. Prevenir predição errada é mais valioso do que
  mostrar algo o tempo todo.
- **`--stable-frames` default 7** (era 5). Filtro temporal mais rígido.
- **Média móvel de softmax**. As últimas N distribuições de probabilidade são
  medianas, e a decisão vem do vetor suavizado — não do frame instantâneo.
  Reduz drasticamente flicker entre classes concorrentes (U↔V).

## Justificativa

- **Custo zero em disco.** Nenhuma reprocessamento de dataset. Todas as mudanças
  são no loader e na inferência.
- **Ataca o modo real de falha.** Val F1 estava saturado — o modelo já sabe o
  Bianka. O que falta é lidar com webcam. Aug mais forte e guardrails atacam
  exatamente isso.
- **Reversível por HPARAMS.** Tudo é ajustável no bloco no topo de `train.py` /
  CLI do `infer.py`. Se degradar num cenário, cada slider volta.

## Impacto medido

| Config | Val F1 macro | Val Accuracy |
|---|---|---|
| Baseline (66 dims, sem aug) | 0.9830 | 0.9896 |
| ADR-006 + ADR-007 (86 dims + aug leve) | 0.9863 | 0.9935 |
| **ADR-008 (aug forte + oversample 4x)** | **0.9842** | **0.9935** |

Val F1 caiu 0.002 — variação esperada. Augmentation forte "confunde" um pouco a
métrica que compartilha domínio com o treino. O ganho é fora do validation, na
webcam — deve ser medido empiricamente.

## O que NÃO foi feito e por quê

- **Baixar MINDS-Libras / V-LIBRASIL agora.** Ambos são dinâmicos (vídeo) e só
  ajudam quando o modelo temporal (`TransformerTemporal`) for implementado.
  Escopo separado.
- **Baixar LSWH100.** Dataset sintético (144k imagens de mãos renderizadas). É
  mais uma distribuição divergente da webcam real, não uma correção.
- **Buscar dataset com alfabeto completo.** Não existe público. J/K/X/Y/Z têm
  movimento e não cabem no MLP. Faltam P, Q, T no Bianka — só via capturas
  próprias.

## Próxima alavanca

Capturas próprias do usuário via `capture.py` continuam sendo o **maior salto
não realizado**. 30 amostras suas por letra confusa via:

```bash
python -m src.capture --sinal m --amostras 40
python -m src.train --extra-dir data/raw
```

isso resolve o domain shift de forma direta — o modelo passa a treinar na sua
webcam, não na câmera da Bianka.

## Referências

- Implementação: `HPARAMS` em `src/train.py`, `LibrasDataset` em `src/dataset.py`,
  loop em `src/infer.py`.
- ADRs anteriores: [ADR-006](ADR-006-augmentation-landmarks.md),
  [ADR-007](ADR-007-features-geometricas.md).
