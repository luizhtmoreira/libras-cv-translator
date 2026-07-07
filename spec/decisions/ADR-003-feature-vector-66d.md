# ADR-003 — Feature vector de 66 dimensões (63 landmarks + 3 delta do pulso)

**Data:** 2026-07-07
**Status:** Aceita
**Contexto:** Participante 2 (Engenharia de Dados)

---

## Contexto

A normalização por translação (ADR-002) resolve invariância posicional frame a frame, mas
introduz um problema para sinais dinâmicos: ao centralizar o pulso em `(0,0,0)` em **cada**
frame independentemente, a trajetória do movimento da mão é apagada.

Por exemplo, o sinal "Obrigado" (mão que sai do peito em direção à frente) teria todos os
frames com pulso em `(0,0,0)`, perdendo a informação de que a mão se moveu.

## Decisão

Adicionar 3 features extras ao vetor de cada frame: `Δx, Δy, Δz` do pulso em relação ao
frame anterior (usando as coordenadas **brutas**, antes da normalização).

**Feature vector final por frame:** `[63 coords normalizadas | Δx_pulso, Δy_pulso, Δz_pulso]`
→ shape `(66,)`

**Tensor final por amostra:** `(30 frames × 66 features)` → shape `(30, 66)`

## Justificativa

- **Separa dois tipos de informação:**
  - *Forma da mão* (o que os dedos estão fazendo) → 63 valores normalizados
  - *Movimento da mão* (para onde a mão está indo) → 3 deltas do pulso
- **Compatível com o Estudo de Ablação:** o modelo baseline usa só `tensor[15]` (frame central),
  que contém os 63 + 3 valores. O modelo temporal usa o tensor inteiro `(30, 66)`.
- **Sinais estáticos:** deltas ≈ `0` em todos os frames → o modelo aprende que ausência de
  movimento é uma feature relevante
- **Sinais dinâmicos:** deltas refletem a direção e velocidade do movimento

## Sobre o frame central para o modelo baseline

Para o modelo estático (baseline), será usado apenas `tensor[15]` (15º frame das 30).
Isso mitiga o efeito de tremor acidental durante a gravação de sinais estáticos:
- A forma da mão (63 valores) reflete o snapshot naquele instante
- Pequenos tremores nos deltas têm peso reduzido (3 de 66 dimensões = 4.5%)

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| Sem delta (só 63 features) | Perde trajetória de sinais dinâmicos |
| Não normalizar posição (manter coords brutas) | Perde invariância posicional; modelo sensível à posição da mão na tela |
| Normalizar relativo ao 1º frame | Preserva trajetória, mas sensível à posição inicial; menos robusto em produção |
| Incluir velocidade (2ª derivada) | Adiciona complexidade sem ganho claro dado o tamanho do dataset |
