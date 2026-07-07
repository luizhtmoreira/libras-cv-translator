# ADR-004 — Capturar 30 frames para sinais estáticos e dinâmicos

**Data:** 2026-07-07
**Status:** Aceita
**Contexto:** Participante 2 (Engenharia de Dados)

---

## Contexto

O backlog previa dois tipos de sinal:
- **Estáticos** (letras do alfabeto): forma da mão em 1 posição fixa
- **Dinâmicos** (palavras com movimento): gesto que evolui no tempo

A questão era se deveríamos capturar formatos diferentes para cada tipo.

## Decisão

Capturar sempre **30 frames** para ambas as categorias, gerando tensores com o mesmo
shape `(30, 66)` em todos os casos.

## Justificativa

**1. Um único formato de dados → uma única pipeline**
Ter dois formatos distintos exigiria dois modelos com arquiteturas incompatíveis, mais um
classificador prévio para rotear entre eles. Isso triplicaria a complexidade do sistema.

**2. Compatibilidade com o Estudo de Ablação**
O mesmo arquivo `.npy` alimenta os dois modelos:
- Baseline: extrai `tensor[15]` → `(66,)` → MLP
- Temporal: usa `tensor` completo → `(30, 66)` → Transformer

**3. Sinais estáticos se auto-distinguem pelos deltas**
Para letras, os deltas do pulso serão próximos de `0` em todos os frames.
O modelo temporal aprende que "delta ≈ 0 por 30 frames = sinal estático".

**4. Robustez a tremor na gravação**
Mesmo que o usuário trema levemente ao gravar uma letra, o modelo baseline
usa apenas o frame central (`tensor[15]`), minimizando o impacto.

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| 1 frame para estáticos, 30 para dinâmicos | Exigiria dois modelos e um classificador de roteamento |
| 60 frames (2 segundos) | Tempo longo para sinais rápidos; aumenta custo computacional sem ganho claro |
| 15 frames | Insuficiente para sinais dinâmicos mais lentos |
