# ADR-002 — Normalização por translação no pulso + escala pela palma

**Data:** 2026-07-07
**Status:** Aceita
**Contexto:** Luiz

---

## Contexto

Os landmarks brutos do MediaPipe são coordenadas relativas ao frame da câmera (0.0 a 1.0).
Isso significa que o modelo veria números diferentes para o mesmo sinal dependendo de:
- **Posição da mão** na tela (canto vs. centro)
- **Distância da câmera** (mão grande = mão próxima, mão pequena = mão longe)
- **Tamanho físico da mão** do usuário

## Decisão

Aplicar normalização em dois passos antes de salvar cada frame:

**Passo 1 — Translação:** subtrair as coordenadas do pulso (landmark 0) de todos os 21 pontos.
Resultado: o pulso sempre fica em `(0, 0, 0)`.

**Passo 2 — Escala:** dividir todos os pontos pela norma euclidiana do vetor
pulso→base do dedo médio (landmark 9).
Resultado: essa distância sempre vale `1.0`, independente do tamanho ou proximidade da mão.

## Justificativa

- **Invariância posicional:** o sinal gravado no canto da tela e no centro geram o mesmo vetor
- **Invariância de escala:** mãos grandes e pequenas geram vetores equivalentes
- **Invariância de distância:** mão próxima ou longe da câmera gera vetores equivalentes
- **O modelo aprende geometria, não posição:** as proporções relativas entre pontos são preservadas

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| Sem normalização | Modelo aprenderia posição/tamanho em vez do gesto |
| Normalizar pelo bounding box | Menos estável; bounding box muda com oclusões parciais |
| PCA whitening | Desnecessariamente complexo; remove interpretabilidade das features |
