# ADR-001 — Usar extração de landmarks em vez de classificação de imagens

**Data:** 2026-07-07
**Status:** Aceita
**Contexto:** Luiz

---

## Contexto

O edital da disciplina proíbe abordagens exclusivamente baseadas em classificação de imagens
(ex: CNN treinada direto nos pixels). Era necessário justificar uma abordagem alternativa com
embasamento técnico.

## Decisão

Usar o **MediaPipe Hand Tracking** para extrair 21 landmarks 3D da mão em vez de processar
pixels brutos. Cada landmark é um ponto `(x, y, z)` no espaço, representando articulações
específicas da mão (pulso, nós dos dedos, pontas dos dedos, etc.).

## Justificativa

- **Atende ao edital:** a entrada do modelo são coordenadas geométricas, não imagens
- **Redução de dimensionalidade:** de `(H × W × 3)` pixels para `63` números por frame,
  eliminando ruído de cor, fundo e iluminação
- **Deploy leve:** o MediaPipe roda em CPU em tempo real; nenhuma GPU necessária para inferência
- **Interpretabilidade:** as features têm significado geométrico direto (posição das juntas)
- **Alinhamento com literatura:** abordagem usada em sistemas de reconhecimento de gestos
  com estado da arte em datasets pequenos

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| CNN em pixels brutos | Proibida pelo edital; requer dataset enorme |
| Segmentação semântica | Computacionalmente pesada; inviável para deploy em tempo real |
| Optical flow (fluxo óptico) | Mais sensível a ruído; não isola a mão do fundo |
