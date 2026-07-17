# ADR-012 — Escopo letras-only e retorno à arquitetura dual

**Data:** 2026-07-16
**Status:** Aceita
**Contexto:** Diogo

---

## Contexto

A ADR-011 unificou letras e palavras em um único TransformerTemporal de 79
classes. O modelo passou os critérios U3 e a validação ao vivo U5, mas com
custo mensurável de precisão nas letras estáticas: F1 macro 0.9839 no unificado
vs **0.9947 no MLP especializado**. As letras dinâmicas (h/j/k/x/y/z) sempre
atingiram F1 = 1.00 nos modelos especializados.

Para a entrega (2026-07-17), precisão máxima na soletração importa mais do que
a UX de um modelo único cobrindo palavras.

## Decisão

1. **Escopo da POC passa a ser apenas o alfabeto (letras).** Palavras
   (MINDS-Libras e V-LIBRASIL) saem do pipeline de treino e inferência.
2. **A arquitetura dual da ADR-010 volta a ser a oficial:**
   - `MLPStatic` — 21 letras estáticas (a..w + y), features 86d (ADR-007);
   - `TransformerTemporal` — somente as **6 letras dinâmicas** h, j, k, x, y, z;
   - `infer_temporal.py` com alternância manual por barra de espaço
     (modo LETRA ↔ modo LETRA DINÂMICA).
3. O modelo unificado (79 classes) e o código exclusivo dele
   (`infer_unified.py`, `eval_groups.py`, `fetch_vlibrasil.py`,
   `organize_minds.py`) são removidos.
4. Ambos os modelos são retreinados com todos os dados de letras disponíveis
   (Brazilian Alphabet, capturas do Diogo em `data/raw`, DATA_LUIZ).

## Consequências

- **Positivas:** máxima precisão por especialização (referências: MLP 0.9947,
  letras dinâmicas 1.00); modelos menores e treino mais rápido em CPU;
  narrativa científica do TCC preservada (ablação estático vs temporal,
  agora demonstrada nas letras dinâmicas).
- **Negativas:** palavras ficam fora da demo; retorna o toggle manual por
  tecla (custo de UX que a ADR-011 tentou eliminar); os "sinais dinâmicos"
  prometidos no backlog passam a ser as letras h/j/k/x/y/z (validar com o
  professor).
- Os dados brutos de palavras em `data/external/` e `data/processed/words`
  **permanecem em disco** (não são usados; deleção só com OK explícito do
  Diogo).
- A letra `y` existe nos dois modelos (estática nas capturas do Diogo,
  dinâmica no DATA_LUIZ) — comportamento correto na arquitetura dual: o
  toggle decide qual modelo responde.

## Referências

- ADR-010 (arquitetura dual — volta a ser a oficial)
- ADR-011 (unificação — revertida por esta ADR)
- `spec/roadmap-refatoracao.md` (plano de execução R0–R6)
