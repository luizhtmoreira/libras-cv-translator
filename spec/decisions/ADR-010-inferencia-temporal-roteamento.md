# ADR-010 — Inferência temporal ao vivo com roteamento manual letra ↔ palavra

**Data:** 2026-07-14
**Status:** Aceita
**Contexto:** Diogo

---

## Contexto

Com o TransformerTemporal implementado (ADR-009), faltava o caminho de
inferência ao vivo para sinais dinâmicos: `infer.py` só roda o MLP frame a
frame. Três problemas precisavam de decisão conjunta:

1. **Onde vive a inferência temporal** — estender `infer.py` ou criar módulo novo?
2. **Como rotear estático ↔ dinâmico ao vivo** — o sistema não sabe se o
   usuário está soletrando um "A" (MLP) ou sinalizando "obrigado" (Transformer).
3. **Colisão de artefatos** — `train.py` salvava `model_config.json`,
   `label_map.json` e `train_log.json` compartilhados. Treinar o Transformer
   sobrescrevia os artefatos do MLP (aconteceu na prática: um dry-run do
   Transformer em 13/07 clobberou o config do MLP real de 15 letras). O
   roteamento exige os **dois** modelos carregados simultaneamente, cada um com
   seu próprio espaço de classes.

## Decisão

**1. Novo módulo `src/infer_temporal.py`** (Caminho A do roadmap):
- Buffer circular de 30 vetores `(66,)` — mesma janela do treino (ADR-004).
- Expansão para `(30, 86)` via `expand_sequence_for_model` só no momento da
  predição (ADR-007), nunca por frame.
- Trigger de predição com três guardas: buffer cheio, mão detectada há ≥ 10
  frames contínuos (evita disparo em transições) e ≥ 5 frames desde a última
  predição (~6 predições/s a 30 fps).
- Palavra só é aceita como "estável" quando duas predições consecutivas
  concordam acima do limiar de confiança (default 0.60 — mais frouxo que o MLP
  porque a distribuição softmax sobre 20+ classes dinâmicas é naturalmente
  mais achatada).
- Perder a mão **não zera o buffer** (o gesto pode continuar após oclusão
  breve), mas zera o contador de estabilidade.

**2. Roteamento manual por tecla** (Opção 1 da Fase 5): barra de espaço alterna
entre modo LETRA (MLP, pipeline idêntico ao `infer.py`) e modo PALAVRA
(Transformer). O buffer temporal é alimentado nos dois modos — trocar para
PALAVRA encontra a janela já quente.

**3. Artefatos por modelo**: `train.py` agora salva
`{model}_config.json`, `{model}_label_map.json` e `{model}_train_log.json`.
Os nomes legados (`model_config.json`, `label_map.json`, `train_log.json`)
continuam sendo escritos **apenas pelo MLP** — são os defaults do `infer.py`,
que permanece funcionando sem mudança.

## Justificativa

- **Módulo novo em vez de unificar no `infer.py`**: o fluxo temporal tem estado
  próprio (buffer, triggers, histórico). Misturar os dois em um arquivo
  dobraria a complexidade do loop principal e arriscaria regressão no caminho
  estático que já funciona.
- **Roteamento manual em vez de detector de movimento**: threshold de energia
  do pulso é sensível a ruído da webcam; um falso positivo envia frame de
  movimento ao MLP (ou vice-versa) e produz erro difícil de depurar ao vivo.
  Para o vídeo de demonstração, o modo manual é determinístico e legível.
- **Label maps separados por modelo**: os espaços de classe são disjuntos
  (letras estáticas × palavras + letras dinâmicas). Um `label_map.json` único
  obrigaria índices globais e acoplaria os dois treinos.

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| Unificar fluxo temporal no `infer.py` | Mais complexo, mais fácil de quebrar o caminho estático estável (roadmap, Fase 4 — Caminho B) |
| Detector de movimento (norma do Δ pulso) para rotear | Threshold ε sensível a ruído; falsos positivos geram predições erradas silenciosas (roadmap, Fase 5 — Opção 2) |
| Modelo único N+M classes para tudo | MLP estático já atinge F1 ≈ 0.98; fundir tudo no Transformer jogaria fora o baseline e o estudo de ablação |
| Predição a cada frame no modo palavra | 30× mais custo de CPU sem ganho de UX; 6 predições/s já supera a cadência humana de sinalização |

## Referências

- `spec/roadmap-completude.md` — Fases 4 e 5.
- ADR-004 (janela de 30 frames), ADR-007 (expansão 66→86 no consumo),
  ADR-009 (TransformerTemporal).
- `src/infer_temporal.py`, `src/train.py` (bloco de persistência).
