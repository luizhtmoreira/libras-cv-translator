# Roadmap de unificação — modelo único para letras + palavras

> **⚠️ SUPERSEDED** pela reversão de 2026-07-16 (ADR-012): escopo voltou a ser
> letras-only com arquitetura dual (ADR-010). Ver `spec/roadmap-refatoracao.md`.
> Mantido apenas como histórico.

> **Documento de handoff / prompt de continuação.** Briefing self-contained para
> uma nova sessão executar a unificação MLP+Transformer em um único modelo.
> Leia até o fim antes de começar. Complementa (não substitui)
> `spec/roadmap-completude.md` — as regras e anti-padrões de lá continuam valendo.

---

## 0. Contexto do projeto (resumo executivo)

**O que é**: POC de tradutor de Libras em tempo real. Webcam → MediaPipe Hand
Landmarker (21 landmarks 3D) → normalização → classificador → filtro temporal.
CNN sobre pixels é **proibida pelo edital**; todo o design é landmarks-based.
TCC/disciplina UnB FCTE, professor Vinicius Rispoli, prazo **17/07/2026**.

**Pipeline de dados (invariantes — NÃO mudar)**:
- Todo `.npy` em disco tem shape **`(30, 66)`** float32: 30 frames × [63 coords
  normalizadas (pulso na origem, escala pela palma — ADR-002) | 3 Δ do pulso
  (ADR-003)]. Sinais estáticos = frame replicado 30× com Δ=0 (ADR-004).
- A expansão para **86 dims** (20 features geométricas rotação-invariantes,
  ADR-007) acontece **no loader/infer**, nunca em disco
  (`features.expand_frame_for_model` / `expand_sequence_for_model`).
- Augmentation (ADR-006/008): rotação ±25°, escala ±15%, translação ±0.08,
  ruído σ=0.015. Em sequências, UM sorteio por amostra aplicado aos 30 frames
  (preserva a trajetória). Oversample 4× com aug fresco por visita.
- HPARAMS ficam no bloco único no topo de `src/train.py`.
- Todo modelo tem `.config()` serializável; artefatos por modelo em `models/`:
  `{modelo}.pth`, `{modelo}_config.json`, `{modelo}_label_map.json`,
  `{modelo}_train_log.json` (ADR-010). Aliases legados (`model_config.json`,
  `label_map.json`, `train_log.json`) são escritos SÓ pelo mlp_static.

**Estado atual (2026-07-15) — tudo treinado e funcionando ao vivo**:

| Modelo | Classes | F1 macro (val) | Dados |
|---|---|---|---|
| `mlp_static` | 21 letras estáticas (a-w + y) | **0.9947** | Bianka (3.838) + capturas Diogo (320, 8 letras) + DATA_LUIZ estáticas (1.000, 20 letras) |
| `transformer_temporal` | 20 palavras MINDS + h/j/k/x/y/z | **0.8275** (letras dinâmicas todas 1.00) | MINDS 3 sinalizadores (288) + DATA_LUIZ dinâmicas (300) |

- Inferência ao vivo: `python -m src.infer_temporal` — dois pipelines, barra de
  espaço alterna modo LETRA (MLP) ↔ PALAVRA (Transformer). Validado ao vivo
  pelo usuário. **A troca manual de modo é o incômodo que motiva este roadmap.**
- Em andamento: integração do **V-LIBRASIL** (UFPE, Kaggle `davimedio01/v-librasil`)
  — subset curado de ~35 expressões do dia a dia (oi, obrigado, por_favor, sim,
  nao, agua, casa, familia, eu_amo_voce…), 3 vídeos/sinal (1 por articulador),
  baixado seletivamente via `src/fetch_vlibrasil.py` (~300 MB, não o dataset
  inteiro de 10,8 GB). Preprocessar com
  `python -m src.preprocess_external --source minds --input data/external/vlibrasil --output data/processed/words`.

**Mapa de dados no disco**:
```
data/processed/alphabet/      3.838 .npy — Bianka, 15 letras estáticas (estúdio)
data/raw/<letra>/             320 .npy — capturas Diogo (webcam alvo!), f,g,n,o,p,q,t,y
data/DATA_LUIZ/raw/<letra>/   1.300 .npy — 26 letras do integrante Luiz (webcam dele)
data/luiz_split/static|dynamic/  symlinks do DATA_LUIZ roteados por movimento
data/processed/words/         288+ .npy — MINDS (20 palavras) + V-LIBRASIL curado
data/raw/words/               (vazio ainda — capturas dinâmicas do Diogo entram aqui)
```

**Anti-padrões já validados (NÃO revisitar)**: mais dataset de estúdio não
resolve webcam (só capturas próprias resolvem — testado 3×); aug acima de
±30°/±20% piora; CLS token no lugar de mean pooling não ganha nada; formato de
disco ≠ (30,66) quebra tudo; regularizar o transformer com dataset pequeno
piorou (ablação 2026-07-14: n_layers 2 + dropout 0.2 → F1 0.8474→0.8242).

---

## 1. A pergunta: dá para unificar em UM modelo?

**Sim — e a arquitetura foi desenhada para isso desde o ADR-004.** Evidências:

1. **Formato único já existe**: estáticos e dinâmicos são `(30, 66)` no disco e
   `(30, 86)` no modelo. O Transformer consome ambos sem nenhuma mudança.
2. **O discriminante estático/dinâmico está nos dados**: sinais estáticos têm
   Δ do pulso ≈ 0 nos 30 frames; dinâmicos têm trajetória (medido no DATA_LUIZ:
   estáticas ~0.002-0.003 de |Δ| médio, dinâmicas 0.005-0.018, separação 2-6×).
   O ADR-004 previu: "o modelo aprende que delta≈0 = sinal estático".
3. **O Transformer já domina letras**: h/j/k/x/y/z todas com F1 1.00 na última
   run. Não há razão arquitetural para ele não aprender também poses paradas —
   mean pooling sobre 30 frames idênticos ≈ o próprio frame.
4. **O caso y prova o conceito**: y existe hoje como classe estática (capturas
   do Diogo, paradas) E dinâmica (Luiz, com movimento). No modelo único viram
   UMA classe com dois perfis de movimento — o modelo aprende ambos.

**O que NÃO fazer**: apagar o MLP. Ele é o baseline do estudo de ablação — a
narrativa científica do trabalho é exatamente "estático de 1 frame vs temporal
de 30 frames". A unificação é decisão de **UX/deploy** (o demo usa 1 modelo);
o MLP continua treinável e reportado no notebook como baseline.

**Riscos reais e mitigação**:

| Risco | Severidade | Mitigação |
|---|---|---|
| Desbalanceamento brutal: Bianka tem até 563/classe, V-LIBRASIL tem 3/classe | ALTA | `max_per_class` no dataset (novo HPARAM, ver Fase U2) + class weights já existentes |
| F1 macro fica ilegível (classes de 3 amostras puxam pra baixo) | média | reportar F1 por grupo (letras estáticas / letras dinâmicas / palavras) além do macro |
| Palavra em execução dispara letra no meio do gesto | média | guardrails já existentes: min-conf + estabilidade de 2 predições consecutivas + trigger de mão estável (≥10 frames) |
| Confusão letra-parada ↔ palavra-que-começa-parada | baixa | delta≈0 sustentado é forte; o transformer vê a janela inteira |
| Regressão nas letras estáticas vs MLP (0.9947) | média | critério de aceitação explícito por grupo (ver U3); fallback documentado |

---

## 2. Plano de confecção (fases U1–U5)

### U1 — Consolidar o dataset unificado (sem código novo)

O `train.py` já aceita múltiplos roots. O treino unificado é:

```bash
python -m src.train --model transformer_temporal \
  --data-dir data/processed/words \
  --extra-dir data/processed/alphabet data/raw data/luiz_split/static \
              data/luiz_split/dynamic data/raw/words
```

- Classes esperadas: ~21 estáticas + 6 letras dinâmicas + 20 MINDS + ~34
  V-LIBRASIL ≈ **80 classes** (y funde estático+dinâmico; banheiro funde
  MINDS+V-LIBRASIL).
- **Antes de rodar**: dry-run (`--dry-run --model transformer_temporal`) e
  conferir no log que o merge de classes está correto (sem duplicatas por
  acento/caixa).

### U2 — Rebalancear (única mudança de código relevante)

Adicionar `HPARAMS["max_per_class"]` (sugestão: 80) aplicado em
`LibrasDataset.__init__`: por classe, se houver mais arquivos que o teto,
amostrar aleatoriamente (seed fixa) o teto. Motivo: 563 Bianka "b" vs 3
"eu_amo_voce" — class weight sozinho não segura ordem de grandeza 200×.
- Manter TODAS as amostras de `data/raw/` e `data/raw/words` (domínio da
  webcam alvo é sagrado — nunca subamostrar essas).
- Implementação: ~20 linhas em `dataset.py`, parâmetro opcional, default None
  (sem teto) para não afetar treinos existentes.

### U3 — Treinar e avaliar por grupo

```bash
python -m src.train --model transformer_temporal --data-dir ... (como U1)
```

Critérios de aceitação (medir no `final_report` do train_log):
- Letras estáticas (grupo): F1 macro do grupo ≥ **0.95** (MLP faz 0.9947 —
  tolerância de 4 pontos pela dificuldade extra).
- Letras dinâmicas: F1 ≥ 0.95 (hoje 1.00).
- Palavras MINDS: sem regressão relevante vs 0.83 atual.
- Palavras V-LIBRASIL (3 amostras): qualquer F1 > 0 já é aceitável nesta fase;
  melhoram com capturas próprias.
- Se as letras estáticas regredirem < 0.90: aumentar `max_per_class` para 150 e
  retreinar; se persistir, PARAR e manter arquitetura dual (fallback: modo
  automático por energia de movimento — Opção 2 da Fase 5 do roadmap antigo —
  em vez de tecla).

### U4 — Inferência unificada

Evoluir `src/infer_temporal.py` (ou criar `src/infer_unified.py` reusando as
classes existentes):
- **Remover o toggle de modo**: só o pipeline do buffer (o `DynamicRecognizer`
  atual vira o único caminho). O MLP some da UX, fica só no notebook.
- UI: última predição estável em destaque + histórico (já existe); adicionar
  **buffer de soletração** — letras consecutivas estáveis acumulam numa string
  visível (ex: "O-I" enquanto soletra), palavras aparecem por extenso.
- Latência: predição a cada 5 frames já roda a ~6/s em CPU com 411K params;
  com 80 classes o head cresce ~30K params — irrelevante.
- Guardrails que já existem e devem permanecer: min-conf, estabilidade de 2
  predições, mão estável ≥10 frames, buffer não zera em oclusão breve.

### U5 — Validação ao vivo + documentação

- Teste ao vivo: soletrar "OI" letra a letra, depois sinalizar "obrigado" —
  sem apertar NENHUMA tecla entre os dois. Critério: ambos reconhecidos.
- Letras problemáticas históricas ao vivo: m, n, d, u, v, r, s, i (5s estáveis).
- Registrar **ADR-011** (modelo único): Contexto → Decisão → Justificativa →
  Alternativas Descartadas (dual com toggle manual — ADR-010; dual com roteador
  por movimento) → Referências. Numeração: próximo é ADR-011.
- Atualizar README (seção inferência), `spec/backlog.md`, notebook (seção 9:
  comparação MLP vs Transformer unificado — a tabela de 3 colunas fica ótima
  no relatório: baseline estático / temporal especializado / unificado).

---

## 3. Dicas operacionais

- Ambiente: `source .venv/bin/activate`; sem CUDA (CPU, treinos de 15-40 min).
- **Sempre** `--dry-run` após mexer em `dataset.py`.
- **Nunca** treinar sem backup: `cp models/transformer_temporal.pth{,.bak}`.
- Treinos em background com `python -u` (stdout sem buffer para monitorar).
- O usuário (Diogo) valida ao vivo — só ele tem webcam. Pausar e pedir o teste
  ao final de U4.
- Se o preprocess do V-LIBRASIL ainda não tiver rodado quando você começar:
  `python -m src.preprocess_external --source minds --input data/external/vlibrasil --output data/processed/words`
  e validar com `python -m src.validate_processed --data-dir data/processed/words`
  (amostras degeneradas de sinais quase-parados do V-LIBRASIL podem aparecer —
  avaliar caso a caso antes de deletar, alguns sinais são de fato pouco móveis).

## 4. Prompt sugerido para a nova sessão

> A partir da spec `spec/roadmap-unificacao.md`: execute as fases U1 a U5 na
> ordem. Atue como IA engineer senior. Faça tudo que não depende de webcam de
> forma autônoma (dataset, código, treinos, avaliação por grupo, documentação)
> e me chame apenas para o teste ao vivo do U5. Se o critério de aceitação do
> U3 falhar após o ajuste de `max_per_class`, pare e me apresente o comparativo
> antes de qualquer decisão de arquitetura.
