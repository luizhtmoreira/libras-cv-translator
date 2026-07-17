# Roadmap de Refatoração — Escopo letras-only, arquitetura dual (R0–R6)

**Data:** 2026-07-16 | **Decisões:** Diogo | **Executor:** IA (este documento é a fonte da verdade para continuar e terminar o trabalho)
**Prazo do projeto:** 2026-07-17 (TCC UnB/FCTE — Visão Computacional com Deep Learning)

---

## 1. A decisão (o que muda e por quê)

Diogo decidiu em 2026-07-16 **reverter a unificação** (ADR-011) e **remover palavras do escopo**:

1. **Deletar o modelo unificado** (transformer de 79 classes) e todo o código exclusivo dele.
2. **Voltar à arquitetura dual da ADR-010**: um MLP para letras estáticas + um TransformerTemporal para sinais dinâmicos, com **modo alternado por tecla** (barra de espaço) no `infer_temporal.py`.
3. **Escopo apenas LETRAS**: remover todas as palavras (MINDS-Libras e V-LIBRASIL) do pipeline. O Transformer passa a classificar somente as **6 letras dinâmicas** (h, j, k, x, y, z).
4. **Retreinar os dois modelos** usando o máximo dos datasets de letras disponíveis.

**Motivação declarada:** maior precisão. Referências que sustentam: MLP especializado F1=0.9947 vs 0.9839 das letras estáticas no unificado; letras dinâmicas F1=1.00 em todos os treinos anteriores.

---

## 2. Mapa da aplicação atual (estado em 2026-07-16, pós-U5)

### Código (`src/`)

| Arquivo | O que faz | Destino na refatoração |
|---|---|---|
| `constants.py` | Cores, FRAMES=30, paths, HAND_CONNECTIONS | **Mantém** |
| `features.py` | extract/normalize landmarks, vetor 66d, expansão 86d (ADR-007) | **Mantém** |
| `dataset.py` | `LibrasDataset` (30,66)→(30,86), `stratified_split`, `max_per_class` | **Adapta** (ver R1.5) |
| `models.py` | `MLPStatic`, `TransformerTemporal`, registries | **Mantém** |
| `train.py` | Pipeline de treino, HPARAMS, artefatos por modelo | **Adapta** (HPARAMS) |
| `capture.py` | Captura estática (burst) → `data/raw/{sinal}/` | **Mantém** (já com RunningMode.VIDEO) |
| `capture_dynamic.py` | Captura dinâmica 30 frames → `data/raw/words/` | **Adapta** (destino vira letras dinâmicas) |
| `preprocess_external.py` | Brazilian Alphabet (imagens) + MINDS (vídeos) → .npy | **Adapta** (remove ramo `minds`) |
| `fetch_vlibrasil.py` | Download seletivo do V-LIBRASIL | **DELETA** (só palavras) |
| `organize_minds.py` | Organização dos zips MINDS | **DELETA** (só palavras) |
| `infer.py` | Ao vivo, só MLP (letras estáticas) | **Mantém** |
| `infer_temporal.py` | Ao vivo DUAL: espaço alterna MLP↔Transformer (ADR-010) | **Vira a entrada principal** (renomear modo "PALAVRA"→"LETRA DINÂMICA") |
| `infer_unified.py` | Ao vivo com modelo único 79 classes (ADR-011) | **DELETA** |
| `eval_groups.py` | Avaliação por grupo semântico (critérios U3) | **DELETA** (existia para o unificado) |
| `report_figures.py` | Matriz de confusão + curvas a partir de `models/` | **Mantém** (recurso `--no-cap-dir` é genérico) |
| `validate_processed.py` | Valida shape/NaN dos .npy processados | **Mantém** (conferir se referencia words) |

### Dados (medido em 2026-07-16)

| Root | Conteúdo | Uso pós-refatoração |
|---|---|---|
| `data/processed/alphabet` | 15 classes estáticas, 3840 amostras (Brazilian Alphabet) | Treino MLP |
| `data/raw` | 8 classes (f,g,n,o,p,q,t,y), 320 amostras — capturas do Diogo | Treino MLP (domínio-alvo webcam) |
| `data/luiz_split/static` | 20 classes estáticas do DATA_LUIZ (**symlinks** → `data/DATA_LUIZ/raw`) | Treino MLP |
| `data/luiz_split/dynamic` | 6 classes dinâmicas h/j/k/x/y/z, **50 amostras cada** (symlinks) | Treino Transformer |
| `data/processed/words` | 53 classes de palavras, 382 amostras (MINDS+V-LIBRASIL) | **FORA** — não usar |
| `data/external/*` | Brutos: minds-zips 10 GB, minds-batch* 7,7 GB, vlibrasil 261 MB, alphabet 141 MB | **FORA** — ver risco §4.3 |

Atenção: os diretórios de classe em `data/luiz_split/*` são symlinks — `find` precisa de `-L` para contar amostras; o loader (`os.listdir`) os segue normalmente.

### Modelos (`models/`)

- `mlp_static.pth` + config/label_map/train_log — **21 classes estáticas, F1=0.9947 (2026-07-14), INTACTO** (a unificação não tocou no MLP). Contém a..w + y (as 5 dinâmicas h/j/k/x/z ficam fora do MLP; `y` existe nos dois mundos — ver §4.4).
- `transformer_temporal.pth` + artefatos — unificado 79 classes → **substituir** pelo retreino de 6 classes.
- `*.bak-pre-unificacao` (59 classes), `*.bak`, `*.bak2` — checkpoints históricos com palavras → deletar **somente após** R6 validar o novo estado.
- `hand_landmarker.task` — modelo MediaPipe, **nunca deletar**.
- Aliases legados (`label_map.json`, `model_config.json`, `train_log.json`) — escritos pelo MLP, defaults do `infer.py`. Mantém.

### Invariantes que NÃO podem ser violados

1. Todo `.npy` em disco é `(30, 66) float32`; estáticos = frame replicado 30× com Δ do pulso = 0 (ADR-004).
2. Expansão 66→86 dims acontece **só** no loader/inferência (ADR-007), nunca em disco.
3. CNN sobre pixels é **proibida pelo edital** — landmarks sempre.
4. Máquina sem CUDA: treinos em CPU, venv em `.venv/`.
5. `.npy`/`.pth` não sobem para o git (`.gitignore` já cobre).
6. Dry-runs de treino usam `--out-dir` no scratchpad — dry-run já sobrescreveu artefato real uma vez.
7. Detecção de mão: manter `RunningMode.VIDEO` + limiares 0.6/0.4/0.4 + 720p aplicados em 2026-07-16 nos 4 scripts de webcam. **Não reverter.**

---

## 3. Fases de execução

### R0 — Segurança (antes de deletar qualquer coisa)
- [ ] Criar branch `refactor/letras-only` e commitar o estado atual (há ~27 arquivos novos/modificados não commitados — o trabalho da unificação validado em U5 merece existir no histórico antes de ser revertido).
- [ ] Confirmar que `models/mlp_static_train_log.json` reporta F1≈0.9947 (é a referência de aceitação do MLP).

### R1 — Remover o modelo unificado
- [ ] Deletar `src/infer_unified.py` e `src/eval_groups.py`.
- [ ] Deletar figuras `reports/transformer_temporal_confusion_matrix.png`, `reports/transformer_temporal_curves.png` e as `*_59cls_*` (todas retratam modelos com palavras).
- [ ] ADR-011: mudar Status para "Revertida (2026-07-16) — ver ADR-012". **Não deletar o arquivo** (ADRs são histórico).
- [ ] Criar `spec/decisions/ADR-012-escopo-letras.md`: decisão de escopo letras-only + arquitetura dual como oficial, motivação (precisão), consequências (palavras fora, Transformer 6 classes).
- [ ] `spec/roadmap-unificacao.md`: adicionar nota no topo — "Superseded pela reversão de 2026-07-16, ver roadmap-refatoracao.md".
- [ ] `src/train.py` HPARAMS: `max_per_class: None` (o teto de 80 existia para o desbalanceamento do dataset unificado; com letras-only ele **cortaria** o alphabet ~256/classe sem necessidade). O mecanismo em `dataset.py` fica — é genérico e testado.

### R2 — Remover palavras do pipeline
- [ ] Deletar `src/fetch_vlibrasil.py` e `src/organize_minds.py`.
- [ ] `src/preprocess_external.py`: remover o ramo `--source minds` (manter `alphabet`).
- [ ] `src/capture_dynamic.py`: destino default deixa de ser `data/raw/words` — apontar para `data/raw/dynamic` (capturas futuras de letras dinâmicas do Diogo).
- [ ] `src/infer_temporal.py`: renomear todo texto de UI/HUD/prints "PALAVRA" → "LETRA DINAMICA" (constantes `MODE_WORD` podem virar `MODE_DYNAMIC`; é cosmético, não estrutural).
- [ ] `data/processed/words`: **não usar**. Não deletar sem OK do Diogo (ver §4.3) — se ele aprovar, deletar junto com `data/external/minds*` e `data/external/vlibrasil`.

### R3 — Retreinos
- [ ] Dry-run de sanidade primeiro: `python -m src.train --dry-run` e `--dry-run --model transformer_temporal` (com `--out-dir` no scratchpad!).
- [ ] **MLP (letras estáticas, 21 classes):**
  ```bash
  python -m src.train --data-dir data/processed/alphabet --extra-dir data/raw data/luiz_split/static
  ```
  Critério: F1 macro ≥ 0.99 no split de validação (referência 0.9947). Se regredir, comparar HPARAMS com o train_log de 14/07 antes de mexer em arquitetura.
- [ ] **Transformer (letras dinâmicas, 6 classes):**
  ```bash
  python -m src.train --model transformer_temporal --data-dir data/luiz_split/dynamic
  ```
  Critério: F1 macro ≥ 0.98 (referência: 1.00 dentro dos modelos maiores; 50 amostras/classe é confortável para 6 classes).
  Nota: com 6 classes o modelo atual (d_model=128, 3 camadas) está superdimensionado — aceitável para o prazo; reduzir só se houver overfit visível nas curvas.
- [ ] Verificar `{modelo}_label_map.json` resultantes: MLP = 21 classes; Transformer = exatamente `{h, j, k, x, y, z}`.

### R4 — Inferência dual como oficial
- [ ] `python -m src.infer_temporal` volta a ser a entrada principal. Os defaults já apontam para `mlp_static*` e `transformer_temporal*` — após o R3 o Transformer terá 6 classes e o script funciona sem mudança estrutural.
- [ ] Smoke test sem webcam: `py_compile` de todos os scripts + carregar os dois modelos + `predict` sobre tensor sintético.
- [ ] **Teste ao vivo (somente o Diogo pode)**: soletrar uma palavra em modo LETRA (m/n/d/u/v/r/s/i estáveis ~5 s cada); espaço; h/j/k/x/y/z em modo LETRA DINÂMICA; conferir que `y` funciona nos dois modos.

### R5 — Documentação
- [ ] `README.md`: remover seções de palavras (MINDS/V-LIBRASIL, download de zips), remover `infer_unified` da lista, `infer_temporal` como recomendado, treino com os dois comandos do R3, "ADRs 001 a 012".
- [ ] `spec/backlog.md`: atualizar §3 (Escopo) — palavras saem, letras dinâmicas h/j/k/x/y/z são os "sinais dinâmicos" da POC; anotar reversão nos itens de unificação/palavras já marcados (não apagar histórico, adicionar nota "revertido em 2026-07-16, ADR-012").
- [ ] `notebooks/MLP_base.ipynb`: remover seções 9/9.1/9.2 (unificação); refazer a seção do Transformer como "letras dinâmicas (6 classes)"; regenerar figuras:
  ```bash
  python -m src.report_figures --model mlp_static --data-dir data/processed/alphabet --extra-dir data/raw data/luiz_split/static
  python -m src.report_figures --model transformer_temporal --data-dir data/luiz_split/dynamic
  ```
  (O notebook não tem campo `id` nas células — nbformat antigo; editar por índice.)
- [ ] A narrativa científica (estudo de ablação) continua válida: MLP baseline estático vs Transformer temporal para sinais com movimento — agora demonstrado nas letras dinâmicas.

### R6 — Validação final e fechamento
- [ ] `python -m py_compile src/*.py` limpo.
- [ ] Classification report dos dois modelos nos train_logs atende os critérios do R3.
- [ ] Teste ao vivo do Diogo aprovado (R4).
- [ ] Deletar os checkpoints antigos de `models/` (unificado + baks com palavras) — só agora.
- [ ] Commit final na branch + push; merge conforme o Diogo decidir.

---

## 4. Riscos e pontos que exigem atenção (ou decisão do Diogo)

1. **Prazo é amanhã (17/07).** Esta refatoração descarta um modelo já validado ao vivo. O caminho todo (R0–R6) é executável em poucas horas porque dados e scripts existem, mas não há folga para exploração — seguir o plano na ordem, sem otimizações não pedidas.
2. **Escopo prometido no edital/backlog**: o backlog §3 prometia "sinais dinâmicos (palavras com movimento, ex: Obrigado)". Com palavras fora, os sinais dinâmicos da POC passam a ser as letras h/j/k/x/y/z — o componente temporal (janela de 30 frames + Transformer) continua demonstrado. **Confirmar com o Diogo/equipe que isso atende o professor.**
3. **Não deletar `data/external/` (≈18 GB) sem OK explícito do Diogo** — foram horas de download e não são re-obtidos facilmente. "Remover palavras" = tirar do pipeline, não necessariamente do disco.
4. **A letra `y`** existe como estática (capturas do Diogo em `data/raw/y`) e dinâmica (Luiz, `data/luiz_split/dynamic/y`). Na arquitetura dual isso é correto e era o estado pré-unificação: ela fica nos DOIS modelos e o toggle decide qual responde.
5. **"Retreinar com mais datasets"**: todos os dados de letras já conhecidos estão listados no §2 e entram nos comandos do R3. Se o Diogo baixou algum dataset de letras novo que não está em `data/`, ele precisa informar o caminho — não inventar fontes.

---

## 5. Referências

- ADR-010 (arquitetura dual + inferência temporal — volta a ser a oficial)
- ADR-011 (unificação — será marcada Revertida), ADR-004/006/007/009 (fundamentos que permanecem)
- `spec/roadmap-completude.md` (fases 1–6 originais), `spec/roadmap-unificacao.md` (superseded)
- Train logs de referência: `models/mlp_static_train_log.json` (0.9947, 2026-07-14)
