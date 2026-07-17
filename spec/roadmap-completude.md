# Roadmap de completude — Libras CV Translator

> **Documento de handoff.** Serve como briefing self-contained para uma nova sessão
> continuar o trabalho sem depender do histórico da conversa anterior. Leia até o
> fim antes de começar — a seção "Anti-padrões" evita perder tempo com abordagens
> que já foram tentadas e reprovadas.

---

## 0. Contexto (leia primeiro)

**Objetivo do projeto**: POC de tradutor de Libras em tempo real via extração de
landmarks com MediaPipe (não CNN em pixels — proibido pelo edital). TCC/disciplina
UnB FCTE, prazo **17/07/2026**.

**Arquitetura resumida**:
```
Webcam → MediaPipe Hand Landmarker → 21 landmarks 3D
       → features.py normaliza (pulso na origem, escala pela palma) → (66,)
       → dataset.py expande on-the-fly com features geométricas → (86,) ou (30, 86)
       → modelo (MLP para estático OU Transformer para dinâmico) → classe
       → filtro temporal → letra/palavra estável
```

**Fonte da verdade do design** — `spec/decisions/ADR-001` até `ADR-009`. Leia todos
antes de tocar em `features.py`, `dataset.py`, `models.py` ou `train.py`. Especialmente:

- ADR-002 (normalização), ADR-003 (feature vector 66), ADR-004 (30 frames sempre)
- ADR-006 (aug landmarks), ADR-007 (features geométricas), ADR-008 (aug forte + guardrails)
- ADR-009 (TransformerTemporal)

**Regras gerais do projeto** (verifique em `CLAUDE.md` se existir + esta seção):

- `.npy` **sempre** com shape `(30, 66)`. A expansão para 86 acontece **no loader**,
  nunca em disco. Isso preserva compat com todos os `.npy` já processados.
- Todo novo modelo tem `.config()` serializável — usado por `infer.py` para
  reconstruir a rede sem hardcode.
- HPARAMS ficam no bloco no topo de `src/train.py`. **Não** espalhe hyperparams
  em vários lugares.
- Não faça downloads sem confirmação: MINDS-Libras é ~5 GB, V-LIBRASIL é 10 GB.
- Registre decisões arquiteturais como novo ADR (`spec/decisions/ADR-XXX-*.md`).

---

## 1. Estado atual (atualizado 2026-07-14 — sessão de execução das Fases 2–6)

### Concluído em 2026-07-14

- **Fase 2.1 (adaptada)**: MINDS via **Zenodo record 2667329** (subset RGB de 3
  sinalizadores, 10,6 GB — o Kaggle tem 44,5 GB em archive único; limite de
  15 GB imposto pelo Diogo). 288 tensores válidos em `data/processed/words`,
  20 classes. Ver adendo no ADR-005 (inclui armadilhas: URLs da API, md5, zips
  >4 GiB precisam de 7z). Novos utilitários: `src/organize_minds.py`,
  `src/validate_processed.py`.
- **Fase 3.1**: Transformer treinado — **F1 macro val 0.8474** (alvo 0.85,
  essencialmente atingido com só 3 sinalizadores). Sem colapso de classes.
  Ablação registrada no bloco HPARAMS de `src/train.py` (n_layers=2 +
  dropout=0.2 piorou → 0.8242; ganho futuro vem de dados, não tuning).
- **Fases 4+5**: `src/infer_temporal.py` criado (Caminho A + Opção 1 — espaço
  alterna letra/palavra). Testado ponta a ponta sem webcam com amostra real
  (pred correta, conf 0.99). ADR-010.
- **Artefatos por modelo**: `train.py` salva `{modelo}_config/label_map/train_log`;
  aliases legados só do MLP. (Um dry-run havia sobrescrito o MLP real — retreinado:
  **F1 0.9842**, 15 classes.)
- **Fase 6 (parcial)**: `src/report_figures.py` + figuras em `reports/` +
  notebook `notebooks/MLP_base.ipynb` com seções 8.x do Transformer.

### Pendente (requer webcam do usuário)

- Fase 1: capturas próprias (letras faltantes f/g/p/q/t + confusas) → retreino MLP.
- Fases 2.2/2.3: palavras próprias (oi, obrigado…) + letras dinâmicas (h/j/k/x/y/z)
  → retreino Transformer com `--extra-dir data/raw/words`.
- Testes ao vivo dos dois modos + vídeo de apresentação.

### Funciona (estado anterior, 2026-07-13)

- **MLP estático (`mlp_static`)**: F1 macro ≈ 0.98 no split de validação do
  Bianka. Cobre 15 letras: `a, b, c, d, e, i, l, m, n, o, r, s, u, v, w`.
- **Pipeline de treino unificado** (`python -m src.train --model <modelo>`):
  augmentation, oversampling, class weights, F1 macro, early stop, scheduler.
- **Pipeline do Transformer implementado**: `TransformerTemporal` em
  `src/models.py`, loader adaptado, dry-run passa (F1=0.84 em dados sintéticos).
- **Inferência ao vivo MLP** (`python -m src.infer`): guardrails de confiança,
  filtro temporal, média móvel de softmax. Bloqueia com erro claro se o config
  aponta pra transformer.
- **Captura via webcam**: `src/capture.py` (estático) e `src/capture_dynamic.py`
  (dinâmico) — ambos salvam `.npy (30, 66)` no formato canônico.
- **Datasets externos**: `src/preprocess_external.py` processa Brazilian Alphabet
  (Bianka) e MINDS-Libras para o mesmo formato canônico.

### Não funciona / não existe

- **Alfabeto estático completo**: faltam `f, g, p, q, t` no dataset.
- **Letras dinâmicas** (J, K, X, Y, Z, e H em algumas variantes): nenhum dado.
- **Palavras** (obrigado, oi, bom dia, etc.): nenhum dado.
- **Treino real do Transformer**: nenhum dataset dinâmico processado ainda.
- **Inferência ao vivo temporal**: `infer.py` só faz single-frame. Sem buffer
  de 30 frames, sem trigger, sem roteamento estático↔dinâmico.
- **Precisão real na webcam** ainda é irregular — F1 val alto, mas domain shift
  entre Bianka (estúdio) e webcam do usuário é o gargalo. Fix: capturas
  próprias do usuário misturadas via `--extra-dir data/raw`.

---

## 2. Meta desta continuação

Cobrir o **alfabeto completo (26 letras)** + **um vocabulário de palavras
frequentes** (mínimo 8-10), com inferência ao vivo estável.

### Divisão por modelo (não negociável — vide ADR-003/004)

| Símbolo | Estático (MLP) | Dinâmico (Transformer) |
|---|---|---|
| a, b, c, d, e, f, g, i, l, m, n, o, p, q, r, s, t, u, v, w | ✅ 19 letras | — |
| h, j, k, x, y, z | — | ✅ 6 letras dinâmicas |
| palavras (obrigado, bom dia, oi, tudo bem, sim, não, por favor, desculpa, ajuda, nome, casa, comida, água…) | — | ✅ |

---

## 3. Backlog priorizado

Cada item tem **Objetivo / Depende de / Arquivos / Comandos / Critério de aceitação**.
Faça em ordem — dependências são reais.

### Fase 1 — Fechar o alfabeto estático (MLP)

Objetivo mais barato e de maior ROI imediato. Não requer código novo, apenas dados.

#### 1.1 Guiar o usuário a gravar letras faltantes + confusas

- **Objetivo**: Completar as letras estáticas do alfabeto e reforçar as
  confusas com dados do próprio usuário. Domínio da webcam dele passa a estar
  representado no dataset — resolve o domain shift documentado em ADR-008.
- **Depende de**: nada.
- **Arquivos**: nenhum (usuário roda `capture.py`).
- **Comandos que o USUÁRIO precisa rodar** (você não tem webcam — apresente e
  peça confirmação de quando terminar):
  ```bash
  # Letras que faltam no Bianka
  python -m src.capture --sinal f --amostras 40
  python -m src.capture --sinal g --amostras 40
  python -m src.capture --sinal p --amostras 40
  python -m src.capture --sinal q --amostras 40
  python -m src.capture --sinal t --amostras 40

  # Letras confusas na webcam (aumentar amostragem)
  python -m src.capture --sinal m --amostras 40
  python -m src.capture --sinal n --amostras 40
  python -m src.capture --sinal d --amostras 40
  python -m src.capture --sinal u --amostras 40
  python -m src.capture --sinal v --amostras 40
  python -m src.capture --sinal r --amostras 40
  python -m src.capture --sinal s --amostras 40
  python -m src.capture --sinal i --amostras 40
  ```
- **Instruções para o usuário**: varie deliberadamente ângulo, distância e
  iluminação a cada ~5 amostras. Sem isso, capturar 40 amostras iguais não ajuda.
- **Critério de aceitação**: `ls data/raw/{f,g,p,q,t,m,n,d,u,v,r,s,i}/` mostra
  ≥ 30 `.npy` em cada.

#### 1.2 Retreinar MLP com Bianka + capturas próprias

- **Objetivo**: MLP com todas as 19 letras estáticas + reforço nas confusas.
- **Depende de**: 1.1 concluído.
- **Arquivos**: nenhum (comando já existe).
- **Comando**:
  ```bash
  python -m src.train --extra-dir data/raw
  ```
- **Critério de aceitação**:
  - Log final imprime **≥ 19 classes**.
  - F1 macro validation ≥ 0.90.
  - **Teste ao vivo real** (`python -m src.infer`) mostra que letras antes
    confusas agora ficam estáveis por 5+ segundos ao serem sinalizadas. Este é
    o critério real — F1 val é métrica secundária.
- **Se F1 val cair muito** (< 0.85): a mistura pode estar desbalanceada.
  Ajuste `HPARAMS["oversample_factor"]` (4 → 6) para dar mais peso à variação,
  ou reduza aug (`rot_range_deg: 25 → 20`) se as capturas próprias já
  incluírem variação natural suficiente.

---

### Fase 2 — Preparar dataset dinâmico (Transformer)

#### 2.1 Baixar e processar MINDS-Libras

- **Objetivo**: obter 20 palavras dinâmicas para treinar o Transformer.
- **Depende de**: nada.
- **Fonte**: `kagglehub` — dataset `j0aopsantos/minds-libras`. Já documentado
  em ADR-005. ~5 GB.
- **Comandos**:
  ```bash
  # Peça permissão ao usuário antes — 5 GB
  python -c "import kagglehub, shutil; \
    p = kagglehub.dataset_download('j0aopsantos/minds-libras'); \
    shutil.move(p, 'data/external/minds-libras')"

  python -m src.preprocess_external --source minds \
    --input data/external/minds-libras \
    --output data/processed/words
  ```
- **Critério de aceitação**: `ls data/processed/words/` mostra 20 subpastas
  (uma por sinal). `python -c "import numpy as np; print(np.load('data/processed/words/aluno/0.npy').shape)"` retorna `(30, 66)`.
- **Cuidado**: o preprocess pode demorar (~15 min em CPU). Rode em background
  ou em terminal separado.

#### 2.2 Gravar palavras adicionais próprias (opcional mas alto ROI)

- **Objetivo**: adicionar palavras não cobertas pelo MINDS (ex: "obrigado",
  "oi", "bom dia" — MINDS tem outras: aluno, amarelo, banco, etc.).
- **Depende de**: nada.
- **Comando por palavra**:
  ```bash
  python -m src.capture_dynamic --sinal obrigado --amostras 30
  python -m src.capture_dynamic --sinal oi --amostras 30
  python -m src.capture_dynamic --sinal bom_dia --amostras 30
  # ... etc
  ```
- **Sugestão de vocabulário mínimo** (o professor tende a valorizar
  cobertura funcional):
  - Saudações: `oi`, `bom_dia`, `boa_tarde`, `boa_noite`, `tchau`
  - Cortesia: `obrigado`, `por_favor`, `desculpa`, `de_nada`
  - Interação: `sim`, `nao`, `ajuda`, `entendi`, `nome`
- **Critério de aceitação**: `data/raw/words/` (ou dir escolhido) tem
  subpastas por palavra com ≥ 20 amostras cada.

#### 2.3 Gravar letras dinâmicas

- **Objetivo**: cobrir H, J, K, X, Y, Z que MLP não pode.
- **Comando por letra**:
  ```bash
  python -m src.capture_dynamic --sinal h --amostras 40
  python -m src.capture_dynamic --sinal j --amostras 40
  python -m src.capture_dynamic --sinal k --amostras 40
  python -m src.capture_dynamic --sinal x --amostras 40
  python -m src.capture_dynamic --sinal y --amostras 40
  python -m src.capture_dynamic --sinal z --amostras 40
  ```
- **Cuidado**: essas letras convivem com palavras no mesmo espaço de classes
  do Transformer. Decida cedo se vai ter 2 modelos temporais (um pra letras,
  outro pra palavras) ou 1 modelo com N+M classes. Recomendação: **1 modelo
  único** — menos manutenção, roteamento no infer não fica ambíguo.

---

### Fase 3 — Treinar o Transformer

#### 3.1 Treino real com dados dinâmicos

- **Objetivo**: primeiro modelo temporal treinado com dado real.
- **Depende de**: 2.1 e/ou 2.2/2.3 concluídos.
- **Comando** (assume MINDS + palavras próprias + letras dinâmicas todos em
  `data/processed/words` e `data/raw/words`):
  ```bash
  python -m src.train --model transformer_temporal \
    --data-dir data/processed/words \
    --extra-dir data/raw/words
  ```
- **Critério de aceitação**:
  - F1 macro validation ≥ 0.85 (mais baixo que MLP é normal — problema
    intrinsecamente mais difícil e dataset menor).
  - Confusion matrix (via `train_log.json` → `final_report`) não mostra
    colapso em 1-2 classes.
- **Tuning esperado**: se overfit (train_acc >> val_acc), reduza:
  - `HPARAMS["transformer"]["d_model"]` 128 → 96
  - `HPARAMS["transformer"]["n_layers"]` 3 → 2
  - Aumente dropout 0.1 → 0.2
- **Se F1 baixo** (< 0.7): capacidade insuficiente ou dataset muito ruidoso.
  Verifique o preprocess: alguns vídeos MINDS têm mão fora do quadro por muito
  tempo. Considere descartar amostras onde `wrist_raw` esteja constante por
  ≥ 25/30 frames (mão parada = amostra degenerada).

---

### Fase 4 — Inferência ao vivo temporal

Essa é a mudança de UX mais visível. Requer novo módulo — **não faça no
`infer.py` existente**, escolha 1 dos 2 caminhos:

**Caminho A (recomendado)** — criar `src/infer_temporal.py`:
- Reusa 90% da estrutura do `infer.py` (webcam, MediaPipe, filtro temporal).
- Diferença: buffer circular de 30 frames de vetores `(66,)`. A cada N
  frames (ex: N=5, dá 6 predições/segundo), expande o buffer para `(30, 86)`
  e roda o Transformer.
- Só ativa a predição quando há mão detectada há ≥ 10 frames contínuos
  (evita disparar em transições).
- UI: mostra a última palavra reconhecida + histórico das últimas 3.

**Caminho B** — unificar em `infer.py`:
- Detecta o modelo pelo config e escolhe o fluxo.
- Mais complexo, mais fácil de quebrar. Só faça se A não for suficiente.

**Pseudo-código do buffer** (para `infer_temporal.py`):

```python
from collections import deque
from src.features import build_feature_vector, expand_sequence_for_model

buffer = deque(maxlen=30)   # 30 vetores (66,)
prev_wrist = None
frames_since_hand = 0
frames_since_pred = 0

while True:
    frame = capture()
    result = landmarker.detect(frame)
    if result.hand_landmarks:
        raw = extract_landmarks(result.hand_landmarks[0])
        normalized, wrist_raw = normalize_landmarks(raw)
        if prev_wrist is None:
            prev_wrist = wrist_raw.copy()
        vec66 = build_feature_vector(normalized, wrist_raw, prev_wrist)
        prev_wrist = wrist_raw.copy()
        buffer.append(vec66)
        frames_since_hand += 1
        frames_since_pred += 1
    else:
        # Perdeu a mão — não zere o buffer, mas conte para trigger
        frames_since_hand = 0
        prev_wrist = None

    # Trigger de predição
    if (
        len(buffer) == 30
        and frames_since_hand >= 10           # mão estável há pelo menos 10 frames
        and frames_since_pred >= 5            # não spam de predição
    ):
        seq = np.stack(list(buffer))          # (30, 66)
        seq86 = expand_sequence_for_model(seq, augment=False)  # (30, 86)
        x = torch.from_numpy(seq86).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        # ... aplicar guardrails (min_conf) como no infer.py estático
        frames_since_pred = 0
```

**Critério de aceitação**:
- Rodar `python -m src.infer_temporal --model models/transformer_temporal.pth
  --config models/transformer_temporal_config.json` e sinalizar "obrigado" 3x
  em sequência. Deve reconhecer ≥ 2/3 (a 3ª falha ainda é aceitável dado o
  domain shift esperado).

---

### Fase 5 — Roteamento estático vs dinâmico ao vivo

**Problema**: usuário sinaliza um "A" (estático) → deve entrar o MLP.
Sinaliza "obrigado" (dinâmico) → deve entrar o Transformer. Sem roteamento, o
sistema não sabe qual modelo consultar.

**Duas opções**:

**Opção 1 (simples e prática)**: sistema com **modo manual** — usuário aperta
tecla para alternar entre "modo letra" e "modo palavra". O `infer_temporal.py`
implementa os dois pipelines internamente, tecla `l` / `p` alterna.

**Opção 2 (elegante mas complexa)**: detector de movimento. Se `delta` do
pulso ao longo dos últimos 30 frames tiver norma < ε, é estático → MLP. Se
tiver norma > ε, é dinâmico → Transformer.
- Risco: threshold ε é sensível a ruído; falso positivo faz MLP receber um
  frame de movimento e chutar errado, ou vice-versa.

**Recomendação**: comece pela Opção 1 (barra de espaço alterna modo). É
suficiente para o vídeo de apresentação e evita bugs sutis.

**Critério de aceitação**: demo consegue traduzir a frase "obrigado A B C" em
Libras — Transformer reconhece "obrigado", usuário aperta espaço, MLP
reconhece cada letra individualmente.

---

### Fase 6 — Notebook final e apresentação

O backlog original prevê `notebooks/01_baseline_mlp.ipynb` como artefato
científico. Depois de todas as fases anteriores:

- Expandir para incluir seções do Transformer.
- Gerar matriz de confusão para os dois modelos.
- Plotar loss/F1 curves de `train_log.json`.
- Incluir 3-4 screenshots ou GIFs curtos da inferência ao vivo.
- Comparar hiperparâmetros testados (fica bonito ter uma tabela de ablação).

Estimativa: 4-6h de trabalho de escrita + geração de figuras.

---

## 4. Anti-padrões (não repita)

Coisas que já foram tentadas e reprovadas em sessões anteriores. Não perca
tempo revisitando:

- ❌ **Baixar mais datasets de estúdio esperando resolver a webcam.** MINDS,
  V-LIBRASIL, LSWH100 — todos gravados em ambiente controlado. Não fecham o
  domain shift. Única solução: capturas do próprio usuário.
- ❌ **Aumentar augmentation acima de ±30° de rotação, ±20% de escala.** Passa
  do ponto onde o modelo aprende invariâncias reais e começa a inventar
  confusões novas. F1 val cai e webcam não melhora.
- ❌ **Trocar mean pooling por CLS token no Transformer.** Testado, sem ganho,
  aumenta risco de bottleneck com dataset pequeno (~1000 amostras).
- ❌ **CNN sobre pixels da webcam.** Proibido pelo edital. Todo o design é
  landmarks-based.
- ❌ **Reprocessar dataset com formato diferente de `(30, 66)`.** Quebra todos
  os `.npy` já processados. As features geométricas (20 extras) e augmentation
  acontecem no loader, não em disco.
- ❌ **Amend em commits publicados / force-push.** Preferir novo commit.
- ❌ **Excluir sem checar** arquivos desconhecidos em `data/raw/` — podem ser
  capturas do usuário em progresso. Sempre `ls` antes.

---

## 5. Arquivos e comandos de referência

### Estrutura relevante
```
src/
  capture.py                 — grava letra estática via webcam
  capture_dynamic.py         — grava sinal dinâmico via webcam
  preprocess_external.py     — Bianka (imagens) e MINDS (vídeos) → .npy (30,66)
  features.py                — normalização, augmentation, expansão para 86
  constants.py               — FEATURE_DIM=66, FEATURE_DIM_EXTENDED=86, FRAMES=30
  dataset.py                 — LibrasDataset (static/sequence), stratified_split
  models.py                  — MLPStatic, TransformerTemporal, MODEL_REGISTRY
  train.py                   — HPARAMS + pipeline; --model escolhe o alvo
  infer.py                   — inferência ao vivo (só MLP por enquanto)

spec/
  backlog.md                 — status por participante
  decisions/ADR-001..009.md  — decisões arquiteturais (LEIA)
  roadmap-completude.md      — este documento

data/
  external/                  — datasets brutos baixados (não subir no git)
  processed/                 — .npy (30,66) processados
  raw/                       — capturas do usuário (não subir no git)

models/
  {model_name}.pth           — checkpoints
  model_config.json          — arquitetura do último treinado (para infer)
  label_map.json             — {classe: idx}
  train_log.json             — HPARAMS + history + report do último treino
```

### Comandos frequentes
```bash
# Setup
source .venv/bin/activate

# Smoke test rápido (sem dados reais)
python -m src.train --dry-run                          # MLP
python -m src.train --dry-run --model transformer_temporal

# Treinos reais
python -m src.train                                     # MLP com defaults
python -m src.train --extra-dir data/raw               # MLP + capturas próprias
python -m src.train --model transformer_temporal \
  --data-dir data/processed/words                       # Transformer com MINDS

# Inferência
python -m src.infer                                     # MLP ao vivo
python -m src.infer --min-conf 0.75 --stable-frames 5  # afrouxa filtros

# Captura
python -m src.capture --sinal <letra> --amostras N
python -m src.capture_dynamic --sinal <palavra> --amostras N
```

---

## 6. Checklist final de entrega (para o vídeo de apresentação)

Não é técnico, mas serve de norte para saber "acabou":

- [ ] MLP treinado com ≥ 19 letras estáticas, F1 val ≥ 0.90.
- [ ] Transformer treinado com ≥ 20 sinais dinâmicos, F1 val ≥ 0.85.
- [ ] Inferência ao vivo do MLP: reconhece letras individuais com < 1s de latência.
- [ ] Inferência ao vivo do Transformer: reconhece palavras dinâmicas.
- [ ] `notebooks/01_baseline_mlp.ipynb` atualizado com resultados dos dois modelos.
- [ ] Matriz de confusão para MLP + Transformer geradas.
- [ ] ADRs de todas as decisões novas registradas.
- [ ] README.md atualizado com comandos das duas famílias.
- [ ] Vídeo de 10min com demonstração funcional.

---

## 7. Dicas operacionais para o próximo Claude

- **Use as tasks** (TaskCreate/TaskUpdate) desde o começo — este projeto vive
  em iterações longas e o histórico ajuda a não repetir passos.
- **Faça dry-run antes de treino real** sempre que mexer em `dataset.py`,
  `features.py` ou `train.py`.
- **Não sobrescreva `models/*.pth` sem backup** se estiver testando um modelo
  novo — `mv models/mlp_static.pth models/mlp_static.pth.bak` primeiro.
- **Ao criar novo ADR**, seguir o padrão dos existentes: Contexto → Decisão →
  Justificativa → Alternativas Descartadas → Referências. Numeração
  sequencial: próximo é ADR-010.
- **Datasets grandes**: peça confirmação explícita antes de baixar > 1 GB.
- **Se o usuário disser "tá confuso ao vivo"** de novo, o caminho SEMPRE é
  capturas próprias — nunca mais engenharia de código. Isso já foi tentado
  três vezes.
