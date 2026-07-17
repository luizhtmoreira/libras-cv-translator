# Tradutor de Libras — Visão Computacional

> POC de tradução de Libras em tempo real via extração de landmarks com MediaPipe.
> Disciplina: Tópicos Especiais em Matemática Aplicada — Visão Computacional com Deep Learning (UnB | FCTE)
> Professor: Vinicius Rispoli | Prazo: 17/07/2026

---

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/luizhtmoreira/libras-cv-translator.git
cd libras-cv-translator
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 3. Baixar o modelo do MediaPipe

O arquivo de modelo não sobe para o git (pesado demais). Baixe com:

```bash
mkdir -p models
python3 -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'models/hand_landmarker.task'
)
print('Modelo baixado.')
"
```

---

## Captura de Dataset (opcional — datasets externos disponíveis)

### Alfabeto (sinais estáticos) — modo burst

```bash
python -m src.capture --sinal A --amostras 40
# opcional: --fotos 5 --intervalo 1.0 (defaults) | --modo janela (legado, 30 frames)
```

ESPAÇO dispara uma **rajada de 5 fotos com 1 s de intervalo** (countdown e flash
na tela). Cada foto vira uma amostra estática `(30, 66)`. Fluxo recomendado:
segure a pose durante a rajada; **entre rajadas, varie ângulo, distância e
iluminação** — 8 rajadas = 40 amostras com boa diversidade.

### Letras dinâmicas (h, j, k, x, y, z — sinais com movimento)

```bash
python -m src.capture_dynamic --sinal j --amostras 50
```

Grava janelas de 30 frames em `data/raw/dynamic/{sinal}/`.

### Teclas

| Tecla | Ação |
|-------|------|
| `ESPAÇO` | Dispara rajada de fotos (`capture`, burst) ou grava 30 frames (`capture_dynamic` / `--modo janela`) |
| `u` | Desfaz a última rajada/amostra salva |
| `q` | Encerra |

### Formato dos dados

Cada amostra é salva em `data/raw/{sinal}/{idx}.npy`:
- **Shape:** `(30, 66)` — 30 frames × 66 features
- `[0:63]` — 21 landmarks normalizados (x, y, z)
- `[63:66]` — Δ(x, y, z) do pulso em relação ao frame anterior

---

## Datasets externos (recomendado)

Em vez de gravar tudo à mão, use os públicos processados pelo próprio pipeline
(ver [ADR-005](spec/decisions/ADR-005-dataset-externo.md)):

```bash
# Alfabeto (imagens estáticas)
git clone https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset.git \
    data/external/brazilian-alphabet
python -m src.preprocess_external --source alphabet \
    --input data/external/brazilian-alphabet \
    --output data/processed/alphabet
```

Produz `.npy` `(30, 66)` — mesmo formato do `capture.py`.

---

## Treino

```bash
# Smoke test (dados sintéticos, sem GPU, ~30s)
python -m src.train --dry-run
python -m src.train --dry-run --model transformer_temporal

# MLP estático (21 letras) — hyperparams no bloco HPARAMS de src/train.py
python -m src.train --data-dir data/processed/alphabet --extra-dir data/raw data/luiz_split/static

# Transformer temporal (6 letras dinâmicas: h/j/k/x/y/z)
python -m src.train --model transformer_temporal --data-dir data/luiz_split/dynamic
```

Artefatos ficam em `models/`, **separados por modelo** (ADR-010) para que MLP e
Transformer coexistam:
- `{modelo}.pth` — pesos do melhor checkpoint (maior F1 macro em validação)
- `{modelo}_label_map.json` — `{classe: idx}` do espaço de classes daquele modelo
- `{modelo}_config.json` — arquitetura para reconstruir sem hardcode
- `{modelo}_train_log.json` — HPARAMS + histórico + report final
- `label_map.json` / `model_config.json` / `train_log.json` — aliases legados,
  escritos apenas pelo MLP (defaults do `infer.py`)

---

## Inferência ao vivo

```bash
# Alfabeto completo: letras estáticas + dinâmicas, toggle por tecla (recomendado — ADR-010/012)
python -m src.infer_temporal

# Só letras estáticas (MLP)
python -m src.infer
```

No `infer_temporal.py` (entrada principal), a **barra de espaço alterna** entre
modo LETRA (MLP frame a frame, 21 letras estáticas) e modo LETRA DINÂMICA
(Transformer sobre janela de 30 frames, ~6 predições/s, classes h/j/k/x/y/z).
Requer os dois modelos treinados em `models/`. A letra `y` existe nos dois
modos — o toggle decide qual modelo responde.

No `infer.py`, a letra reconhecida aparece sobre a mão (verde = predição estável,
amarelo = ainda instável). `q` sai.

---

## Notebook

`notebooks/relatorio_final.ipynb` — **relatório científico completo** (introdução,
métodos, resultados, conclusão e referências), com justificativa de estratégia,
arquiteturas e otimização. Roda no Colab: monta o Drive, clona o repo e descompacta
`libras_dataset_drive.zip` (gere com `bash scripts/make_drive_zip.sh` e suba para
`MyDrive/libras/`).

---

## Decisões de arquitetura

Documentadas em [`spec/decisions/`](spec/decisions/) — ADRs 001 a 012.
Escopo atual (ADR-012): **apenas o alfabeto**, arquitetura dual — MLP para as
21 letras estáticas + Transformer temporal para as 6 letras dinâmicas.
