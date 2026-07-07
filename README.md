# Tradutor de Libras — Visão Computacional

> POC de tradução de Libras em tempo real via extração de landmarks com MediaPipe.
> Disciplina: Tópicos Especiais em Matemática Aplicada — Visão Computacional com Deep Learning (UnB | FCTE)
> Professor: Vinicius Rispoli | Prazo: 17/07/2026

---

## Visão Geral

Este projeto implementa um tradutor de Libras usando **extração de marcos espaciais (landmarks)** em vez de classificação direta de imagens (proibida pelo edital). A abordagem é conduzida como um **Estudo de Ablação**, comparando um modelo estático (baseline) com um modelo temporal (Transformer), usando os mesmos dados.

### Pipeline

```
Webcam (OpenCV)
    ↓
MediaPipe Hand Tracking
  → 21 landmarks 3D por frame
    ↓
Normalização Espacial
  → Translação: pulso como origem (0,0,0)
  → Escala: distância pulso→dedo médio = 1.0
  → Delta: Δ(x,y,z) do pulso entre frames (captura movimento)
    ↓
Buffer de 30 frames → tensor (30, 66)
    ↓
.npy salvo em data/raw/{sinal}/{idx}.npy
    ↓
[Modelo Baseline]       [Modelo Temporal]
tensor[:, 15, :]        tensor completo
shape (66,)             shape (30, 66)
MLP                     Transformer
```

---

## Estrutura do Projeto

```
libras-cv-translator/
├── src/
│   └── capture.py        # Pipeline de captura (webcam → .npy)
├── data/
│   └── raw/              # Dataset bruto por sinal (ignorado pelo git)
│       ├── a/
│       │   ├── 0.npy
│       │   └── ...
│       └── obrigado/
│           └── ...
├── models/               # Modelos pré-treinados do MediaPipe (ignorado pelo git)
│   └── hand_landmarker.task
├── notebooks/            # Jupyter Notebooks para treino e análise
├── spec/
│   ├── backlog.md        # Especificação técnica e backlog de tarefas
│   └── decisions/        # Registro de Decisões de Arquitetura (ADRs)
├── requirements.txt
└── .gitignore
```

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
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 3. Baixar o modelo do MediaPipe

O arquivo de modelo não está no repositório (muito pesado). Baixe com:

```bash
mkdir -p models
python3 -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'models/hand_landmarker.task'
)
print('Modelo baixado com sucesso.')
"
```

---

## Captura de Dataset

```bash
# Ativa o ambiente virtual
source .venv/bin/activate

# Captura 50 amostras do sinal "A"
python src/capture.py --sinal A --amostras 50

# Captura 50 amostras de "obrigado"
python src/capture.py --sinal obrigado --amostras 50
```

### Teclas durante a captura

| Tecla | Ação |
|-------|------|
| `ESPAÇO` | Inicia gravação de 1 amostra (30 frames ≈ 1 segundo) |
| `u` | Desfaz a última amostra salva (apaga o arquivo) |
| `q` | Encerra o script |

### Formato dos dados

Cada amostra é salva como um tensor NumPy:
- **Shape:** `(30, 66)` — 30 frames × 66 features
- **Features por frame:**
  - `[0:63]` — 21 landmarks normalizados (x, y, z) × 21 pontos
  - `[63:66]` — Δ(x, y, z) do pulso em relação ao frame anterior

### Verificar amostras gravadas

```bash
python3 -c "
import numpy as np, glob
for f in sorted(glob.glob('data/raw/**/*.npy', recursive=True)):
    t = np.load(f)
    print(f'{f}: shape={t.shape}')
"
```

---

## Dataset — Sinais Planejados

| Categoria | Sinais | Amostras por sinal | Total |
|-----------|--------|-------------------|-------|
| Estático (letras) | A–Z (26) | 50 | 1.300 |
| Dinâmico (palavras) | 10 sinais | 50 | 500 |
| **Total** | **36** | **50** | **1.800** |

---

## Regras de Versionamento

- **Nunca suba** para o git: arquivos `.npy`, `.csv`, `.venv/`, `models/`
- O `.gitignore` já está configurado para bloquear esses arquivos
- O Google Colab é usado **somente** para treino e notebook final
- Funções auxiliares ficam em `src/` (importáveis pelo notebook)
