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

## Captura de Dataset

```bash
python src/capture.py --sinal A --amostras 50
```

### Teclas

| Tecla | Ação |
|-------|------|
| `ESPAÇO` | Grava 1 amostra (30 frames ≈ 1 segundo) |
| `u` | Desfaz a última amostra salva |
| `q` | Encerra |

### Formato dos dados

Cada amostra é salva em `data/raw/{sinal}/{idx}.npy`:
- **Shape:** `(30, 66)` — 30 frames × 66 features
- `[0:63]` — 21 landmarks normalizados (x, y, z)
- `[63:66]` — Δ(x, y, z) do pulso em relação ao frame anterior

---

## Decisões de arquitetura

Documentadas em [`spec/decisions/`](spec/decisions/).
