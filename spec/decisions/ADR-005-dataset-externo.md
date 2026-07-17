# ADR-005 — Usar datasets externos processados pelo próprio pipeline

**Data:** 2026-07-09
**Status:** Aceita
**Contexto:** Diogo

---

## Contexto

O backlog original previa que a equipe gravasse o dataset inteiro com `capture.py` e
`capture_dynamic.py`. Restam ~8 dias até o prazo (17/07/2026) e a coleta manual tem três
problemas:

- **Diversidade zero:** todas as amostras viriam de 1–3 sinalizadores da equipe, sem
  variação de biotipo, ângulo, iluminação — modelo com overfit garantido.
- **Custo de tempo:** 15 letras × 50 amostras × 5 segundos ≈ 1h de gravação só do alfabeto,
  sem contar palavras dinâmicas. Bloqueia o desenvolvimento do modelo.
- **Reprodutibilidade:** um relatório científico sem dataset público não é reproduzível.

## Decisão

Usar dois datasets públicos processados pelo nosso pipeline MediaPipe em batch,
gerando `.npy` no formato canônico `(30, 66)` que o resto do sistema já espera:

1. **Brazilian Sign Language Alphabet** (biankatpas — GitHub) — 4.411 imagens de 15 letras
   estáticas do alfabeto de Libras.
2. **MINDS-Libras** (Kaggle / Zenodo) — 1.158 vídeos de 20 sinais dinâmicos executados por
   12 sinalizadores, 5 repetições cada.

O script `src/preprocess_external.py` faz a conversão:

- **Imagens** (alfabeto): MediaPipe `RunningMode.IMAGE` → landmarks → normalização →
  vetor `(66,)` replicado 30x com delta zero (sinal estático).
- **Vídeos** (MINDS): MediaPipe `RunningMode.VIDEO` frame a frame → reamostra 30 frames
  uniformemente do trecho com mão detectada → deltas do pulso entre pares consecutivos.

A equipe **ainda pode** gravar amostras com `capture.py` para complementar (mesmo formato,
mesma pasta `data/raw/`) — mas isso deixa de ser bloqueante.

## Justificativa

- **Desbloqueio imediato do modelo:** o Participante 1 pode treinar sem esperar a coleta.
- **Diversidade de sinalizadores:** 12 pessoas no MINDS-Libras vs. 3 na equipe.
- **Reprodutibilidade:** qualquer avaliador pode baixar os datasets e rodar
  `preprocess_external.py` + `train.py` para reproduzir os resultados.
- **Compatibilidade com o pipeline existente:** os `.npy` de saída têm exatamente o mesmo
  shape `(30, 66)` dos gravados pela equipe. Zero refactor no `dataset.py`, no `train.py`
  ou no `infer.py`.

## Trade-offs aceitos

- **Domain shift para o demo ao vivo.** O treino ocorre em imagens/vídeos de fundo controlado;
  a inferência acontece com a webcam da equipe. Mitigação: a normalização por translação no
  pulso + escala pela palma (ADR-002) elimina invariância de posição/tamanho — o que sobra é
  ruído de fundo, mas o MediaPipe já isola a mão antes do nosso pipeline.
- **Vídeos do MINDS podem ter mão perdida em alguns frames.** Aceitável: o preprocessador
  descarta o vídeo se detectar mão em menos de 5 frames.
- **Alguns sinais do MINDS-Libras usam duas mãos**, mas o pipeline captura só uma
  (`num_hands=1`). Filtragem manual dos sinais compatíveis no momento do preprocess.

## Reprodutibilidade

```bash
# 1. Alfabeto
git clone https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset.git \
    data/external/brazilian-alphabet
python -m src.preprocess_external --source alphabet \
    --input data/external/brazilian-alphabet \
    --output data/processed/alphabet

# 2. MINDS-Libras (subset Zenodo — ver adendo 2026-07-14 abaixo)
mkdir -p data/external/minds-zips
for s in 07 03 01; do
  curl -L --fail -o data/external/minds-zips/Sinalizador$s.zip \
    "https://zenodo.org/api/records/2667329/files/Sinalizador$s.zip/content"
done
# extrair e organizar por label em data/external/minds-libras/<sinal>/*.mp4
python -m src.preprocess_external --source minds \
    --input data/external/minds-libras \
    --output data/processed/words

# 3. Treino
python -m src.train --data-dir data/processed/alphabet
```

## Adendo (2026-07-14) — subset Zenodo em vez do Kaggle completo

O dataset no Kaggle (`j0aopsantos/minds-libras`) é servido como **um único
archive de 44,5 GB** (não os ~5 GB estimados originalmente) e não permite
download parcial. Por restrição de banda/disco definida pelo Diogo (**≤ 15 GB**),
a fonte passa a ser o **Zenodo record 2667329** (RGB-only, MP4 1080p), que é
particionado por sinalizador — permitindo baixar um subconjunto:

- `Sinalizador07.zip` (2,5 GB) + `Sinalizador03.zip` (3,7 GB) +
  `Sinalizador01.zip` (4,4 GB) = **10,6 GB**.
- Cobertura: 20 sinais × 3 sinalizadores × 5 repetições ≈ 300 vídeos
  (~15 amostras/classe antes de augmentation).

Armadilhas de reprodutibilidade encontradas (2026-07-14):

- As URLs de download são as da **API do record versionado**
  (`/api/records/2667329/files/...`); as URLs "bonitas" do record 2667328
  retornam 404.
- **Sempre validar md5** contra o manifest da API antes de extrair.
- Zips **> 4 GiB** (Sinalizador01, 11, 12) têm ZIP64 malformado (Archive
  Utility do macOS): `zipfile` do Python falha mesmo com md5 correto —
  extrair com `7z` (ver docstring de `src/organize_minds.py`).

Trade-off aceito: menos diversidade de sinalizadores (3 de 12) → F1 do
Transformer tende a ficar abaixo do alvo de 0.85 até ser complementado com
capturas próprias (`capture_dynamic.py`), que já eram necessárias contra o
domain shift de qualquer forma. Mais sinalizadores podem ser adicionados
incrementalmente baixando outros zips (o preprocess continua do índice onde
parou por classe).

## Referências

- Brazilian Sign Language Alphabet Dataset — https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset
- MINDS-Libras (Kaggle) — https://www.kaggle.com/datasets/j0aopsantos/minds-libras
- MINDS-Libras (Zenodo) — https://data.niaid.nih.gov/resources?id=zenodo_2667328
