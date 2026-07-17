# ADR-011 — Modelo único letras + palavras (fim do toggle manual)

**Data:** 2026-07-15
**Status:** Revertida (2026-07-16) — ver ADR-012. Histórico: aceita em 2026-07-15 (critérios U3 atendidos; validação ao vivo U5 aprovada pelo Diogo em 2026-07-16)
**Contexto:** Diogo

---

## Contexto

A ADR-010 resolveu a inferência ao vivo com **dois modelos** (MLP estático +
TransformerTemporal) roteados manualmente pela barra de espaço. Funciona, mas
tem um custo de UX visível no demo: o usuário precisa saber de antemão se o
próximo sinal é letra ou palavra e apertar uma tecla entre "soletrar OI" e
"sinalizar obrigado". Para o vídeo de apresentação, isso quebra a narrativa de
"tradutor em tempo real".

Três fatos novos tornaram o modelo único viável, revertendo a análise da
ADR-010 (que o rejeitou):

1. **O discriminante estático↔dinâmico está nos dados.** Toda amostra em disco
   é `(30, 66)` com Δ do pulso nas features 63–65 (ADR-004). Sinais estáticos
   têm Δ≈0 nos 30 frames; dinâmicos têm 2–6× mais energia de movimento (medido
   no DATA_LUIZ). O modelo pode aprender o roteamento que a ADR-010 delegava a
   uma tecla.
2. **O Transformer já domina letras dinâmicas** (h/j/k/x/y/z com F1 = 1.00 no
   modelo especializado de 59 classes) — não há evidência de que misturar
   letras a palavras degrade o mecanismo de atenção.
3. **`max_per_class` (fase U2) removeu o bloqueio de desbalanceamento**: sem
   teto, o dataset unificado tinha 200× de razão entre classes (563 "b" Bianka
   vs 3 "eu_amo_voce" V-LIBRASIL), que class weight sozinho não segura.
   Com teto 80 + roots da webcam protegidos (`data/raw`, `data/raw/words`),
   a razão cai para ~40× nas classes raras, dentro do que o weighting cobre.

Fusões de classe no espaço unificado (79 classes): `y` (perfil estático do
Diogo + dinâmico do Luiz → uma classe), `banheiro` (MINDS 15 + V-LIBRASIL 3).

## Decisão

**1. Um único TransformerTemporal treinado no dataset unificado** — 26 letras
+ 53 palavras (79 classes), todos os roots num só treino:

```bash
python -m src.train --model transformer_temporal \
  --data-dir data/processed/words \
  --extra-dir data/processed/alphabet data/raw data/luiz_split/static data/luiz_split/dynamic
```

**2. Novo módulo `src/infer_unified.py`** — mesmo pipeline temporal da
ADR-010 (buffer de 30 frames, expansão 66→86 só na predição, trigger com
guardas, estabilidade por dupla concordância), **sem toggle de modo**. UX
nova: **buffer de soletração** — letras estáveis acumulam ("O-I"), palavras
entram por extenso e fecham o bloco de soletração; pausa sem mão (~20 frames)
também fecha; mesmo sinal só recomita após cooldown de 2 s.

**3. Aceitação por grupo, não por F1 global** (`src/eval_groups.py`): o F1
macro global é ilegível com classes de 3 amostras pesando igual a letras com
centenas. Critérios (roadmap-unificacao.md, U3):

| Grupo | Critério | Referência pré-unificação | Unificado (2026-07-15) |
|---|---|---|---|
| Letras estáticas (20) | F1 ≥ 0.95 | MLP especializado: 0.9947 | **0.9839** ✓ |
| Letras dinâmicas (6) | F1 ≥ 0.95 | Transformer especializado: 1.00 | **1.0000** ✓ |
| Palavras MINDS (20) | F1 ≥ 0.80 | 0.8064 | **0.8058** ✓ |
| Palavras V-LIBRASIL (33) | F1 > 0 | 0.1919 (2–3 amostras/classe) | **0.1263** ✓ |

Treino de 90 épocas (melhor checkpoint na 78), 79 classes, val acc 0.9389.
Custo da unificação: −0.011 nas letras estáticas vs o MLP e −0.066 no
V-LIBRASIL (esperado: 2–3 amostras/classe se diluem em 79 classes); MINDS e
letras dinâmicas sem regressão. `max_per_class=80` foi suficiente — o
fallback de 150 não foi necessário.

**4. O baseline NÃO morre.** `mlp_static` e `infer_temporal.py` permanecem no
repo: o MLP é o baseline do estudo de ablação (exigência metodológica do
relatório) e o `infer_temporal.py` é o fallback documentado caso o modelo
único regrida ao vivo. A unificação é decisão de **UX/deploy**, não de
metodologia.

## Justificativa

- **UX do demo**: soletrar e sinalizar sem trocar de modo é exatamente o que
  "tradutor em tempo real" promete; o toggle manual era um artefato da
  arquitetura, não do problema.
- **Menos artefatos no caminho crítico**: um só `.pth` + config + label_map
  para carregar, um só espaço de classes para depurar ao vivo.
- **Roteamento aprendido > roteamento por threshold**: a alternativa de
  detector de movimento (rejeitada na ADR-010 por sensibilidade a ruído)
  continua rejeitada — a diferença é que agora o *modelo* aprende o
  discriminante a partir dos mesmos dados, sem ε hardcoded.
- **Risco controlado**: critérios de aceitação por grupo com STOP explícito —
  se letras estáticas ficarem < 0.90 mesmo com `max_per_class=150`, a
  arquitetura dual da ADR-010 permanece a oficial.

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| Dual com toggle manual (ADR-010, status quo) | UX ruim no demo: exige tecla entre soletração e palavra; usuário precisa conhecer a taxonomia interna do sistema |
| Dual com roteador por movimento (norma do Δ pulso) | Mesmo motivo da ADR-010: threshold ε sensível a ruído de webcam; falso positivo roteia para o modelo errado e gera erro silencioso difícil de depurar ao vivo |
| Unificar e deletar o MLP baseline | Joga fora o baseline do estudo de ablação — o relatório perde a comparação metodológica que justifica a arquitetura |
| Dois modelos em ensemble (média de softmax) | Espaços de classe disjuntos tornam o ensemble mal-definido; dobra o custo de inferência por frame sem resolver o roteamento |

## Referências

- `spec/roadmap-unificacao.md` — fases U1–U5, critérios de aceitação.
- ADR-004 (janela 30 frames, Δ do pulso), ADR-007 (expansão 66→86 no consumo),
  ADR-009 (TransformerTemporal), ADR-010 (arquitetura dual — agora fallback).
- `src/dataset.py` (`max_per_class`), `src/eval_groups.py`,
  `src/infer_unified.py`.
