# Especificação Técnica e Backlog: Tradutor de Libras (Visão Computacional)

## 1. Visão Geral do Projeto
* **Disciplina:** Tópicos Especiais em Matemática Aplicada: Visão Computacional com Deep Learning (UnB | FCTE).
* **Professor:** Vinicius Rispoli.
* **Prazo Limite:** 17 de julho de 2026.
* **Objetivo:** Desenvolver uma Prova de Conceito (POC) de um tradutor de Libras em tempo real.
* **Restrição Principal:** O edital proíbe abordagens exclusivamente de classificação de imagens.

## 2. Justificativa Científica e Metodologia
Para atender ao rigor exigido no trabalho (estruturação em introdução, métodos, resultados e conclusão), este projeto será conduzido como um **Estudo de Ablação**. 

Em vez de utilizar arquiteturas pesadas de segmentação semântica, utilizaremos a extração de marcos espaciais (*landmarks*) via **MediaPipe** como técnica de isolamento geométrico. A hipótese é que o mapeamento espacial em janelas temporais permite a tradução eficiente de sinais dinâmicos, garantindo viabilidade de execução em tempo real (*deploy* leve).

## 3. Escopo da Prova de Conceito (POC)
Para garantir a entrega no prazo, o dicionário da POC será restrito e focado em demonstrar a capacidade da arquitetura:
* **Sinais Estáticos (1 Frame):** Alfabeto básico (A, B, C, etc.) de frente para a câmera — 21 letras no MLP.
* **Sinais Dinâmicos (Janela de Tempo):** As letras do alfabeto que envolvem movimento — **H, J, K, X, Y, Z** — classificadas pelo Transformer temporal sobre janelas de 30 frames. *(Revisão 2026-07-16, ADR-012: palavras com movimento, ex. "Obrigado", saíram do escopo; o componente temporal da arquitetura continua demonstrado nas letras dinâmicas.)*
* **Fora do Escopo (ADR-012, 2026-07-16):** Palavras e expressões (MINDS-Libras / V-LIBRASIL) — removidas do pipeline para maximizar a precisão nas letras (modelos especializados: MLP F1=0.9947 vs 0.9839 no unificado).
* **Fora do Escopo:** Sinais direcionais que dependem do ângulo Z da palma (ex: "Nome"), para evitar explosão da dimensionalidade do dataset.
* **Fora do Escopo:** Sinais que envolvem duas mãos simultaneamente. O pipeline captura `num_hands=1`; suporte a 2 mãos exigiria dobrar o feature vector (132d), definir uma estratégia de padding para mão ausente e resolver a ambiguidade de atribuição esquerda/direita — complexidade incompatível com o prazo.

## 4. Arquitetura da Solução
1.  **Entrada:** Captura de vídeo via webcam (OpenCV).
2.  **Pré-processamento (Isolamento):** Extração de 21 coordenadas 3D (X, Y, Z) da mão utilizando MediaPipe Hand Tracking.
3.  **Vetorização:** Armazenamento sequencial em tensores NumPy (`.npy`) padronizados em janelas de 30 *frames* (1 segundo).
4.  **Modelagem:** Rede neural sequencial (Transformer ou MLP com janela deslizante) para classificar o tensor espaço-temporal.

---

## 5. Backlog de Tarefas (Distribuição Sugerida)

Abaixo estão as frentes de trabalho. Os membros devem assumir as responsabilidades conforme a disponibilidade, garantindo que não haja gargalos.

### Participante 1 (Foco: Modelagem IA e Estudo de Ablação)
*Carga pesada de iteração contínua.*
* [x] Configurar a arquitetura base da rede neural (PyTorch) recebendo tensores `(Amostras, Frames, Coordenadas)` — `src/dataset.py`, `src/models.py`, `src/train.py`.
* [x] Gerar dados sintéticos (*mocks*) com NumPy para testar o fluxo da rede antes da gravação real — flag `--dry-run` em `src/train.py`.
* [x] Treinar o *Baseline* (Modelo Estático) para sinais de 1 frame — F1 macro=0.9863 no split de validação do Brazilian Alphabet (2026-07-13).
* [x] Métricas de avaliação com F1 macro + weighted, balanced accuracy e classification_report por classe (2026-07-13).
* [x] Augmentation de landmarks para mitigar domain shift Bianka→webcam — ADR-006 (2026-07-13).
* [x] Features geométricas rotação-invariantes (66→86 dims) — ADR-007 (2026-07-13).
* [x] Suporte a mesclar diretórios de dados (`--extra-dir` em train.py) para complementar com capturas próprias — (2026-07-13).
* [ ] Coletar capturas próprias com `capture.py` para letras confusas (M, N) e retreinar mesclando com Bianka.
* [x] Desenvolver o Modelo Temporal (Transformer) — implementado em `src/models.py::TransformerTemporal`, pipeline em `src/train.py --model transformer_temporal` (ADR-009). Smoke-test validado (F1=0.84 em dados sintéticos, 5 epochs).
* [x] Treinar o modelo temporal com dataset dinâmico real — F1 macro=0.8474 em 20 palavras do MINDS-Libras (subset de 3 sinalizadores, 288 amostras; ablação registrada em `src/train.py`) (2026-07-14). **[Revertido em 2026-07-16 — escopo letras-only, ADR-012]**
* [x] Expansão de vocabulário com V-LIBRASIL (2026-07-15): subset curado de 35 expressões (105 vídeos, 261 MB via `src/fetch_vlibrasil.py` — download seletivo, não os 10,8 GB). Transformer retreinado com **59 classes**: letras dinâmicas F1=1.00 (sem regressão), MINDS F1 médio=0.81, expressões novas F1 médio=0.19 (2–3 amostras/classe — esperado; ativam com capturas próprias). Checkpoint 26-classes preservado em `transformer_temporal.pth.bak2` para rollback em 1 comando. Classe `comer` removida (1 amostra útil). **[Revertido em 2026-07-16 — escopo letras-only, ADR-012]**
* [x] Retreino com dados da equipe (2026-07-14): MLP com Bianka + capturas do Diogo + 20 letras estáticas do DATA_LUIZ → **F1=0.9947, 21 classes**. Transformer com MINDS + letras dinâmicas h/j/k/x/y/z do DATA_LUIZ → **F1=0.8275, 26 classes; letras dinâmicas todas com F1=1.00**. Roteamento estático/dinâmico do DATA_LUIZ decidido por medição do delta do pulso (dinâmicas têm 2–6× o movimento). Palavras MINDS com support baixo (n=3 no val) oscilaram — melhora esperada com capturas próprias de palavras.
* [x] Criar `infer_temporal.py` para inferência ao vivo temporal — buffer de 30 frames + trigger + roteamento manual letra↔palavra por tecla (ADR-010, 2026-07-14).
* [x] Unificação letras + palavras num modelo único (ADR-011, 2026-07-15): `max_per_class` em `dataset.py` (subamostragem com seed fixa, roots da webcam protegidos), avaliação por grupo em `src/eval_groups.py`, inferência sem toggle em `src/infer_unified.py` com buffer de soletração. Roadmap em `spec/roadmap-unificacao.md`. **[Revertido em 2026-07-16 — escopo letras-only, ADR-012]**
* [x] Treino unificado (79 classes, 2026-07-15) — critérios U3 **todos atendidos**: letras estáticas F1=0.9839 (≥0.95), letras dinâmicas F1=1.00, MINDS F1=0.8058 (sem regressão vs 0.8064), V-LIBRASIL F1=0.1263 (>0). 90 épocas em CPU, melhor checkpoint na 78. Ver `python -m src.eval_groups` e ADR-011. **[Revertido em 2026-07-16 — escopo letras-only, ADR-012]**
* [x] Validação ao vivo do modelo único (U5, 2026-07-16): `infer_unified` funcionando na webcam sem troca de modo — letras, letras dinâmicas e palavras num só fluxo. **[Revertido em 2026-07-16 — escopo letras-only, ADR-012]**
* [ ] Documentar em *logs* cada hiperparâmetro testado, funções de perda e motivos de (não) convergência para o relatório final — `models/train_log.json` já persiste HPARAMS + history + report a cada treino.

### Participante 2 — Luiz (Foco: Engenharia de Dados e Pipeline Visual)
*Carga pesada inicial. Libera o fluxo para o resto do time.*
* [x] Criar o script `capture.py` utilizando OpenCV para abrir a webcam.
* [x] Integrar o MediaPipe no script para extrair os 21 pontos da mão em tempo real.
* [x] Desenvolver a lógica matemática de normalização espacial (ex: centralizar as coordenadas usando o pulso como origem).
* [x] Implementar a lógica de gravação em janelas fixas (30 *frames*) e exportação em `.npy`.
* [ ] Liderar a gravação do *dataset* com o grupo (Coletar N amostras por sinal) — **opcional** após ADR-005: uso de datasets externos (MINDS-Libras + Brazilian Alphabet) via `src/preprocess_external.py`.

### Participante 3 (Foco: Documentação Científica e Audiovisual)
*Carga pesada na reta final (13/07 a 17/07).*
* [x] Estruturar o repositório do projeto (README, pastas de `src`, `data`, `notebooks`) — feito em 2026-07-09.
* [x] Construir o esqueleto do Jupyter Notebook no Colab/GitHub (Introdução e Metodologia) — `notebooks/MLP_base.ipynb`.
* [ ] Redigir a justificativa técnica detalhando o uso do MediaPipe como substituto otimizado para a segmentação tradicional.
* [ ] Gerar e formatar as matrizes de confusão e gráficos de *loss/accuracy* a partir dos dados do Participante 1.
* [ ] Roteirizar, gravar e editar o vídeo de apresentação de 10 minutos demonstrando a arquitetura e os resultados[cite: 13].

---

## 6. Regras de Versionamento
* O repositório do GitHub é a única fonte da verdade.
* Arquivos pesados (`.npy`, `.csv`) e ambientes virtuais (`.venv`) estão estritamente proibidos de subir para o repositório (`.gitignore` já configurado).
* O Google Colab será usado exclusivamente para rodar o Jupyter Notebook final e processar o treinamento; funções auxiliares devem ficar modularizadas na pasta `src/`.