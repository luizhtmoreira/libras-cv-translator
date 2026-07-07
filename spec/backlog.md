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
* **Sinais Estáticos (1 Frame):** Alfabeto básico (A, B, C, etc.) de frente para a câmera.
* **Sinais Dinâmicos (Janela de Tempo):** Palavras com movimento (ex: "Obrigado", "Desculpa", "Por favor").
* **Fora do Escopo:** Sinais direcionais que dependem do ângulo Z da palma (ex: "Nome"), para evitar explosão da dimensionalidade do dataset.

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
* [ ] Configurar a arquitetura base da rede neural (PyTorch/TensorFlow) recebendo tensores `(Amostras, Frames, Coordenadas)`.
* [ ] Gerar dados sintéticos (*mocks*) com NumPy para testar o fluxo da rede antes da gravação real.
* [ ] Treinar o *Baseline* (Modelo Estático) para sinais de 1 frame.
* [ ] Desenvolver e treinar o Modelo Temporal (Transformer) para analisar janelas de 30 frames.
* [ ] Documentar em *logs* cada hiperparâmetro testado, funções de perda e motivos de (não) convergência para o relatório final.

### Participante 2 — Luiz (Foco: Engenharia de Dados e Pipeline Visual)
*Carga pesada inicial. Libera o fluxo para o resto do time.*
* [x] Criar o script `capture.py` utilizando OpenCV para abrir a webcam.
* [x] Integrar o MediaPipe no script para extrair os 21 pontos da mão em tempo real.
* [x] Desenvolver a lógica matemática de normalização espacial (ex: centralizar as coordenadas usando o pulso como origem).
* [x] Implementar a lógica de gravação em janelas fixas (30 *frames*) e exportação em `.npy`.
* [ ] Liderar a gravação do *dataset* com o grupo (Coletar N amostras por sinal).

### Participante 3 (Foco: Documentação Científica e Audiovisual)
*Carga pesada na reta final (13/07 a 17/07).*
* [ ] Estruturar o repositório do projeto (README, pastas de `src`, `data`, `notebooks`).
* [ ] Construir o esqueleto do Jupyter Notebook no Colab/GitHub (Introdução e Metodologia).
* [ ] Redigir a justificativa técnica detalhando o uso do MediaPipe como substituto otimizado para a segmentação tradicional.
* [ ] Gerar e formatar as matrizes de confusão e gráficos de *loss/accuracy* a partir dos dados do Participante 1.
* [ ] Roteirizar, gravar e editar o vídeo de apresentação de 10 minutos demonstrando a arquitetura e os resultados[cite: 13].

---

## 6. Regras de Versionamento
* O repositório do GitHub é a única fonte da verdade.
* Arquivos pesados (`.npy`, `.csv`) e ambientes virtuais (`.venv`) estão estritamente proibidos de subir para o repositório (`.gitignore` já configurado).
* O Google Colab será usado exclusivamente para rodar o Jupyter Notebook final e processar o treinamento; funções auxiliares devem ficar modularizadas na pasta `src/`.