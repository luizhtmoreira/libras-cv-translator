# Roteiro do vídeo de entrega (10 minutos, 2 integrantes)

Formato sugerido: gravação de tela (notebook + demo ao vivo) com narração, câmera do
apresentador opcional. Integrante A faz a demo na própria webcam (recomendado: quem tem
capturas próprias no treino, pois o modelo está calibrado nesse domínio). Integrante B
apresenta métodos e resultados sobre o notebook.

## Linha do tempo

### Bloco 1 — Abertura e problema (0:00–1:00) — Integrante A
- Apresentar a dupla e o tema: tradutor de Libras em tempo real, restrito ao alfabeto manual.
- Motivação em 2 frases: acessibilidade; soletração cobre nomes e palavras fora de vocabulário.
- A restrição do edital: proibido classificar imagens diretamente (CNN sobre pixels).
  Deixar claro que isso definiu a estratégia do projeto.

### Bloco 2 — Estratégia: landmarks (1:00–2:30) — Integrante A
- Tela: célula do pipeline no notebook (seção 3.1).
- Explicar o isolamento geométrico: MediaPipe extrai 21 pontos 3D da mão; o classificador
  vê só geometria (63 valores), não pixels.
- Os 3 ganhos: modelos pequenos (CPU, tempo real), invariância a fundo/iluminação,
  percepção delegada a um modelo pré-treinado.
- Mostrar rapidamente a normalização (pulso na origem, escala pela palma) e as 20
  features geométricas invariantes a rotação (86 no total).

### Bloco 3 — Dados e augmentation (2:30–4:00) — Integrante B
- Tela: tabela de datasets + célula de contagem (seção 3.2).
- As 4 fontes (dataset externo de imagens + capturas próprias + capturas da equipe,
  estáticas e dinâmicas) convergindo para o mesmo formato (30, 66).
- Domain shift estúdio→webcam: o aprendizado do projeto — só dados do domínio alvo
  resolvem; augmentation ajuda mas não substitui.
- Augmentation em landmarks (rotação/escala/translação/ruído) com coerência temporal:
  uma transformação por sequência, senão a trajetória (que é o sinal) é destruída.

### Bloco 4 — Arquiteturas e treino (4:00–5:30) — Integrante B
- Tela: seções 3.7 e 3.8 do notebook.
- A hipótese do estudo de ablação: letras estáticas = configuração (1 frame basta);
  letras dinâmicas h/j/k/x/y/z = trajetória (exige janela temporal).
- MLP baseline: por que MLP (entrada já é feature de alto nível, nada a convoluir).
- Transformer: janela (30, 86), atenção vs recorrência, positional encoding, pre-LN.
- Treino em 30 segundos: Adam, class weights, F1 macro como métrica, early stopping,
  melhor checkpoint. Citar a ablação de hiperparâmetros (reduzir capacidade piorou).

### Bloco 5 — DEMO AO VIVO (5:30–8:00) — Integrante A
O coração do vídeo. Ensaiar antes; gravar a demo separada como backup.
- Abrir a aplicação de inferência ao vivo (janela da webcam com esqueleto da mão).
- Modo LETRA: soletrar um nome curto (ex: "DIOGO" ou "UNB"), segurando cada letra
  ~3 s até estabilizar (verde). Narrar os guardrails: suavização + consenso de 7
  frames + confiança mínima — é isso que elimina o flickering.
- Apertar espaço → modo LETRA DINÂMICA: fazer h, j, k, x, y, z (as que estiverem
  mais confiáveis; j, y e z costumam ser as mais visuais). Narrar: buffer de 30
  frames sempre quente, ~6 predições/s, aceita com 2 predições concordantes.
- Bônus rápido: mostrar o y nos DOIS modos (variante estática e dinâmica) e explicar
  que a tecla decide qual modelo responde.

### Bloco 6 — Resultados (8:00–9:00) — Integrante B
- Tela: métricas e matrizes de confusão do notebook (seção 5).
- MLP: F1 macro ≈ 0.99 em 21 classes; erros raros em pares vizinhos (m/n, u/v).
- Transformer: F1 = 1.00 nas 6 dinâmicas.
- A leitura da ablação em 2 frases: o MLP é estruturalmente cego ao movimento (o frame
  central de um "j" é um "i"); o Transformer resolve exatamente essa lacuna. E a
  especialização venceu o modelo unificado testado antes (0.9947 vs 0.9839).

### Bloco 7 — Limitações e conclusão (9:00–10:00) — Integrante A
- Limitações honestas: uma mão só; domain shift residual; palavras fora do escopo final.
- Trabalhos futuros em 3 itens: palavras com dados próprios; roteador automático de
  modo; duas mãos.
- Fechamento: a restrição do edital virou vantagem — sistema leve, tempo real, sem GPU.

## Dicas de gravação
1. **Ensaiar a demo 2–3 vezes** e escolher letras que estabilizam rápido na sua
   iluminação. Gravar uma tomada da demo antes, como plano B de edição.
2. Iluminação frontal e fundo limpo na demo (o detector agradece); mão dentro do quadro.
3. Gravar tela em 1080p; zoom no notebook (Ctrl+ +) para o texto ficar legível.
4. Cada bloco pode ser gravado em separado e montado na edição — mais fácil que uma
   tomada única de 10 min.
5. Margem: mirar em 9:30 de conteúdo; estourar 10 min é pior que sobrar 30 s.

## Divisão resumida
| Integrante | Blocos | Tempo total |
|---|---|---|
| A (demo) | 1, 2, 5, 7 | ~5:30 |
| B (métodos/resultados) | 3, 4, 6 | ~4:00 |
