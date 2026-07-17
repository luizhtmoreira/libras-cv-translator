# ADR-006 — Augmentation de landmarks no loader (não em disco)

**Data:** 2026-07-13
**Status:** Aceita
**Contexto:** Diogo

---

## Contexto

Após treinar o baseline `MLPStatic` no alfabeto processado (Bianka), a validação
interna atingiu F1 macro = 0.9830 — mas a inferência ao vivo na webcam mostrou
confusões sistemáticas em pares com dedos próximos (N↔M, D↔U↔R, S↔I, V↔U↔D).

O trade-off do ADR-005 já reconhecia esse **domain shift entre dataset de estúdio
e webcam do usuário**. A validação alta com falha ao vivo confirmou que o modelo
memorizou o domínio Bianka e não generaliza para novas iluminações, ângulos e
distâncias.

## Decisão

Adicionar augmentation geométrica **online no loader**, aplicada apenas ao split
de treino, aos 21 landmarks (21, 3) já normalizados pela ADR-002:

1. **Rotação 3D** (Euler XYZ) ±15° em torno do pulso (origem após normalização)
2. **Escala isotrópica** ±10%
3. **Translação** ±0.05 (unidades de palma)
4. **Ruído gaussiano** σ=0.01 por coordenada

Os deltas do pulso `[63:66]` **não** sofrem augmentation — eles têm semântica
temporal (movimento bruto, ADR-003) e adicionar ruído aí distorceria o sinal
que distingue estático de dinâmico.

Configurável no bloco `HPARAMS` de `src/train.py`. Todas as faixas são liga/desliga
por override na CLI ou edição direta.

## Justificativa

- **Ataca o domain shift diretamente:** as perturbações simulam variações que
  aparecem naturalmente entre uma foto de estúdio e um frame de webcam
  (mão inclinada, mais perto/longe da câmera, ruído do detector).
- **Zero custo em disco:** os `.npy` continuam `(30, 66)`. Nenhum reprocessamento
  do dataset é necessário. Compatível com ADR-003 e ADR-004.
- **Preserva o significado geométrico:** rotação/escala em torno do pulso não
  quebra a normalização da ADR-002 (o pulso continua na origem).
- **Só no treino:** validação e inferência ao vivo recebem os landmarks intactos,
  garantindo que a métrica reflita o desempenho real.
- **Efeito medido:** F1 macro no split de validação passou de 0.9830 → 0.9863
  (ganho pequeno, esperado — o dataset já estava saturado). O ganho relevante é
  na webcam, verificado empiricamente.

## Alternativas Descartadas

| Alternativa | Motivo da rejeição |
|---|---|
| Augmentation em pixels antes do MediaPipe | Custo alto (rodar MP imagem por imagem em cada epoch); os landmarks já sintetizam o resultado |
| Gerar `.npy` aumentados em disco | Explode uso de espaço; menos diverso (mesma perturbação toda epoch) |
| Perturbar também os deltas | Quebra a semântica de movimento (ADR-003) — deltas são sinal, não ruído |
| Faixas maiores (±30°, ±30% escala) | Modelo aprende invariâncias que não existem na inferência real; degrada validação |

## Referências

- Feature engineering pipeline: `src/features.py::augment_landmarks`
- Hiperparâmetros: `src/train.py` (bloco `HPARAMS["augmentation"]`)
