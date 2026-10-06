# graph-ocr

Detecção de comprovantes Pix adulterados com **OCR (texto + bbox + confiança) + grafo**, organizada como cascata.

## Cascata

| # | Camada | Decide quando | Custo |
|---|--------|---------------|-------|
| 1 | `sha256` | bytes idênticos → score 1.0 | ~0 ms |
| 2 | `phash` | similaridade < 0.7 → documentos diferentes (sem OCR) | ~10 ms |
| 3 | OCR + grafo | só os pares ambíguos; acha palavras alteradas | segundos |

Camada 3:

1. **OCR** (`app/api/adapters/ocr`): Tesseract, RapidOCR (PP-OCR/ONNX) ou LLM (Claude vision) atrás da mesma interface `read(image) -> [Word(text, bbox, conf)]`. Resultado em cache por `sha256 + motor`.
2. **Grafo k-NN de palavras**: nó = palavra, aresta = vizinhança espacial. O contexto desambigua palavras repetidas (`R$`, `do`...).
3. **Alinhamento de grafos** (`alignment_service`): cada nó recebe uma assinatura geométrica (distâncias aos vizinhos / altura de palavra, invariante a escala e rotação). Pares candidatos A↔B saem dela, sem exigir texto igual. RANSAC propõe similaridades (ou afim, `alignment_model=affine`) e escolhe a que mais sobrepõe os dois grafos; o texto igual é só um bônus (`alignment_text_weight`, 0 = puramente geométrico). Refinamento por mínimos quadrados. Palavras editadas viram outliers e não distorcem o ajuste.
4. **Matching por vetor de texto + cosseno** (`matcher=vector`, padrão; `vector_matching_service`): cada linha do OCR vira um nó com vetor de n-gramas de caracteres (sensível a dígitos, ao contrário de embeddings de frase) e o grafo k-NN das linhas dá o contexto (`context_weight`). O pareamento é a atribuição ótima sobre o cosseno, sem igualdade exata de texto e sem geometria absoluta, então é indiferente a deslocamento, escala, rotação e perspectiva. Cosseno ≥ `cos_same` → `matched`; entre `cos_edit` e `cos_same` ("quase igual") → `modified`. **Campo trocado por inteiro** (cosseno ~0, ex.: código da transação): linhas de cosseno ~1 viram âncoras, e uma linha livre ocupa o mesmo campo se os vizinhos-âncora coincidem (`slot_overlap`) → `modified`. Vizinhos também editados são ignorados, então edições adjacentes não se contaminam. `matcher=iou` mantém o método antigo (texto exato + IoU).
5. **Forense**: Δ altura, Δ largura/char, Δ baseline, Δ confiança do par (medidos após o alinhamento) e anomalia da palavra frente aos vizinhos de linha (não exige template).
6. **Grafo do caso** (documento ↔ chave Pix/CNPJ/ID de transação) para achar identificadores compartilhados e **regra de consistência**: a data exibida precisa bater com o timestamp (UTC) embutido no ID Pix `E...`.

## Uso

```bash
make run image6.jpeg                      # compara com a imagem base (image.png) e gera TODAS as saídas
make run original.png suspeita.png        # comparação explícita A x B (também gera todas as saídas)
make run image6.jpeg BASE=outra.png       # troca a imagem base (padrão: GRAPH_OCR_BASE_IMAGE)
make run image6.jpeg ENGINE=tesseract     # troca o motor de OCR (rapidocr | tesseract | llm)
```

Saídas em `outputs/` (para `image6.jpeg` contra `image.png`):
`image__vs__image6.png` (comparação), `..__zoom.png` (campos alterados), `..__graphs.png` (grafos sobrepostos) e `image6__analysis.png` (análise isolada da imagem suspeita) e, para comparar a leitura do OCR, `image__ocr.txt` e `image6__ocr.txt` (texto por linha + tabela de palavras com confiança e bbox).

A análise isolada usa só regras objetivas (ex.: data exibida x timestamp do ID Pix): `suspicious` ou `no_evidence`, que **não prova autenticidade**. A anomalia geométrica é impressa como sinal fraco e não entra no veredito (é ruidosa).

No modo completo (`make run`) o OCR prevalece sobre o phash: o phash baixo só vira "documento diferente" se o conteúdo também diferir (`GRAPH_OCR_OCR_DIFFERENT_THRESHOLD`). No caminho rápido (sem `--full`), o phash continua decidindo sozinho.

```bash
uv sync
uv run python scripts/compare.py data/samples/original.png data/samples/fraud_day.png --full
make demo        # original vs cada fraude -> outputs/
make api         # POST /compare, POST /compare/visual
make test
```

Saídas em `outputs/`: `<A>__vs__<B>.png` (originais, OCR com bboxes por status, mapas de diferença) e `..__zoom.png` (campos alterados ampliados).

## Motor de OCR

`GRAPH_OCR_OCR_ENGINE=auto|tesseract|rapidocr|llm`. `auto` usa Tesseract se o binário estiver instalado (`winget install UB-Mannheim.TesseractOCR`), senão RapidOCR. `llm` exige `ANTHROPIC_API_KEY`; bboxes de LLM são aproximadas e a confiança é auto-reportada.

## Limites

- O `phash` não é invariante a rotação/escala: um print do mesmo comprovante bem rotacionado pode cair abaixo do limiar (0.7) e ser rejeitado como "documento diferente" antes do OCR. Ajuste `GRAPH_OCR_PHASH_THRESHOLD` ou use `--full`.
- O alinhamento cobre similaridade e afim (translação, rotação, escala, cisalhamento); perspectiva (foto torta de tela/papel) não: o erro de registro varia ao longo da página e várias palavras viram `moved`, com `added`/`removed` espúrios. Falta um modelo de homografia.
- OCR tokeniza diferente entre imagens (junta `06/06/2021 às`, divide `08/06/2021 - 20:45`, corta palavras na borda): isso gera `modified`/`removed` espúrios. O matching ainda é 1↔1 entre palavras.
- Print comprimido (WhatsApp) degrada bbox e confiança; use o par com cuidado.
- Edição limpa (mesma fonte/tamanho) não aparece na geometria; quem pega é a diferença de texto e as regras de consistência.
- `suspicion` e `geometry_anomaly` são heurísticas, não probabilidades calibradas.
