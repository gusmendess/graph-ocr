# graph-ocr

Busca reversa de comprovantes em cascata, da etapa mais barata à mais cara: **sha256 → phash → vetor multimodal (cosseno)**. Para cada par (consulta, documento indexado) devolve um **score de 0 a 1** de quão similares são.

## Cascata

Cada par para no primeiro estágio que decide:

| # | Estágio | Decide quando | Score do par | Custo |
|---|---------|---------------|--------------|-------|
| 1 | `sha256` | bytes idênticos | 1.0 | ~0 ms |
| 2 | `phash` | similaridade perceptual < `GRAPH_OCR_PHASH_THRESHOLD` (0.3) | a própria similaridade do phash | ~10 ms |
| 3 | vetor | só os pares que sobraram | cosseno composto do vetor multimodal | OCR + rede neural |

O vetor da consulta (OCR + DINOv2) é calculado uma única vez, e só se algum par chegar ao estágio 3. Se todos os pares forem decididos por sha256/phash, nem o OCR roda.

O resultado ordena primeiro quem passou pelos portões (sha256, depois vetor) e por último quem o phash barrou, porque os scores têm significados diferentes: o cosseno mede semelhança de conteúdo, o phash só diz que o par é visualmente distante.

## O vetor (estágio 3)

Três blocos, cada um L2-normalizado:

| Bloco | O que captura | Como | Usa OCR? |
|---|---|---|---|
| `visual` | aparência (banco, cores, logo, estrutura) | DINOv2-small em ONNX (384-d), imagem inteira com margem branca | não |
| `text` | conteúdo, incluindo dígitos | n-gramas de caracteres sobre todo o texto do OCR (2048-d, hash) | sim |
| `layout` | onde o texto fica na página | grade 16×16 sobre o bbox do conteúdo, fração coberta por texto (256-d); invariante a escala e translação | sim |

**Score do par** = média ponderada dos cossenos de cada bloco (`GRAPH_OCR_WEIGHT_*`), cada cosseno truncado em 0: 0 = nada parecido, 1 = idêntico. Os blocos ficam separados no índice, então dá para reponderar ou buscar só por texto/visual sem reindexar. Um bloco ausente (ex.: sem rede para baixar o modelo) é ignorado e os pesos são renormalizados.

O OCR (`app/api/adapters/ocr`) é Tesseract, RapidOCR (PP-OCR/ONNX) ou LLM (Claude vision) atrás da mesma interface `read(image) -> [Word(text, bbox, conf)]`, com cache por `sha256 + motor`. `GRAPH_OCR_OCR_ENGINE=auto|tesseract|rapidocr|llm`: `auto` usa Tesseract se o binário estiver instalado, senão RapidOCR. `llm` exige `ANTHROPIC_API_KEY`.

## Uso

```bash
make index                                       # indexa data/samples/documents (ou: make index <pasta ou imagens>)
make run-vector <imagem>                         # cascata contra cada documento indexado
make run-vector <imagem> K=8 WEIGHTS=0,1,0       # 8 pares; só texto (visual,texto,layout)
make run-vector <imagem> INDEX=<pasta>           # (re)constrói o índice a partir de outra pasta antes de buscar
make run-vector <imagem> PLOTS=0                 # sem as figuras
make api                                         # POST /index e POST /search
```

Nenhum nome de arquivo está fixo no código: as pastas padrão vêm de `GRAPH_OCR_INDEX_SOURCE` e `GRAPH_OCR_IMAGES_SOURCE`.

Para que **todos** os pares passem pelo vetor (sem o phash barrar): `GRAPH_OCR_PHASH_THRESHOLD=0`.

Se o índice estiver vazio, `make run-vector` indexa `GRAPH_OCR_INDEX_SOURCE` (padrão `data/samples/documents`) sozinho. O índice fica em `.cache/index` e guarda o rótulo (nome do arquivo) da época da indexação; ao renomear arquivos, apague a pasta ou use `INDEX=`.

### Saídas (`outputs/`)

| Arquivo | Quando sai |
|---|---|
| `<imagem>__similarity.json` | sempre: score, estágio que decidiu e cosseno por bloco de cada par |
| `<imagem>__ocr.txt` | quando o OCR rodou: texto por linha e tabela de palavras com confiança e bbox |
| `<imagem>_vs_<candidato>_sha256.png` | sempre, exceto com `PLOTS=0`: uma por par da tabela. `[original] A`, `[original] B` e `\|A − B\|` dos originais (fica todo preto se os bytes são idênticos); o título traz quantos dos 64 caracteres do hash coincidem. Se A e B têm tamanhos diferentes, B é redimensionada para A só para essa diferença |
| `<imagem>_vs_<candidato>_phash.png` | idem: `[phash] A` e `[phash] B` pré-processados (a miniatura 32×32 em tons de cinza que o hash enxerga) e `\|A − B\|` delas; o título traz a similaridade, os bits que diferem e o resultado contra o limiar |
| `<imagem>__vectors_*.png` | quando o estágio de vetor rodou, exceto com `PLOTS=0` |

Figuras de vetores (para slides):

| Arquivo | Pergunta que responde |
|---|---|
| `_bars.png` | quanto a consulta se parece com cada documento (ela própria entra no topo, com 1.000), e qual bloco explica o score? |
| `_matrix.png` | como todos os documentos se parecem entre si? (grupos aparecem como blocos escuros) |
| `_map.png` | mapa 2D (PCA do vetor composto): quem fica perto de quem? Pontos quase idênticos dividem um rótulo |
| `_map3d.png` | o mesmo mapa com a 3ª componente do PCA |
| `_zoom.png` | zoom no aglomerado de documentos que se sobrepõem no mapa (2D e 3D): o PCA é recalculado só com eles, então a escala é a das diferenças entre eles e os pontos se separam; cada um leva o score com a consulta. Só sai se há um aglomerado de pelo menos 3 documentos |
| `_combined.png` | as quatro juntas, como visão geral |

Os scores das figuras usam a mesma regra do índice. O mapa PCA é ilustrativo: com poucos documentos, as distâncias são relativas e dependem da projeção.

API: `POST /index` (file) e `POST /search?k=5&w_visual=&w_text=&w_layout=` (file).

## Estrutura

```
app/            API (FastAPI) e a lógica: endpoints, use_cases, services, repositories, adapters (OCR), domain, infrastructure
  tests/        testes unitários (dublês no lugar de OCR e modelos: rápidos, sem rede)
scripts/        a mesma lógica como CLI (run_vector, index, index_images, search_text)
integration_tests/   testes com os modelos reais sobre data/samples
data/samples/
  documents/    comprovantes de exemplo (example1.png, ...): o que `make index` e `make run-vector` indexam
  dresses/      fotos de exemplo (example1.png, ...): o que `make index-images` indexa
docker/         Dockerfile (o docker-compose.yml fica na raiz)
.cache/         modelos baixados, caches de OCR e índices (não versionado)
outputs/        figuras e JSONs gerados (não versionado)
```

Para usar as suas imagens, ponha-as nessas pastas (qualquer nome) e rode `make index` / `make index-images`. Os índices guardam o nome do arquivo da época da indexação: ao renomear ou trocar arquivos, apague `.cache/index` e `.cache/semantic_index`.

## Testes

```bash
make test               # 52 testes unitários, menos de 1 s, sem modelos nem rede
make test-integration   # com RapidOCR, DINOv2 e SigLIP2 reais sobre data/samples; pula se os modelos não estiverem em .cache/models
```

Os unitários usam um OCR de mentira (devolve palavras a partir de uma marca no pixel) e, para a busca de imagens, um embedder de cor no lugar do SigLIP. Cobrem a cascata (qual estágio decide, o OCR não rodar quando o phash barra, contagem por estágio sobre todos os candidatos), o índice, os pesos, a persistência e a recusa de índice de outro modelo. Não medem a qualidade do SigLIP nem do DINOv2: isso fica nos de integração, que só conferem o ranking nas fotos de exemplo.

## Docker

```bash
make docker-build
make docker-up                                           # API em http://localhost:8000/docs
docker compose run --rm cli scripts/index.py             # CLI dentro do container
docker compose run --rm cli scripts/search_text.py "green dress" -k 6
```

`.cache` (modelos, caches e índices), `data` (somente leitura) e `outputs` são montados do host: os modelos (SigLIP2 ≈ 1,5 GB) são baixados uma vez, na primeira execução, e não entram na imagem. O compose fixa `GRAPH_OCR_OCR_ENGINE=rapidocr` para o resultado coincidir com a execução local.

## Limites

- **Com o limiar em 0.3 o phash quase nunca barra nada.** Para imagens sem relação, a similaridade do phash fica em torno de 0.50 (medido com ruído: média 0.509, desvio 0.062; só 2 de 2000 pares abaixo de 0.3). Uma nota fiscal sintética contra os comprovantes Pix teve 0.41 a 0.63, nenhum abaixo de 0.3, e os comprovantes entre si ficaram em 0.56 ou mais. Ou seja, as faixas se sobrepõem e nenhum limiar separa "outro tipo de documento" com segurança usando só o phash. A nota fiscal desse teste é sintética (não há nota fiscal real nas amostras).
- **O phash não é invariante a recorte, rotação nem escala.** Um limiar alto o bastante para barrar documentos distintos (acima de ~0.5) também barraria prints recortados ou fotos de tela do mesmo comprovante (nas amostras, 0.56 a 0.63).
- **O score mede semelhança, não adulteração.** Original e versão editada saem com cosseno altíssimo (nas amostras, 0.994 a 1.000), porque mudam poucos caracteres de um documento inteiro. O vetor não aponta *onde* a imagem difere.
- Os pesos e o limiar do phash não estão calibrados: só há um tipo de comprovante nas amostras, e o bloco visual não foi validado contra documentos de outros emissores.
- O bloco de texto é léxico (n-gramas de caracteres): não entende sinônimos nem "ano", então não serve para busca por linguagem natural.
- O OCR erra e quebra palavras de forma diferente entre imagens (`R$` lido como `RS`, `pagamento` em `pagam` + `ento`); isso move um pouco o vetor de texto.
- O índice é exato em numpy (serve até dezenas de milhares de documentos); a interface `VectorIndex` permite trocar por FAISS/HNSW/pgvector.

## Busca de imagens por linguagem natural

Segundo pipeline, independente do de documentos: acha fotos a partir de uma frase ("green dress"), sem OCR, sem rótulos e sem legendas. Usa o **SigLIP2** (Apache-2.0, ONNX, só onnxruntime), que coloca **texto e imagem no mesmo espaço vetorial**: o cosseno entre o vetor da frase e o da foto é alto quando a foto mostra o que a frase descreve. O DINOv2 do pipeline de documentos não faz isso: ele mede aparência parecida entre duas imagens, mas não entende texto.

```bash
make index-images                                # indexa data/samples/dresses (ou: make index-images <pasta ou fotos>)
make search-text Q="green dress" K=6             # top 6 + folha de contato em outputs/search_green_dress.png
make search-text Q="green dress" PLOTS=0         # sem a folha de contato
```

API: `POST /images/index` (file) e `GET /images/search?q=green dress&k=6`.

Índice próprio em `.cache/semantic_index` (um vetor por imagem, mais uma cópia da foto). O índice registra com qual modelo foi criado e **recusa** misturar vetores de modelos ou pré-processamentos diferentes (apague a pasta e reindexe ao trocar). Os modelos são baixados na primeira execução para `.cache/models`.

Modelo (`GRAPH_OCR_SIGLIP_MODEL`) e pré-processamento (`GRAPH_OCR_SIGLIP_RESIZE`), medidos nas 6 fotos reais com 16 consultas (cor, tom, tecido, modelo, cenário; a verdade de cada foto é o que se vê nela):

| Configuração | Consultas perfeitas | Tempo por imagem | Disco |
|---|---|---|---|
| base, **int8** (quantizado) | 13 a 14 de 16 (mAP 0,92 a 0,95) | 0,1 s | 0,4 GB |
| so400m, **int8** | 6 a 9 de 16 (mAP 0,76 a 0,85) | | 1,2 GB |
| **base, sem quantização, `pad` (padrão)** | **16 de 16** (mAP 1,000) | 0,13 s | 1,5 GB |
| so400m, sem quantização (`GRAPH_OCR_SIGLIP_MODEL=so400m`) | 15 a 16 de 16 em todas as variações | 2,5 s | 4,5 GB |

- **A quantização int8 degrada o ranking** e prejudica mais o modelo grande (o grande em int8 ficou pior que o pequeno). Por isso o padrão é sem quantização.
- `pad` (preserva a proporção da foto, com margem cinza) ficou igual ou melhor que `squash` (esticar) no modelo pequeno; no grande, empatam.
- Recortes múltiplos com máximo e frase-modelo ("a photo of a …") **não ajudaram** e ficaram de fora.
- O modelo grande é mais robusto a variações de pré-processamento, mas 20 vezes mais lento; vale quando a precisão importa mais que o tempo.
- **Limite da medição:** só 6 fotos e 16 consultas, e a configuração foi escolhida nesse mesmo conjunto. O ganho da quantização (mesmo pipeline, só trocando o arquivo) é sólido; o "16 de 16" não é uma medida de precisão em fotos novas.
- O cosseno entre texto e imagem fica na faixa de 0,1 a 0,15 nos acertos por construção do SigLIP (escala e viés aprendidos): vale pela **ordem** do ranking, não como probabilidade.
- A busca é global (compara a frase com a imagem inteira). Não localiza onde está o objeto na foto.
