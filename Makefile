.PHONY: install test test-integration api lint run-vector index index-images search-text docker-build docker-up docker-down

# DOCUMENTOS: cascata sha256 -> phash -> vetor multimodal
# make index [<pasta ou imagens>]      -> indexa (padrão: GRAPH_OCR_INDEX_SOURCE = data/samples/documents)
# make run-vector <imagem> [K=8] [WEIGHTS=0.05,0.9,0.05] [INDEX=<pasta>] [ENGINE=rapidocr] [PLOTS=0]
#   K = quantos pares | WEIGHTS = visual,texto,layout | INDEX = (re)constrói o índice antes de buscar
#   PLOTS=0 não gera as figuras (outputs/)
#   Sem o phash barrar pares: GRAPH_OCR_PHASH_THRESHOLD=0 (todos os pares passam pelo vetor)
#
# IMAGENS POR LINGUAGEM NATURAL: SigLIP2 (texto e imagem no mesmo espaço vetorial)
# make index-images [<pasta ou fotos>] -> indexa (padrão: GRAPH_OCR_IMAGES_SOURCE = data/samples/dresses)
# make search-text Q="green dress" [K=6] [PLOTS=0]   -> as K fotos mais parecidas com a frase + folha de contato
#
# TESTES
# make test               -> unitários: rápidos, sem modelos e sem rede
# make test-integration   -> com os modelos reais (já baixados em .cache/models; pulam sozinhos se faltarem)
#
# DOCKER
# make docker-build | docker-up | docker-down
GOALS = run-vector index index-images
FILES = $(filter-out $(GOALS),$(MAKECMDGOALS))
ifneq ($(filter $(GOALS),$(MAKECMDGOALS)),)
# os nomes dos arquivos viram "alvos"; este alvo curinga os ignora
.PHONY: $(FILES)
%:
	@:
endif

run-vector:
	uv run python scripts/run_vector.py $(FILES) $(if $(K),-k $(K)) $(if $(WEIGHTS),--weights $(WEIGHTS)) $(if $(INDEX),--index $(INDEX)) $(if $(ENGINE),--engine $(ENGINE)) $(if $(PLOTS),$(if $(filter 0,$(PLOTS)),--no-plots))

index:
	uv run python scripts/index.py $(FILES) $(if $(ENGINE),--engine $(ENGINE))

index-images:
	uv run python scripts/index_images.py $(FILES)

search-text:
	uv run python scripts/search_text.py "$(Q)" $(if $(K),-k $(K)) $(if $(PLOTS),$(if $(filter 0,$(PLOTS)),--no-plots))


install:
	uv sync

test:
	uv run pytest -q

test-integration:
	uv run pytest -q -m integration

api:
	uv run uvicorn app.main:app --reload --port 8000

lint:
	uv run pylint app

docker-build:
	docker compose build

docker-up:
	docker compose up -d api

docker-down:
	docker compose down
