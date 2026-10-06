.PHONY: install test api demo lint run run-vector index

# make run-vector <imagem>   -> pipeline VETOR + COSSENO: score de 0 a 1 entre a imagem e cada documento indexado
#                               (sem OCR+grafo, sem pareamento por campo, sem veredito de adulteração)
# make run-vector <imagem> K=8 WEIGHTS=0.2,0.6,0.2 INDEX=data/samples ENGINE=rapidocr
#   K = quantos pares | WEIGHTS = visual,texto,layout | INDEX = (re)constrói o índice antes de buscar
# make index <pasta ou imagens...>   -> só indexa (vetor multimodal visual + texto + layout)
#
# make run <imagem>                  -> cascata OCR + grafo contra a imagem base (image.png)
# make run <referencia> <suspeita>   -> cascata OCR + grafo A x B
# make run <imagem> BASE=outra.png ENGINE=tesseract|rapidocr|llm
GOALS = run run-vector index
FILES = $(filter-out $(GOALS),$(MAKECMDGOALS))
ifneq ($(filter $(GOALS),$(MAKECMDGOALS)),)
# os nomes dos arquivos viram "alvos"; este alvo curinga os ignora
.PHONY: $(FILES)
%:
	@:
endif

run:
	uv run python scripts/run.py $(FILES) $(if $(BASE),--base $(BASE)) $(if $(ENGINE),--engine $(ENGINE))

run-vector:
	uv run python scripts/run_vector.py $(FILES) $(if $(K),-k $(K)) $(if $(WEIGHTS),--weights $(WEIGHTS)) $(if $(INDEX),--index $(INDEX)) $(if $(ENGINE),--engine $(ENGINE))

index:
	uv run python scripts/index.py $(FILES) $(if $(ENGINE),--engine $(ENGINE))


install:
	uv sync

test:
	uv run pytest -q

api:
	uv run uvicorn app.main:app --reload --port 8000

demo:
	uv run python scripts/run_samples.py

lint:
	uv run pylint app
