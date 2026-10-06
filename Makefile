.PHONY: install test api demo lint run run-vector index search

# make run-vector <imagem>           -> pipeline de vetor: busca multimodal -> OCR + grafo nos candidatos
#                                       -> todas as saídas da melhor referência (como o `make run`)
# make run-vector <imagem> K=8 TOP=2 WEIGHTS=0.2,0.6,0.2 INDEX=data/samples ENGINE=rapidocr
#   K = candidatos da busca | TOP = quantas referências geram saídas | INDEX = (re)constrói o índice

# make run <imagem>                  -> compara com a imagem base (image.png) e gera todas as saídas
# make run <referencia> <suspeita>   -> comparação explícita A x B (também gera todas as saídas)
# make run <imagem> BASE=outra.png ENGINE=tesseract|rapidocr|llm
#
# make index <pasta ou imagens...>   -> indexa (vetor multimodal visual + texto + layout)
# make search <imagem>               -> busca reversa por cosseno nos documentos indexados
# make search <imagem> COMPARE=1     -> + cascata de adulteração (OCR + grafo) em cada candidato
# make search <imagem> K=10 WEIGHTS=0.2,0.6,0.2   (visual,texto,layout)
GOALS = run run-vector index search
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
	uv run python scripts/run_vector.py $(FILES) $(if $(K),-k $(K)) $(if $(TOP),--top $(TOP)) $(if $(WEIGHTS),--weights $(WEIGHTS)) $(if $(INDEX),--index $(INDEX)) $(if $(ENGINE),--engine $(ENGINE))

index:
	uv run python scripts/index.py $(FILES) $(if $(ENGINE),--engine $(ENGINE))

search:
	uv run python scripts/search.py $(FILES) $(if $(K),-k $(K)) $(if $(WEIGHTS),--weights $(WEIGHTS)) $(if $(COMPARE),--compare) $(if $(ENGINE),--engine $(ENGINE))


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
