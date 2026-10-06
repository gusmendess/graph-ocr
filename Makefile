.PHONY: install test api demo lint run

# make run <imagem>                 -> compara com a imagem base (image.png) e gera todas as saídas
# make run <referencia> <suspeita>  -> comparação explícita A x B (também gera todas as saídas)
# make run <imagem> BASE=outra.png ENGINE=tesseract|rapidocr|llm
RUN_FILES = $(filter-out run,$(MAKECMDGOALS))
ifneq ($(filter run,$(MAKECMDGOALS)),)
# os nomes dos arquivos viram "alvos"; este alvo curinga os ignora
.PHONY: $(RUN_FILES)
%:
	@:
endif

run:
	uv run python scripts/run.py $(RUN_FILES) $(if $(BASE),--base $(BASE)) $(if $(ENGINE),--engine $(ENGINE))


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
