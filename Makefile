PYTHON ?= python3
VENV := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install run test

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	@test -n "$(INPUT_DIR)" || (echo "INPUT_DIR is required" && exit 1)
	@test -n "$(OUTPUT_DIR)" || (echo "OUTPUT_DIR is required" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 1)
	$(PYTHON_BIN) -m clip_image_similarity.cli \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model $(MODEL) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(OVERWRITE),--overwrite,)

test: install
	$(PIP) install pytest pytest-cov
	$(PYTHON_BIN) -m pytest --cov=clip_image_similarity --cov=metrics --cov-report=term-missing tests
