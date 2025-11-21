PYTHON ?= python3
VENV := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install run

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
		$(if $(OVERWRITE),--overwrite,)
