PYTHON ?= python3
VENV := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install install-current-env run run-current-env anonymize-labels anonymize-labels-current-env test test-current-env

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

install-current-env:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run: install
	@test -n "$(INPUT_DIR)" || (echo "INPUT_DIR is required" && exit 1)
	@test -n "$(OUTPUT_DIR)" || (echo "OUTPUT_DIR is required" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 1)
	$(PYTHON_BIN) -m clip_image_similarity.cli \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model $(MODEL) \
		$(if $(PRETRAINED),--pretrained $(PRETRAINED),) \
		$(if $(CHECKPOINT_PATH),--checkpoint_path $(CHECKPOINT_PATH),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(ANONYMIZE_LABELS),--anonymize-labels $(ANONYMIZE_LABELS),) \
		$(if $(PAIRWISE_DTYPE),--pairwise-dtype $(PAIRWISE_DTYPE),) \
		$(if $(OVERWRITE),--overwrite,)

run-current-env: install-current-env
	@test -n "$(INPUT_DIR)" || (echo "INPUT_DIR is required" && exit 1)
	@test -n "$(OUTPUT_DIR)" || (echo "OUTPUT_DIR is required" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 1)
	$(PYTHON) -m clip_image_similarity.cli \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--model $(MODEL) \
		$(if $(PRETRAINED),--pretrained $(PRETRAINED),) \
		$(if $(CHECKPOINT_PATH),--checkpoint_path $(CHECKPOINT_PATH),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(ANONYMIZE_LABELS),--anonymize-labels $(ANONYMIZE_LABELS),) \
		$(if $(PAIRWISE_DTYPE),--pairwise-dtype $(PAIRWISE_DTYPE),) \
		$(if $(OVERWRITE),--overwrite,)

anonymize-labels: install
	@test -n "$(OUTPUT_DIR)" || (echo "OUTPUT_DIR is required" && exit 1)
	@test -n "$(LABELS)" || (echo "LABELS is required" && exit 1)
	$(PYTHON_BIN) -m clip_image_similarity.generate_anonymous_labels \
		--output-dir $(OUTPUT_DIR) \
		--labels $(LABELS) \
		$(if $(OVERWRITE),--overwrite,)

anonymize-labels-current-env: install-current-env
	@test -n "$(OUTPUT_DIR)" || (echo "OUTPUT_DIR is required" && exit 1)
	@test -n "$(LABELS)" || (echo "LABELS is required" && exit 1)
	$(PYTHON) -m clip_image_similarity.generate_anonymous_labels \
		--output-dir $(OUTPUT_DIR) \
		--labels $(LABELS) \
		$(if $(OVERWRITE),--overwrite,)

test: install
	$(PIP) install pytest pytest-cov
	$(PYTHON_BIN) -m pytest --cov=clip_image_similarity --cov=metrics --cov-report=term-missing tests

test-current-env: install-current-env
	$(PYTHON) -m pip install pytest pytest-cov
	$(PYTHON) -m pytest --cov=clip_image_similarity --cov=metrics --cov-report=term-missing tests
