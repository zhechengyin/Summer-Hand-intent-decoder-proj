PYTHON ?= python
DATA_ARCHIVE ?= data.tar.gz

.PHONY: help setup-dev install-hooks lint format test firmware-test clean data-archive data-restore

help:
	@echo "setup-dev     Install runtime/development dependencies and Git hooks"
	@echo "install-hooks Install pre-commit and pre-push hooks"
	@echo "lint          Run every pre-commit check over tracked files"
	@echo "format        Apply Ruff fixes and formatting to active Python code"
	@echo "test          Run the repository test suite"
	@echo "firmware-test Compile and compare C inference against frozen Python"
	@echo "clean         Remove local build and tool-cache files only"
	@echo "data-archive  Pack data/processed into $(DATA_ARCHIVE)"
	@echo "data-restore  Restore data/processed from $(DATA_ARCHIVE)"

setup-dev:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type pre-push

install-hooks:
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type pre-push

lint:
	$(PYTHON) -m pre_commit run --all-files

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

firmware-test:
	$(PYTHON) models/finger_movements/cssd_lda/firmware/tools/validate_firmware.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

# Archive only reproducible processed arrays. Raw source data is intentionally
# excluded because it should be downloaded from the authoritative source.
data-archive:
	test -d data/processed
	tar -czf "$(DATA_ARCHIVE)" data/processed
	@echo "Wrote $(DATA_ARCHIVE)"

data-restore:
	test -f "$(DATA_ARCHIVE)"
	tar -xzf "$(DATA_ARCHIVE)"
	@echo "Restored data/processed from $(DATA_ARCHIVE)"
