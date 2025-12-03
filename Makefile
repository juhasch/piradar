VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MKDOCS := $(VENV)/bin/mkdocs

.PHONY: help docs-install gen-regmap docs-serve docs-build docs-clean

help:
	@echo "Available targets:"
	@echo "  docs-install   - Setup .venv pip and install doc tools"
	@echo "  gen-regmap     - Generate docs/registermap.md from YAML"
	@echo "  docs-serve     - Serve MkDocs locally (with auto-regenerate)"
	@echo "  docs-build     - Build static site into 'site/'"
	@echo "  docs-clean     - Remove 'site/'"

docs-install:
	@echo "Ensuring pip in $(VENV) ..."
	@$(PYTHON) -m ensurepip --upgrade >/dev/null 2>&1 || true
	@$(PYTHON) -m pip install -q --upgrade pip
	@echo "Installing doc dependencies ..."
	@$(PIP) install -q -r docs/requirements-docs.txt

gen-regmap: docs-install
    @echo "Generating docs/registermap.md ..."
    @$(PYTHON) scripts/generate_registermap_md.py --input piradar/bgt60tr13c_registermap.yaml --output docs/registermap.md

docs-serve: gen-regmap
	@$(MKDOCS) serve

docs-build: gen-regmap
	@$(MKDOCS) build --clean

docs-clean:
	rm -rf site
	@echo "Cleaned site/"

