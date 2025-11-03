# Makefile

# Check if .env exists
ifeq (,$(wildcard .env))
$(error .env file is missing at .env. Please create one based on .env.example)
endif

# Load environment variables from .env
include .env

.PHONY: mypy clean help ruff-check ruff-check-fix ruff-format ruff-format-fix all-check all-fix

################################################################################
## Prek Commands
################################################################################

prek-run: ## Run prek hooks
	@echo "Running prek hooks..."
	prek run --all-files
	@echo "Prek checks complete."

################################################################################
## Linting
################################################################################

# Linting (just checks)
ruff-check: ## Check code lint violations (--diff to show possible changes)
	@echo "Checking Ruff formatting..."
	uv run ruff check .
	@echo "Ruff lint checks complete."

ruff-check-fix: ## Auto-format code using Ruff
	@echo "Formatting code with Ruff..."
	uv run ruff check . --fix --exit-non-zero-on-fix
	@echo "Formatting complete."

################################################################################
## Formatting
################################################################################

# Formatting (just checks)
ruff-format: ## Check code format violations (--diff to show possible changes)
	@echo "Checking Ruff formatting..."
	uv run ruff format . --check
	@echo "Ruff format checks complete."

ruff-format-fix: ## Auto-format code using Ruff
	@echo "Formatting code with Ruff..."
	uv run ruff format .
	@echo "Formatting complete."

#################################################################################
## Static Type Checking
#################################################################################

mypy: ## Run MyPy static type checker
	@echo "Running MyPy static type checker..."
	uv run mypy
	@echo "MyPy static type checker complete."

################################################################################
## Cleanup
################################################################################

clean: ## Clean up cached generated files
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete."

################################################################################
## Composite Commands
################################################################################

all-check: ruff-format ruff-check clean ## Run all: linting, formatting and type checking

all-fix: ruff-format-fix ruff-check-fix mypy clean ## Run all fix: auto-formatting and linting fixes

################################################################################
## Help
################################################################################

help: ## Display this help message
	@echo "Default target: $(.DEFAULT_GOAL)"
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
