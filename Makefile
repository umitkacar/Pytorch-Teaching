.PHONY: help install install-dev test test-cov test-parallel lint format type-check security check clean run docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)PyTorch Teaching - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install the package
	@echo "$(BLUE)Installing package...$(NC)"
	pip install -e .

install-dev: ## Install the package with development dependencies
	@echo "$(BLUE)Installing package with dev dependencies...$(NC)"
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=pytorch_teaching --cov-report=html --cov-report=term --cov-report=xml tests/

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	pytest -n auto tests/

test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests verbosely...$(NC)"
	pytest -vv tests/

lint: ## Run linting
	@echo "$(BLUE)Running linters...$(NC)"
	ruff check src tests

lint-fix: ## Run linting with auto-fix
	@echo "$(BLUE)Running linters with auto-fix...$(NC)"
	ruff check --fix src tests

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black src tests

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check src tests

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src -ll

check: format-check lint type-check security ## Run all checks
	@echo "$(GREEN)✓ All checks passed!$(NC)"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run: ## Run the CLI tool
	@echo "$(BLUE)Running pytorch-teach...$(NC)"
	pytorch-teach --help

run-info: ## Show system info
	@echo "$(BLUE)Showing system info...$(NC)"
	pytorch-teach info

run-list: ## List all lessons
	@echo "$(BLUE)Listing lessons...$(NC)"
	pytorch-teach list

run-lesson-1: ## Run Lesson 1
	@echo "$(BLUE)Running Lesson 1...$(NC)"
	pytorch-teach run 1

run-lesson-21: ## Run Lesson 21 (ExecutorTorch)
	@echo "$(BLUE)Running Lesson 21 (ExecutorTorch)...$(NC)"
	pytorch-teach run 21

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs && mkdocs serve

pre-commit: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

build: clean ## Build package
	@echo "$(BLUE)Building package...$(NC)"
	python -m build

publish-test: build ## Publish to TestPyPI
	@echo "$(YELLOW)Publishing to TestPyPI...$(NC)"
	python -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	@echo "$(RED)Publishing to PyPI...$(NC)"
	python -m twine upload dist/*

hatch-test: ## Run tests with hatch
	@echo "$(BLUE)Running tests with hatch...$(NC)"
	hatch run test

hatch-cov: ## Run tests with coverage using hatch
	@echo "$(BLUE)Running tests with coverage (hatch)...$(NC)"
	hatch run test-cov

hatch-check: ## Run all checks with hatch
	@echo "$(BLUE)Running all checks (hatch)...$(NC)"
	hatch run check

hatch-all: ## Run complete workflow with hatch
	@echo "$(BLUE)Running complete workflow (hatch)...$(NC)"
	hatch run all

doctor: ## Run health check
	@echo "$(BLUE)Running health check...$(NC)"
	pytorch-teach doctor

# CI/CD targets
ci-test: install-dev lint test-cov ## CI test pipeline
	@echo "$(GREEN)✓ CI tests passed!$(NC)"

ci-build: clean build ## CI build pipeline
	@echo "$(GREEN)✓ CI build completed!$(NC)"

# Quick development targets
dev: install-dev ## Setup development environment
	@echo "$(GREEN)✓ Development environment ready!$(NC)"

quick-check: format lint ## Quick code quality check
	@echo "$(GREEN)✓ Quick checks passed!$(NC)"
