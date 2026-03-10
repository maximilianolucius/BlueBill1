# Makefile for BlueBill App Testing

.PHONY: help test test-unit test-integration test-performance test-all
.PHONY: test-coverage test-html test-ci smoke-test setup-test install-test-deps
.PHONY: clean lint format check run-dev run-prod

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
TEST_PATH := tests/test_bluebill_app.py
APP_PATH := bluebill_app.py

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)BlueBill App - Test & Development Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)Testing Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(test|Test)' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Development Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v -E '(test|Test)' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  make test-unit          # Run unit tests only"
	@echo "  make test-coverage      # Run tests with coverage"
	@echo "  make smoke-test         # Quick smoke test"
	@echo "  make install-test-deps  # Install testing dependencies"

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(PYTHON) run_tests.py --type all

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTHON) run_tests.py --type unit

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) run_tests.py --type integration

test-performance: ## Run performance tests only
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(PYTHON) run_tests.py --type performance

test-system: ## Run system endpoint tests
	@echo "$(BLUE)Running system tests...$(RESET)"
	$(PYTHON) run_tests.py --type system

test-smartdoc: ## Run SmartDoc tests only
	@echo "$(BLUE)Running SmartDoc tests...$(RESET)"
	$(PYTHON) run_tests.py --type smartdoc

test-fiscal: ## Run Fiscal Classifier tests only
	@echo "$(BLUE)Running Fiscal Classifier tests...$(RESET)"
	$(PYTHON) run_tests.py --type fiscal

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) run_tests.py --coverage

test-html: ## Run tests with HTML coverage report
	@echo "$(BLUE)Running tests with HTML coverage report...$(RESET)"
	$(PYTHON) run_tests.py --coverage --html
	@echo "$(GREEN)Coverage report generated: htmlcov/index.html$(RESET)"

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTHON) run_tests.py --parallel

test-fast: ## Run tests excluding slow ones
	@echo "$(BLUE)Running fast tests only...$(RESET)"
	$(PYTHON) run_tests.py --fast

test-ci: ## Run tests for CI/CD environment
	@echo "$(BLUE)Running CI/CD test suite...$(RESET)"
	$(PYTHON) -c "from run_tests import run_ci_tests; run_ci_tests()"

smoke-test: ## Run quick smoke test
	@echo "$(BLUE)Running smoke test...$(RESET)"
	$(PYTHON) run_tests.py smoke

# Setup and installation
setup-test: ## Setup test environment
	@echo "$(BLUE)Setting up test environment...$(RESET)"
	$(PYTHON) run_tests.py setup

install-test-deps: ## Install testing dependencies
	@echo "$(BLUE)Installing test dependencies...$(RESET)"
	$(PIP) install -r requirements-test.txt

install-deps: ## Install all dependencies
	@echo "$(BLUE)Installing application dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(BLUE)Installing test dependencies...$(RESET)"
	$(PIP) install -r requirements-test.txt

# Development targets
run-dev: ## Run application in development mode
	@echo "$(BLUE)Starting application in development mode...$(RESET)"
	$(PYTHON) $(APP_PATH)

run-prod: ## Run application in production mode
	@echo "$(BLUE)Starting application in production mode...$(RESET)"
	uvicorn bluebill_app:app --host 0.0.0.0 --port 8001 --workers 4

lint: ## Run code linting
	@echo "$(BLUE)Running linter...$(RESET)"
	flake8 $(APP_PATH) $(TEST_PATH) --max-line-length=120 --ignore=E501,W503
	@echo "$(GREEN)Linting complete!$(RESET)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(RESET)"
	black $(APP_PATH) $(TEST_PATH) --line-length=120
	@echo "$(GREEN)Code formatting complete!$(RESET)"

check: ## Run all quality checks
	@echo "$(BLUE)Running all quality checks...$(RESET)"
	@make lint
	@make test-unit
	@echo "$(GREEN)All checks passed!$(RESET)"

# Cleanup targets
clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(RESET)"
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf test_reports/
	rm -rf .coverage
	rm -rf *.pyc
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "$(GREEN)Cleanup complete!$(RESET)"

clean-db: ## Clean test databases
	@echo "$(BLUE)Cleaning test databases...$(RESET)"
	rm -f test_*.db
	rm -f smartdoc_persistence.db
	@echo "$(GREEN)Database cleanup complete!$(RESET)"

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	$(PYTHON) -c "
import json
from $(APP_PATH) import app
with open('api_docs.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"
	@echo "$(GREEN)API documentation generated: api_docs.json$(RESET)"

# Database management
init-db: ## Initialize database
	@echo "$(BLUE)Initializing database...$(RESET)"
	$(PYTHON) -c "from bluebill_app import persistence_manager; persistence_manager.init_database()"
	@echo "$(GREEN)Database initialized!$(RESET)"

backup-db: ## Backup database
	@echo "$(BLUE)Creating database backup...$(RESET)"
	cp smartdoc_persistence.db "smartdoc_backup_$(shell date +%Y%m%d_%H%M%S).db" 2>/dev/null || echo "No database to backup"
	@echo "$(GREEN)Database backup complete!$(RESET)"

# Monitoring and health
health: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	curl -f http://localhost:8001/health || echo "$(RED)Application not running$(RESET)"

status: ## Show application status
	@echo "$(BLUE)Application Status:$(RESET)"
	@ps aux | grep "bluebill_app" | grep -v grep || echo "$(YELLOW)Application not running$(RESET)"
	@echo ""
	@echo "$(BLUE)Database Status:$(RESET)"
	@ls -la smartdoc_persistence.db 2>/dev/null || echo "$(YELLOW)No database file found$(RESET)"

# Development workflow
dev-setup: ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@make install-deps
	@make setup-test
	@make init-db
	@echo "$(GREEN)Development environment ready!$(RESET)"

dev-test: ## Quick development test cycle
	@echo "$(BLUE)Running development test cycle...$(RESET)"
	@make format
	@make lint
	@make test-fast
	@echo "$(GREEN)Development test cycle complete!$(RESET)"

# CI/CD workflow
ci-full: ## Full CI/CD pipeline
	@echo "$(BLUE)Running full CI/CD pipeline...$(RESET)"
	@make clean
	@make install-deps
	@make lint
	@make test-ci
	@make docs
	@echo "$(GREEN)CI/CD pipeline complete!$(RESET)"

# Utility targets
show-config: ## Show current configuration
	@echo "$(BLUE)Current Configuration:$(RESET)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Test file: $(TEST_PATH)"
	@echo "App file: $(APP_PATH)"
	@echo "Working directory: $(shell pwd)"

watch-tests: ## Watch files and run tests on changes (requires entr)
	@echo "$(BLUE)Watching files for changes...$(RESET)"
	find . -name "*.py" | entr -c make test-unit

# Help for specific test types
help-testing: ## Show detailed testing help
	@echo "$(BLUE)BlueBill App - Testing Guide$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test Types:$(RESET)"
	@echo "  Unit tests:        Fast, isolated tests for individual functions"
	@echo "  Integration tests: Tests for complete workflows and interactions"
	@echo "  Performance tests: Tests for speed and resource usage"
	@echo "  System tests:      Tests for system-level endpoints"
	@echo ""
	@echo "$(YELLOW)Test Commands:$(RESET)"
	@echo "  make test               # Run all tests"
	@echo "  make test-unit          # Unit tests only"
	@echo "  make test-integration   # Integration tests only"
	@echo "  make test-coverage      # Tests with coverage"
	@echo "  make test-html          # Tests with HTML coverage report"
	@echo "  make smoke-test         # Quick functionality check"
	@echo ""
	@echo "$(YELLOW)Advanced Testing:$(RESET)"
	@echo "  make test-parallel      # Run tests in parallel (faster)"
	@echo "  make test-fast          # Skip slow tests"
	@echo "  make test-ci            # Full CI/CD test suite"
	@echo ""
	@echo "$(YELLOW)Test Reports:$(RESET)"
	@echo "  Coverage: htmlcov/index.html"
	@echo "  JUnit: test_reports/junit.xml"
	@echo "  HTML: test_reports/report.html"