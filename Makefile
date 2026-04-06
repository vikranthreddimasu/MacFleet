.PHONY: test lint format install dev clean bench

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=macfleet --cov-report=term-missing

lint:
	ruff check macfleet/ tests/
	mypy macfleet/ --ignore-missing-imports

format:
	ruff format macfleet/ tests/
	ruff check --fix macfleet/ tests/

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

bench:
	python -m macfleet.cli.main bench --type compute
	python -m macfleet.cli.main bench --type network --size-mb 1
	python -m macfleet.cli.main bench --type allreduce --size-mb 1

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
