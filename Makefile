.PHONY: install train serve dashboard simulate test lint format clean \
        docker-build docker-up docker-down docker-train docker-logs

install:
	pip install -e ".[dev]"

train:
	python -m ml_ab_platform.cli.main train

serve:
	python -m ml_ab_platform.cli.main serve

dashboard:
	python -m ml_ab_platform.cli.main dashboard

simulate:
	python -m ml_ab_platform.cli.main simulate --scenario clear-winner --requests 2000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# ---- Docker --------------------------------------------------------------- #
docker-build:
	docker compose build

# One-off: train models into the shared volume before first `up`.
docker-train:
	docker compose run --rm train

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
