.PHONY: setup lint format test bench bench-check serve docker-build docker-run clean

UV ?= uv
BENCH_DIR ?= benchmarks

setup:
	$(UV) sync --all-extras
	$(UV) run clausewise prefetch-embeddings

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run python scripts/check_quality.py

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest -q

bench:
	$(UV) run python -m clausewise.bench --out $(BENCH_DIR)
	$(UV) run python scripts/check_readme_numbers.py

bench-check:
	$(UV) run python -m clausewise.bench --out $(BENCH_DIR) --check benchmarks/thresholds.json

serve:
	$(UV) run clausewise serve --host 127.0.0.1 --port 8000

docker-build:
	docker build -t clausewise:latest .

docker-run:
	docker run --rm -p 8000:8000 clausewise:latest

clean:
	rm -rf .pytest_cache .ruff_cache .hypothesis data/*.jsonl.gz data/*.joblib
