PT ?= pt-cu126

.PHONY: ci check lint format test test-all test-oldest install

ci: format lint check test-all

check:
	uv run --extra $(PT) pyright

lint:
	uvx ruff check --fix

format:
	uvx ruff format

test:
	uv run pytest -m "not torch"
	uv run xdoctest robokit

test-all:
	uv run --extra $(PT) --extra mj --extra mjwarp pytest -n 8 -m "not slow" .
	uv run --extra $(PT) xdoctest robokit

# The oldest allowed versions, as PR CI runs them.
test-oldest:
	UV_PROJECT_ENVIRONMENT=.venv-oldest UV_RESOLUTION=lowest-direct uv run --python 3.8 --extra pt-cpu --extra mj --extra mjwarp pytest .

install:
	uv sync --extra $(PT)
