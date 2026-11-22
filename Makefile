UV ?= uv
RUN := $(UV) run
SRC := src/ tests/

.PHONY: sync format format-check lint type-check test check-all

sync:
	$(UV) sync

format:
	$(RUN) autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive $(SRC)
	$(RUN) black $(SRC)
	$(RUN) isort $(SRC)
	$(MAKE) lint

format-check:
	$(RUN) black --check $(SRC)
	$(RUN) isort --check-only $(SRC)
	$(MAKE) lint

lint:
	$(RUN) flake8 $(SRC)
	$(MAKE) type-check

type-check:
	$(RUN) mypy src/

test:
	$(RUN) pytest tests/

check-all: format-check test
