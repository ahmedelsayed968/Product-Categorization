.EXPORT_ALL_VARIABLES:
.PHONY: venv install pre-commit clean

GLOBAL_PYTHON = $(shell python3.10 -c 'import sys; print(sys.executable)')
LOCAL_PYTHON = ./.venv/bin/python
LOCAL_PRE_COMMIT = ./.venv/lib/python3.10/site-packages/pre_commit

setup: venv install pre-commit

venv: $(GLOBAL_PYTHON)
	@echo "Creating .venv..."
	poetry env use $(GLOBAL_PYTHON)

install: ${LOCAL_PYTHON}
	@echo "Installing dependencies..."
	poetry install --no-root --sync

pre-commit: ${LOCAL_PYTHON} ${LOCAL_PRE_COMMIT}
	@echo "Setting up pre-commit..."
	. ./.venv/bin/activate && poetry run pre-commit install
	. ./.venv/bin/activate && poetry run pre-commit autoupdate

clean:
	rm -rf .git/hooks .venv poetry.lock
