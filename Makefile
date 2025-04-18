.PHONY: all install run test build clean

install:
	pip install -r requirements.txt

test:
	pytest

run:
	python3 -m gui.main_window

build:
	pyinstaller focuspanel.spec

clean:
	rm -rf __pycache__ build dist *.spec .pytest_cache .mypy_cache