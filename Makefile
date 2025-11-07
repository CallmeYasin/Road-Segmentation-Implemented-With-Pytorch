.PHONY: install test lint build run clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt  # For dev tools
	# If GPU needed: pip install -r requirements-cuda.txt

test:
	pytest -v

lint:
	flake8 .
	mypy .

clean:
	rm -rf __pycache__ *.pyc