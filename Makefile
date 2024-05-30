#.PHONY: test

all: install test

test:
	python -m unittest discover -s tests

install:
	pip install -r requirements.txt

format:
	black **/*.py --line-length 80