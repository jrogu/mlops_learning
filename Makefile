#.PHONY: test

all: install test

test:
	python -m unittest tests.test_python_broadcasting

install:
	pip install -r requirements.txt