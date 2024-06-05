IMAGE_NAME=mlops_imagem
TAG=latest

.PHONY: all
all: install test format build 

.PHONY: test
test:
	python -m unittest discover -s tests

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

.PHONY: clean
clean:
	docker rmi $(IMAGE_NAME):$(TAG)

.PHONY: format
format:
	black **/*.py --line-length 80
