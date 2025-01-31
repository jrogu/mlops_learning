CONTAINER_NAME=mlops_container
IMAGE_NAME=mlops_image
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

.PHONY: docker run
run:
	docker run --name $(CONTAINER_NAME) -d $(IMAGE_NAME):$(TAG)

.PHONY: clean
clean:
	docker rmi $(IMAGE_NAME):$(TAG)

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

.PHONY:
compose_build:
	docker compose build

.PHONY:
compose_up:
	docker compose up -d

.PHONY:
compose_down:
	docker compose down

.PHONY: format	
format:
	black **/*.py --line-length 80
