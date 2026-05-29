.PHONY: help install requirements train test api docker-build docker-up docker-down clean inference

help:
	@echo "Diabetic Retinopathy Detection - Available Commands"
	@echo "=================================================="
	@echo "make install          - Install project dependencies"
	@echo "make train            - Train the model"
	@echo "make inference        - Run inference on test images"
	@echo "make test             - Run tests"
	@echo "make api              - Start FastAPI server (🌐 BROWSER)"
	@echo "make clean            - Clean cache and temporary files"
	@echo "make docker-build     - Build Docker image"
	@echo "make docker-up        - Start Docker containers"
	@echo "make docker-down      - Stop Docker containers"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python scripts/train.py --config config.yaml --plot
	@echo "✓ Training completed"

inference:
	@if [ -z "$(IMAGE)" ] && [ -z "$(DIR)" ]; then \
		echo "Usage: make inference IMAGE=path/to/image.jpg"; \
		echo "       make inference DIR=path/to/images/"; \
	else \
		if [ -n "$(IMAGE)" ]; then \
			python scripts/inference.py --image $(IMAGE) --model models/checkpoints/best_model.pt; \
		else \
			python scripts/inference.py --directory $(DIR) --model models/checkpoints/best_model.pt --output results.json; \
		fi \
	fi

test:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✓ Tests completed. Coverage report: htmlcov/index.html"

api:
	@echo "🌐 Starting FastAPI Server..."
	@echo "   Open browser: http://localhost:8000"
	@echo "   API Docs: http://localhost:8000/docs"
	python scripts/api.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "✓ Cache cleaned"

docker-build:
	docker build -t retinopathy-detection:latest .
	@echo "✓ Docker image built"

docker-up:
	docker-compose up -d
	@echo "✓ Docker containers started"

docker-down:
	docker-compose down
	@echo "✓ Docker containers stopped"

lint:
	black src/ scripts/
	flake8 src/ scripts/
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/ tests/
	@echo "✓ Code formatted"
