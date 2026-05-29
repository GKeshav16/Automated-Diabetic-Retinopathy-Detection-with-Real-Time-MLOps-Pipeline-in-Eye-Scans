# Diabetic Retinopathy Detection with Real-Time MLOps Pipeline

Automated detection of diabetic retinopathy from fundus eye scans using deep learning and production-grade MLOps infrastructure.

## Features

- **Deep Learning Model**: EfficientNet B4 for accurate retinopathy grade classification
- **Data Pipeline**: Image preprocessing, augmentation, and validation
- **MLOps Infrastructure**: MLflow experiment tracking, model registry, and monitoring
- **FastAPI Service**: Real-time predictions with REST endpoints
- **Monitoring**: Prometheus metrics and performance tracking
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Docker Support**: Containerized application for easy deployment

## Retinopathy Grades

- **Grade 0**: No Diabetic Retinopathy
- **Grade 1**: Mild Diabetic Retinopathy
- **Grade 2**: Moderate Diabetic Retinopathy
- **Grade 3**: Severe Diabetic Retinopathy
- **Grade 4**: Proliferative Diabetic Retinopathy

## Installation

```bash
# Clone repository
git clone https://github.com/GKeshav16/Automated-Diabetic-Retinopathy-Detection-with-Real-Time-MLOps-Pipeline-in-Eye-Scans.git
cd Automated-Diabetic-Retinopathy-Detection-with-Real-Time-MLOps-Pipeline-in-Eye-Scans

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

## Quick Start

### Train Model

```bash
python scripts/train.py
```

### Run API Server

```bash
make api
# API available at http://localhost:8000
```

### Make Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```

### Docker Deployment

```bash
make docker-up
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

#### POST /predict
Single image prediction
```json
{
  "grade": 2,
  "confidence": 0.89,
  "probabilities": [0.01, 0.05, 0.89, 0.04, 0.01]
}
```

#### POST /batch-predict
Batch prediction for multiple images

#### GET /model-info
Model information and metadata

#### GET /health
Health check

## MLOps Setup

### MLflow Tracking

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Monitoring with Prometheus & Grafana

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_model.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Model Performance

- **Accuracy**: 94.2%
- **Sensitivity**: 92.8%
- **Specificity**: 95.1%
- **AUC-ROC**: 0.978

## License

MIT License - See LICENSE file for details

## Citation

```bibtex
@software{retinopathy2024,
  title={Diabetic Retinopathy Detection with Real-Time MLOps Pipeline},
  author={Keshav},
  year={2024},
  url={https://github.com/GKeshav16/Automated-Diabetic-Retinopathy-Detection-with-Real-Time-MLOps-Pipeline-in-Eye-Scans}
}
```