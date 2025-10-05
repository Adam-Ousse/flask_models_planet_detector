# Exoplanet Classification API

A Flask-based REST API for classifying celestial objects as exoplanets or false positives using machine learning models. Designed for deployment on Google Cloud Run.

## Features

- **Multiple Dataset Support**: Handles different dataset types (Kepler, K2) with automatic model selection
- **Binary Classification**: Predicts whether a celestial object is a confirmed exoplanet or a false positive
- **Cloud-Ready**: Optimized for Google Cloud Run deployment with Docker
- **Model Caching**: Efficient in-memory caching of loaded models
- **Health Monitoring**: Built-in health check endpoints

## API Endpoints

### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Exoplanet Classification API",
  "available_datasets": ["kepler", "k2"]
}
```

### 2. Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body (Option 1 - Dictionary of arrays):**
```json
{
  "dataset_type": "kepler",
  "data": {
    "koi_period": [10.5, 20.3],
    "koi_depth": [100.5, 200.8],
    "koi_duration": [3.5, 4.2],
    ...
  }
}
```

**Request Body (Option 2 - Array of objects):**
```json
{
  "dataset_type": "k2",
  "data": [
    {
      "koi_period": 10.5,
      "koi_depth": 100.5,
      "koi_duration": 3.5,
      ...
    },
    {
      "koi_period": 20.3,
      "koi_depth": 200.8,
      "koi_duration": 4.2,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": ["CONFIRMED", "FALSE POSITIVE"],
  "probabilities": [[0.15, 0.85], [0.92, 0.08]],
  "dataset_type": "kepler",
  "num_samples": 2
}
```

### 3. List Models
```http
GET /models
```

**Response:**
```json
{
  "kepler": {
    "model_exists": true,
    "preprocessor_exists": true,
    "cached": true
  },
  "k2": {
    "model_exists": true,
    "preprocessor_exists": true,
    "cached": false
  }
}
```

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

The API will be available at `http://localhost:8080`

### Test the API
```bash
# Health check
curl http://localhost:8080/

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_type": "kepler",
    "data": {
      "koi_period": [10.5],
      "koi_depth": [100.5],
      ...
    }
  }'
```

## Docker Deployment

### Build Docker Image
```bash
docker build -t exoplanet-api .
```

### Run Docker Container Locally
```bash
docker run -p 8080:8080 exoplanet-api
```

### Test Docker Container
```bash
curl http://localhost:8080/
```

## Google Cloud Run Deployment

### Prerequisites
- Google Cloud SDK installed and configured
- Google Cloud project created
- Billing enabled on your Google Cloud project

### Deploy to Cloud Run

1. **Set your project ID:**
```bash
gcloud config set project YOUR_PROJECT_ID
```

2. **Build and push to Google Container Registry:**
```bash
# Build the image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/exoplanet-api

# Or use Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

3. **Deploy to Cloud Run:**
```bash
gcloud run deploy exoplanet-api \
  --image gcr.io/YOUR_PROJECT_ID/exoplanet-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300
```

4. **Get the service URL:**
```bash
gcloud run services describe exoplanet-api --region us-central1 --format 'value(status.url)'
```

### Alternative: One-Command Deployment
```bash
gcloud run deploy exoplanet-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Project Structure

```
flask_models/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── .dockerignore                   # Docker ignore file
├── README.md                       # This file
├── models/
│   └── k2_logistic_model.joblib   # Trained ML models
└── preprocessors/
    └── k2_preprocessor.joblib     # Data preprocessors
```

## Adding New Models

To add support for a new dataset type:

1. Train your model and save it in the `models/` directory
2. Save the preprocessor in the `preprocessors/` directory
3. Update the `MODEL_CONFIG` dictionary in `app.py`:

```python
MODEL_CONFIG = {
    "new_dataset": {
        "model": "models/new_model.joblib",
        "preprocessor": "preprocessors/new_preprocessor.joblib"
    }
}
```

## Configuration

### Environment Variables
- `PORT`: Server port (default: 8080)
- `PYTHONUNBUFFERED`: Enable Python logging (set to 1)

### Gunicorn Configuration
Adjust worker and thread settings in `Dockerfile` based on your needs:
- `--workers`: Number of worker processes (CPU-bound scaling)
- `--threads`: Number of threads per worker (I/O-bound scaling)
- `--timeout`: Request timeout in seconds

## Monitoring and Logging

Cloud Run automatically captures logs. View them with:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=exoplanet-api" --limit 50
```

## Cost Optimization

- Cloud Run charges only for actual usage
- The container uses Python slim image for smaller size
- Models are cached in memory to reduce disk I/O
- Configure minimum instances to 0 for cost savings (cold starts) or 1+ for better performance

## Security Considerations

- Add authentication for production use
- Use Secret Manager for sensitive configuration
- Implement rate limiting
- Add input validation and sanitization
- Enable CORS if needed for web clients

## Troubleshooting

### Common Issues

1. **Memory errors**: Increase memory allocation in Cloud Run deployment
2. **Timeout errors**: Increase timeout value or optimize preprocessing
3. **Model not found**: Ensure model files are included in Docker image
4. **Feature mismatch**: Verify input features match training data

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
