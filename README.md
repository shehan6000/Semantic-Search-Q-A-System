# Semantic Search & Q&A System 

## Overview

A production-ready semantic search and question answering system that uses Google Cloud's Vertex AI to provide intelligent responses based on Stack Overflow data. The system combines embedding-based semantic search with large language model generation to deliver accurate, context-aware answers.

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Deployment](#deployment)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)
9. [Performance](#performance)

## Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │ ── │   FastAPI API    │ ── │  Q&A Service    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              │                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │ ── │  Search Service  │ ── │ Embedding Service│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              │                         │
                        ┌──────────────────┐    ┌─────────────────┐
                        │   ScaNN Index    │ ── │  Vertex AI      │
                        └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Query Processing**: User query → Embedding generation
2. **Semantic Search**: Query embedding → Similarity search → Relevant documents
3. **Answer Generation**: Context + Query → LLM → Formatted response
4. **Response Delivery**: Structured JSON response to client

## Installation

### Prerequisites

- Python 3.10+
- Google Cloud Project with Vertex AI enabled
- Service account credentials with Vertex AI permissions

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd semantic_qa_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Google Cloud credentials

# Run the application
python main.py
```

### Docker Installation

```bash
# Build the image
docker build -t semantic-qa-system .

# Run the container
docker run -p 8080:8080 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
  -v /path/to/credentials.json:/credentials.json \
  semantic-qa-system
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | Google Cloud Project ID | Required |
| `GOOGLE_CLOUD_REGION` | GCP Region | `us-central1` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | Required |
| `API_HOST` | FastAPI host address | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8080` |

### Model Configuration

```python
# In config/settings.py
EMBEDDING_MODEL = "textembedding-gecko@001"    # For text embeddings
GENERATION_MODEL = "text-bison@001"           # For answer generation
MAX_OUTPUT_TOKENS = 1024                      # Response length limit
TEMPERATURE = 0.2                             # Creativity control (0-1)
```

### Search Configuration

```python
SEARCH_NUM_LEAVES = 25           # ScaNN index leaves
SEARCH_LEAVES_TO_SEARCH = 10     # Leaves to search per query
TRAINING_SAMPLE_SIZE = 2000      # Samples for index training
```

## API Reference

### Base URL
```
http://localhost:8080
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "database_info": {
    "total_documents": 1500,
    "columns": ["input_text", "output_text", "embeddings"],
    "embeddings_shape": [1500, 768]
  }
}
```

#### 2. Semantic Search (POST)
```http
POST /search
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "How to concatenate dataframes in pandas",
  "use_approximate": true,
  "top_k": 1
}
```

**Parameters:**
- `query` (string, required): The search question
- `use_approximate` (boolean, optional): Use approximate search for speed (default: true)
- `top_k` (integer, optional): Number of results to return (default: 1)

**Response:**
```json
{
  "success": true,
  "query": "How to concatenate dataframes in pandas",
  "answer": "You can concatenate pandas DataFrames using pd.concat()...",
  "source_document": {
    "id": 42,
    "question": "Best way to combine multiple pandas dataframes",
    "answer": "Use pd.concat([df1, df2]) or df1.append(df2)...",
    "similarity_score": 0.894
  },
  "search_method": "approximate",
  "latency_ms": 245.67
}
```

#### 3. Semantic Search (GET)
```http
GET /search?query=your+query&use_approximate=true&top_k=1
```

#### 4. Get Document
```http
GET /documents/{doc_id}
```

**Response:**
```json
{
  "id": 42,
  "question": "Best way to combine multiple pandas dataframes",
  "answer": "Use pd.concat([df1, df2]) or df1.append(df2)...",
  "embeddings": [0.123, 0.456, ...]
}
```

## Usage Examples

### Python Client

```python
import requests
import json

class QAClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def ask_question(self, question, use_approximate=True):
        payload = {
            "query": question,
            "use_approximate": use_approximate,
            "top_k": 1
        }
        
        response = requests.post(f"{self.base_url}/search", json=payload)
        return response.json()
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = QAClient()

# Check health
print(client.health_check())

# Ask a question
result = client.ask_question("How to merge dataframes in pandas?")
print(f"Answer: {result['answer']}")
print(f"Similarity Score: {result['source_document']['similarity_score']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# Search with POST
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to handle missing data in pandas", "use_approximate": true}'

# Search with GET
curl "http://localhost:8080/search?query=How%20to%20handle%20missing%20data%20in%20pandas&use_approximate=true"
```

### JavaScript Client

```javascript
class QAClient {
    constructor(baseUrl = 'http://localhost:8080') {
        this.baseUrl = baseUrl;
    }

    async askQuestion(question, useApproximate = true) {
        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: question,
                use_approximate: useApproximate,
                top_k: 1
            })
        });
        return await response.json();
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
}

// Usage
const client = new QAClient();

// Ask a question
client.askQuestion('How to filter dataframe rows in pandas?')
    .then(result => {
        console.log('Answer:', result.answer);
        console.log('Similarity:', result.source_document.similarity_score);
    });
```

## Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  semantic-qa:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_CLOUD_PROJECT=your-project-id
      - GOOGLE_CLOUD_REGION=us-central1
      - GOOGLE_APPLICATION_CREDENTIALS=/credentials.json
    volumes:
      - ./credentials.json:/credentials.json:ro
    restart: unless-stopped
```

### Google Cloud Run

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/your-project/semantic-qa-system

gcloud run deploy semantic-qa-system \
  --image gcr.io/your-project/semantic-qa-system \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-project-id \
  --allow-unauthenticated
```

### Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-qa-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-qa
  template:
    metadata:
      labels:
        app: semantic-qa
    spec:
      containers:
      - name: semantic-qa
        image: semantic-qa-system:latest
        ports:
        - containerPort: 8080
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/credentials.json"
        volumeMounts:
        - name: google-cloud-key
          mountPath: /credentials.json
          subPath: credentials.json
      volumes:
      - name: google-cloud-key
        secret:
          secretName: google-cloud-credentials
```

## Monitoring & Logging

### Health Monitoring

The system provides a health endpoint that reports:
- Service status
- Database connectivity
- Embeddings status
- Model availability

### Logging Configuration

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)
```

### Key Metrics to Monitor

- **Latency**: Response time for search requests
- **Success Rate**: Percentage of successful queries
- **Embedding Cache Hit Rate**: Cache performance
- **Error Rates**: By error type and endpoint
- **Resource Usage**: CPU, memory, API quotas

## Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Error: Google Cloud credentials not found
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

#### 2. Model Not Available
```bash
# Error: Model not found in region
# Ensure Vertex AI API is enabled and models are available in your region
gcloud services enable aiplatform.googleapis.com
```

#### 3. High Latency
- Use approximate search (`use_approximate=true`)
- Reduce `top_k` parameter
- Implement response caching
- Use larger instance types in production

#### 4. Rate Limiting
```python
# Implement client-side retry logic
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), 
       stop=stop_after_attempt(3))
def make_api_call():
    # Your API call here
    pass
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance

### Benchmarks

| Operation | Approximate Search | Exact Search |
|-----------|-------------------|--------------|
| Single Query | ~200-300ms | ~500-800ms |
| Throughput | ~100 QPS | ~50 QPS |
| Memory Usage | Medium | High |

### Optimization Tips

1. **Use Approximate Search**: For production use, enable approximate search
2. **Batch Requests**: Process multiple queries in batches when possible
3. **Caching**: Implement Redis or similar for frequent queries
4. **CDN**: Cache static assets and common responses
5. **Load Balancing**: Distribute traffic across multiple instances

### Scaling

- **Horizontal Scaling**: Deploy multiple instances behind a load balancer
- **Vertical Scaling**: Use larger machine types for higher throughput
- **Database Sharding**: Partition embeddings across multiple indices
- **Async Processing**: Use async endpoints for non-blocking operations

## Data Management

### Adding New Documents

To update the knowledge base:

1. Add new questions and answers to the CSV file
2. Generate embeddings for new documents
3. Rebuild the ScaNN index
4. Reload the application

### Embedding Generation

```python
from services.embedding_service import embedding_service

# Generate embeddings for new documents
new_questions = ["New question 1", "New question 2"]
new_embeddings = embedding_service.get_embeddings(new_questions)

# Save embeddings
import pickle
with open('new_embeddings.pkl', 'wb') as f:
    pickle.dump(new_embeddings, f)
```

## Security

### Authentication

The system relies on Google Cloud IAM for authentication. Ensure your service account has:

- `roles/aiplatform.user` for Vertex AI access
- `roles/storage.objectViewer` if using Cloud Storage for data

### API Security

- Use HTTPS in production
- Implement API keys or OAuth2 for client authentication
- Set up CORS properly for web clients
- Rate limiting implementation recommended

## Support

For issues and questions:

1. Check the health endpoint: `/health`
2. Review application logs
3. Verify Google Cloud project configuration
4. Ensure sufficient quotas for Vertex AI APIs

