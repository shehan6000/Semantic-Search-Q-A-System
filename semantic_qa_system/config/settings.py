import os
from typing import Dict, Any

class Settings:
    """Application configuration settings"""
    
    def __init__(self):
        self.PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
        self.REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        self.CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        
        # Model configurations
        self.EMBEDDING_MODEL = "textembedding-gecko@001"
        self.GENERATION_MODEL = "text-bison@001"
        
        # Search configurations
        self.SEARCH_NUM_LEAVES = 25
        self.SEARCH_LEAVES_TO_SEARCH = 10
        self.TRAINING_SAMPLE_SIZE = 2000
        self.MAX_OUTPUT_TOKENS = 1024
        self.TEMPERATURE = 0.2
        
        # Data paths
        self.DATA_PATH = "data/so_database_app.csv"
        self.EMBEDDINGS_PATH = "data/question_embeddings_app.pkl"
        
        # API configurations
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8080"))

settings = Settings()