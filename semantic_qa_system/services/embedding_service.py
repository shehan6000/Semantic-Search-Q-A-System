import numpy as np
from typing import List, Optional
from vertexai.language_models import TextEmbeddingModel

from config.settings import settings
from utils.helpers import encode_text_to_embedding_batched

class EmbeddingService:
    """Service for handling text embeddings"""
    
    def __init__(self):
        self.model = TextEmbeddingModel.from_pretrained(settings.EMBEDDING_MODEL)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        try:
            if len(texts) == 1:
                # Single embedding
                embedding = self.model.get_embeddings(texts)[0].values
                return np.array([embedding])
            else:
                # Batch embeddings
                return encode_text_to_embedding_batched(
                    sentences=texts,
                    api_calls_per_second=20/60,
                    batch_size=5
                )
        except Exception as e:
            raise Exception(f"Failed to get embeddings: {str(e)}")
    
    def get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        return self.get_embeddings([text])[0]

embedding_service = EmbeddingService()