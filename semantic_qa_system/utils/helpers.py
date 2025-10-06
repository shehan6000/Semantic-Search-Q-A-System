import time
from typing import List
import numpy as np
from vertexai.language_models import TextEmbeddingModel

from config.settings import settings

def encode_text_to_embedding_batched(
    sentences: List[str],
    api_calls_per_second: float = 20/60,
    batch_size: int = 5
) -> np.ndarray:
    """Batch process text embeddings with rate limiting"""
    embeddings = []
    delay = 1 / api_calls_per_second
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            batch_embeddings = TextEmbeddingModel.from_pretrained(
                settings.EMBEDDING_MODEL
            ).get_embeddings(batch)
            
            for embedding in batch_embeddings:
                embeddings.append(embedding.values)
            
            # Rate limiting
            if i + batch_size < len(sentences):
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            # Add zero embeddings for failed batches
            for _ in batch:
                embeddings.append(np.zeros(768))  # Adjust based on embedding dimension
    
    return np.array(embeddings)