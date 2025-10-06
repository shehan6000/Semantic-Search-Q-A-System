import numpy as np
import pandas as pd
import scann
from typing import Tuple, List, Optional
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from services.embedding_service import embedding_service

class SearchService:
    """Service for semantic search operations"""
    
    def __init__(self, database: pd.DataFrame, embeddings: np.ndarray):
        self.database = database
        self.embeddings = embeddings
        self.index = self._build_index()
    
    def _build_index(self) -> scann.scann_ops_pybind.ScannSearcher:
        """Build ScaNN index for efficient similarity search"""
        try:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )
            
            index = scann.scann_ops_pybind.builder(
                normalized_embeddings, 
                num_neighbors=10, 
                distance_measure="dot_product"
            ).tree(
                num_leaves=settings.SEARCH_NUM_LEAVES,
                num_leaves_to_search=settings.SEARCH_LEAVES_TO_SEARCH,
                training_sample_size=min(settings.TRAINING_SAMPLE_SIZE, len(normalized_embeddings))
            ).score_ah(
                2,
                anisotropic_quantization_threshold=0.2
            ).reorder(100).build()
            
            return index
        except Exception as e:
            raise Exception(f"Failed to build search index: {str(e)}")
    
    def semantic_search(self, query: str, k: int = 1) -> Tuple[List[int], List[float]]:
        """Perform semantic search using approximate nearest neighbors"""
        try:
            query_embedding = embedding_service.get_single_embedding(query)
            
            # Normalize query embedding for cosine similarity
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            
            neighbors, distances = self.index.search(
                normalized_query, 
                final_num_neighbors=k
            )
            
            return neighbors.tolist(), distances.tolist()
        except Exception as e:
            raise Exception(f"Semantic search failed: {str(e)}")
    
    def exact_search(self, query: str, k: int = 1) -> Tuple[List[int], List[float]]:
        """Perform exact cosine similarity search"""
        try:
            query_embedding = embedding_service.get_single_embedding(query)
            
            # Calculate cosine similarity
            cos_sim_array = cosine_similarity(
                [query_embedding], 
                self.embeddings
            )[0]
            
            # Get top k results
            top_k_indices = np.argpartition(cos_sim_array, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(cos_sim_array[top_k_indices])][::-1]
            
            similarities = cos_sim_array[top_k_indices].tolist()
            
            return top_k_indices.tolist(), similarities
        except Exception as e:
            raise Exception(f"Exact search failed: {str(e)}")

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Retrieve document by ID"""
        try:
            if 0 <= doc_id < len(self.database):
                row = self.database.iloc[doc_id]
                return {
                    'id': doc_id,
                    'question': row['input_text'],
                    'answer': row['output_text'],
                    'embeddings': row['embeddings']
                }
            return None
        except Exception as e:
            raise Exception(f"Failed to get document {doc_id}: {str(e)}")