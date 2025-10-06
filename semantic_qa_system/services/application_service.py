import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List, Optional
import time

from config.settings import settings
from services.search_service import SearchService
from services.qa_service import QAService

class ApplicationService:
    """Main application service coordinating search and Q&A"""
    
    def __init__(self):
        self.database = self._load_database()
        self.embeddings = self._load_embeddings()
        self.search_service = SearchService(self.database, self.embeddings)
        self.qa_service = QAService()
    
    def _load_database(self) -> pd.DataFrame:
        """Load the Stack Overflow database"""
        try:
            database = pd.read_csv(settings.DATA_PATH)
            return database
        except Exception as e:
            raise Exception(f"Failed to load database: {str(e)}")
    
    def _load_embeddings(self) -> np.ndarray:
        """Load pre-computed embeddings"""
        try:
            with open(settings.EMBEDDINGS_PATH, 'rb') as file:
                embeddings = pickle.load(file)
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to load embeddings: {str(e)}")
    
    def search_and_answer(self, query: str, use_approximate: bool = True, k: int = 1) -> Dict[str, Any]:
        """Perform semantic search and generate answer"""
        start_time = time.time()
        
        try:
            # Perform search
            if use_approximate:
                doc_ids, similarities = self.search_service.semantic_search(query, k)
            else:
                doc_ids, similarities = self.search_service.exact_search(query, k)
            
            # Get the most relevant document
            best_doc_id = doc_ids[0]
            best_similarity = similarities[0]
            document = self.search_service.get_document(best_doc_id)
            
            if not document:
                return {
                    'success': False,
                    'error': 'No relevant document found',
                    'latency_ms': (time.time() - start_time) * 1000
                }
            
            # Build context
            context = f"Question: {document['question']}\nAnswer: {document['answer']}"
            
            # Generate answer
            answer = self.qa_service.generate_answer(query, context)
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'source_document': {
                    'id': document['id'],
                    'question': document['question'],
                    'answer': document['answer'],
                    'similarity_score': best_similarity
                },
                'search_method': 'approximate' if use_approximate else 'exact',
                'latency_ms': (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the loaded database"""
        return {
            'total_documents': len(self.database),
            'columns': self.database.columns.tolist(),
            'embeddings_shape': self.embeddings.shape
        }

# Global application service instance
app_service = ApplicationService()