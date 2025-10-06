from typing import Dict, Any, Optional
from vertexai.language_models import TextGenerationModel

from config.settings import settings

class QAService:
    """Service for generating answers using LLM"""
    
    def __init__(self):
        self.model = TextGenerationModel.from_pretrained(settings.GENERATION_MODEL)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using context and query"""
        try:
            prompt = self._build_prompt(query, context)
            
            response = self.model.predict(
                prompt=prompt,
                temperature=settings.TEMPERATURE,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS
            )
            
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for the generation model"""
        return f"""Here is the context: {context}

Using the relevant information from the context, provide an answer to the query: "{query}".

If the context doesn't provide any relevant information, answer with:
[I couldn't find a good match in the document database for your query]

Answer:"""

qa_service = QAService()