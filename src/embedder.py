from sentence_transformers import SentenceTransformer
from typing import List

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_embedding_dimension()
        
    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document chunks/passages WITHOUT any instruction prefix.
        Normalizes embeddings to unit length.
        """
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed search queries WITH the required instruction prefix.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_query = f"{instruction}{query}"
        embedding = self.model.encode(
            formatted_query, 
            normalize_embeddings=True
        )
        return embedding.tolist()