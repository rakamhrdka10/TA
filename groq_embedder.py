from neo4j_graphrag.embeddings.base import Embedder as BaseEmbedder
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str):
        return self.model.encode(text).tolist()
    
    def embed_query(self, query: str):
        return self.embed_text(query)

# Inisialisasi dengan model default
Embedder = SentenceTransformerEmbedder()