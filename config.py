from neo4j import GraphDatabase

# Konfigurasi Neo4j
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")

# Konfigurasi index embedding
INDEX_NAME = "ayat_embeddings"  # Nama indeks di Neo4j
DIMENSION = 768  # Disesuaikan dengan model embedding yang digunakan
LABEL = "Tafsir"  # Label node di Neo4j
EMBEDDING_PROPERTY = "embedding"  # Properti yang menyimpan embedding

# Koneksi ke Neo4j
driver = GraphDatabase.driver(URI, auth=AUTH)

GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"  # Pastikan model ini benar