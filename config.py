from neo4j import GraphDatabase

# Konfigurasi Neo4j
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")

# Label dan Property
QURAN_LABEL = "Quran"
SURAH_LABEL = "Surah"
AYAT_LABEL = "Ayat"
EMBEDDING_PROPERTY = "embedding"

# Nama Vector Index
AYAT_INDEX = "ayat_embeddings"
SURAH_INDEX = "surah_embeddings"
DIMENSION = 768  # Sesuai model embedding

# Koneksi ke Neo4j
driver = GraphDatabase.driver(URI, auth=AUTH)

# Konfigurasi Groq
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"