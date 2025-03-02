from neo4j import GraphDatabase
from config import driver, DIMENSION
import sys

def create_indices():
    try:
        with driver.session() as session:
            # Index untuk embedding Ayat (menggunakan VECTOR INDEX)
            session.run("""
                CREATE VECTOR INDEX ayat_embeddings IF NOT EXISTS
                FOR (a:Ayat) 
                ON (a.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """, dim=DIMENSION)

            # Index untuk embedding Surah (menggunakan VECTOR INDEX)
            session.run("""
                CREATE VECTOR INDEX surah_embeddings IF NOT EXISTS
                FOR (s:Surah) 
                ON (s.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """, dim=DIMENSION)

            # Verifikasi indeks yang berhasil dibuat
            check = session.run("""
                SHOW INDEXES 
                WHERE name IN ['ayat_embeddings', 'surah_embeddings'] 
                AND type = 'VECTOR'
            """)

            created_indexes = [record["name"] for record in check]

            if "ayat_embeddings" in created_indexes and "surah_embeddings" in created_indexes:
                print("✅ Semua indeks vektor berhasil dibuat")
                print("Detail Index:")
                print(f"- Nama: ayat_embeddings (Ayat)")
                print(f"- Nama: surah_embeddings (Surah)")
                print(f"- Dimensi: {DIMENSION}")
                print(f"- Similarity Function: cosine")
            else:
                print("❌ Gagal membuat salah satu indeks!")
                sys.exit(1)

    except Exception as e:
        print(f"❌ Error saat membuat indeks: {str(e)}")
        sys.exit(1)
    finally:
        driver.close()

if __name__ == "__main__":
    create_indices()