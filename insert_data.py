import json
import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm
from config import driver, DIMENSION
from groq_embedder import Embedder

def chunk_text(text, max_tokens=512, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap  # Geser ke depan dengan overlap
    
    return chunks

def validate_embedding(vector):
    if not isinstance(vector, list):
        raise ValueError("Embedding harus berupa list")

    if len(vector) != DIMENSION:
        raise ValueError(f"❌ Dimensi embedding salah: {len(vector)}, diharapkan {DIMENSION}")
    
    if all(v == 0 for v in vector):
        raise ValueError("Embedding tidak valid (all zeros)")
    
    return vector

def flatten_embeddings(embeddings):
    """Mengambil rata-rata dari beberapa embedding untuk menjaga dimensi tetap 768."""
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    return validate_embedding(avg_embedding)  # Pastikan tetap 768 dimensi

def insert_quran_data():
    with open("quran.json", "r", encoding="utf-8") as file:
        quran_data = json.load(file)
    
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Hapus semua data sebelumnya
            session.run("CREATE (:Quran {name: 'Al-Quran'})")  # Buat root node Al-Quran
            
            total_ayat = sum(len(surah["text"]) for surah in quran_data)
            progress_bar = tqdm(total=total_ayat, desc="Memproses Ayat")
            
            for surah in quran_data:
                surah_id = int(surah["number"])
                surah_name = surah["name"]
                surah_name_latin = surah["name_latin"]
                number_of_ayah = int(surah["number_of_ayah"])
                
                surah_text = f"Surah {surah_name} ({surah_name_latin}), jumlah ayat {number_of_ayah}"
                surah_chunks = chunk_text(surah_text)
                surah_embeddings = [Embedder.embed_text(chunk) for chunk in surah_chunks]
                
                # Ambil rata-rata embedding agar sesuai format yang diterima Neo4j
                flattened_surah_embedding = flatten_embeddings(surah_embeddings)
                
                session.run(
                    """MATCH (q:Quran {name: 'Al-Quran'})
                        CREATE (s:Surah {
                            number: $number,
                            name: $name,
                            name_latin: $name_latin,
                            number_of_ayah: $number_of_ayah,
                            embedding: $embedding
                        })
                        CREATE (q)-[:HAS_SURAH]->(s)
                    """,
                    {
                        "number": surah_id,
                        "name": surah_name,
                        "name_latin": surah_name_latin,
                        "number_of_ayah": number_of_ayah,
                        "embedding": flattened_surah_embedding
                    }
                )
                
                for ayah_num, ayah_text in surah["text"].items():
                    translation = surah.get("translations", {}).get("id", {}).get("text", {}).get(ayah_num, "")
                    tafsir = surah.get("tafsir", {}).get("id", {}).get("kemenag", {}).get("text", {}).get(ayah_num, "")

                    # Format teks yang akan di-embed (termasuk nomor ayat)
                    combined_text = f"Surah {surah_name} Ayat {ayah_num}: {ayah_text} | Terjemahan: {translation} | Tafsir: {tafsir}"
                    
                    ayah_chunks = chunk_text(combined_text)
                    ayah_embeddings = [Embedder.embed_text(chunk) for chunk in ayah_chunks]
                    
                    # Ambil rata-rata embedding agar sesuai format yang diterima Neo4j
                    flattened_ayah_embedding = flatten_embeddings(ayah_embeddings)
                    
                    session.run(
                        """MATCH (s:Surah {number: $surah_number})
                            CREATE (a:Ayat {
                                number: $number,
                                text: $text,
                                translation: $translation,
                                tafsir: $tafsir,
                                embedding: $embedding
                            })
                            CREATE (s)-[:HAS_AYAT]->(a)
                        """,
                        {
                            "surah_number": surah_id,
                            "number": int(ayah_num),
                            "text": ayah_text,
                            "translation": translation,
                            "tafsir": tafsir,
                            "embedding": flattened_ayah_embedding
                        }
                    )
                    
                    progress_bar.update(1)
            
            progress_bar.close()
            print("✅ Data berhasil dimasukkan!")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        driver.close()

if __name__ == "__main__":
    insert_quran_data()
