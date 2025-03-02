import json
from neo4j import GraphDatabase
from tqdm import tqdm
from config import driver
from groq_embedder import Embedder

def validate_embedding(vector):
    if not isinstance(vector, list):
        raise ValueError("Embedding harus berupa list")
    if len(vector) != 768:  
        raise ValueError(f"Dimensi embedding salah: {len(vector)}")
    if all(v == 0 for v in vector):
        raise ValueError("Embedding tidak valid (all zeros)")

def insert_quran_data():
    with open("quran.json", "r", encoding="utf-8") as file:
        quran_data = json.load(file)

    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Hapus semua data sebelumnya

            # Membuat node Quran (sebagai root)
            session.run("CREATE (:Quran {name: 'Al-Quran'})")

            total_ayat = sum(len(surah["text"]) for surah in quran_data)
            progress_bar = tqdm(total=total_ayat, desc="Memproses Ayat")

            for surah in quran_data:
                surah_id = surah["_id"]["$oid"]
                surah_name = surah["name_latin"]
                surah_number = surah["number"]
                number_of_ayat = len(surah["text"])

                # Generate embedding untuk Surah
                surah_text = f"Surah {surah_name}, nomor {surah_number}, jumlah ayat {number_of_ayat}"
                try:
                    surah_embedding = Embedder.embed_text(surah_text)
                    validate_embedding(surah_embedding)
                except Exception as e:
                    print(f"❌ Error embedding Surah {surah_name}: {str(e)}")
                    continue

                # Insert Surah dengan embedding
                session.run(
                    """MATCH (q:Quran {name: 'Al-Quran'})
                    CREATE (s:Surah {
                        id: $id,
                        name_latin: $name_latin,
                        number: $number,
                        number_of_ayat: $number_of_ayat,
                        embedding: $embedding
                    })
                    CREATE (q)-[:HAS_SURAH]->(s)
                    """,
                    {
                        "id": surah_id,
                        "name_latin": surah_name,
                        "number": int(surah_number),
                        "number_of_ayat": number_of_ayat,
                        "embedding": surah_embedding
                    }
                )

                # Insert Ayat
                for ayah_num, ayah_text in surah["text"].items():
                    translation = surah["translations"]["id"]["text"].get(ayah_num, "")
                    tafsir = surah["tafsir"]["id"]["kemenag"]["text"].get(ayah_num, "")

                    # Gabungkan teks ayat, terjemahan, dan tafsir untuk embedding
                    combined_text = f"Surah {surah_name} Ayat {ayah_num}: {ayah_text} {translation} {tafsir}"
                    try:
                        ayat_embedding = Embedder.embed_text(combined_text)
                        validate_embedding(ayat_embedding)
                    except Exception as e:
                        print(f"❌ Error embedding Ayat {ayah_num} di Surah {surah_name}: {str(e)}")
                        continue

                    # Insert Ayat dengan embedding
                    session.run(
                        """MATCH (s:Surah {id: $surah_id})
                        CREATE (a:Ayat {
                            id: $ayat_id,
                            number: $number,
                            text: $text,
                            embedding: $embedding
                        })
                        CREATE (s)-[:HAS_AYAT]->(a)
                        """,
                        {
                            "surah_id": surah_id,
                            "ayat_id": f"{surah_id}-{ayah_num}",
                            "number": int(ayah_num),
                            "text": ayah_text,
                            "embedding": ayat_embedding
                        }
                    )

                    # Insert Terjemahan (tanpa embedding)
                    if translation:
                        session.run(
                            """MATCH (a:Ayat {id: $ayat_id})
                            CREATE (t:Translation {text: $translation})
                            CREATE (a)-[:HAS_TRANSLATION]->(t)""",
                            {
                                "ayat_id": f"{surah_id}-{ayah_num}",
                                "translation": translation
                            }
                        )

                    # Insert Tafsir (tanpa embedding)
                    if tafsir:
                        session.run(
                            """MATCH (a:Ayat {id: $ayat_id})
                            CREATE (t:Tafsir {text: $tafsir})
                            CREATE (a)-[:HAS_TAFSIR]->(t)""",
                            {
                                "ayat_id": f"{surah_id}-{ayah_num}",
                                "tafsir": tafsir
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
