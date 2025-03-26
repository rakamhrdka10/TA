import json
import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm
from config import driver, DIMENSION
from groq_embedder import Embedder
from sklearn.metrics.pairwise import cosine_similarity
import time

class QuranRelator:
    def __init__(self, driver, threshold=0.75, k=10):
        self.driver = driver
        self.threshold = threshold
        self.k = k  # Jumlah tetangga terdekat yang akan dihubungkan
        self.ayat_embeddings = {}
        self.ayat_data = []  # Untuk menyimpan data dalam format yang mudah diolah

    def load_embeddings(self):
        """Ambil embedding semua ayat yang ada di database"""
        try:
            with self.driver.session() as session:
                query = """
                    MATCH (s:Surah)-[:HAS_AYAT]->(a:Ayat)
                    RETURN s.number AS surah_number, a.number AS ayah_number, a.embedding AS embedding
                """
                result = session.run(query)
                for record in result:
                    surah_num = record["surah_number"]
                    ayah_num = record["ayah_number"]
                    embedding = record["embedding"]
                    
                    # Simpan embedding dalam dict untuk akses cepat
                    self.ayat_embeddings[(surah_num, ayah_num)] = embedding
                    
                    # Simpan informasi dalam format array untuk komputasi batch
                    self.ayat_data.append({
                        'surah_number': surah_num,
                        'ayah_number': ayah_num,
                        'embedding': embedding
                    })
            
            print("✅ Embedding berhasil dimuat!")
            print(f"Jumlah embedding yang dimuat: {len(self.ayat_embeddings)}")
        except Exception as e:
            print(f"❌ Error saat memuat embedding: {str(e)}")
            import traceback
            traceback.print_exc()

    def batch_process_knn(self, batch_size=100):
        """Proses KNN dalam batch untuk menghemat memori"""
        try:
            total_ayat = len(self.ayat_data)
            total_relations = 0
            start_time = time.time()
            
            # Siapkan array numpy untuk semua embedding
            all_embeddings = np.array([data['embedding'] for data in self.ayat_data])
            
            with self.driver.session() as session:
                # Proses dalam batch untuk menghemat memori
                for i in tqdm(range(0, total_ayat, batch_size), desc="Memproses Batch KNN"):
                    end_idx = min(i + batch_size, total_ayat)
                    batch_embeddings = all_embeddings[i:end_idx]
                    
                    # Hitung similarity matrix untuk batch saat ini dengan semua ayat
                    # Ini menghitung similarity dari setiap ayat di batch dengan semua ayat di dataset
                    similarity_matrix = cosine_similarity(batch_embeddings, all_embeddings)
                    
                    # Untuk setiap ayat dalam batch
                    for batch_idx, sim_scores in enumerate(similarity_matrix):
                        global_idx = i + batch_idx
                        ayat1 = self.ayat_data[global_idx]
                        
                        # Dapatkan K tetangga terdekat
                        # Urutkan berdasarkan similarity score
                        # -1 karena kita tidak ingin memasukkan ayat itu sendiri (similarity=1.0)
                        top_indices = np.argsort(sim_scores)[-self.k-1:-1][::-1]
                        
                        batch_relations = []
                        for neighbor_idx in top_indices:
                            # Skip jika itu adalah ayat yang sama
                            if neighbor_idx == global_idx:
                                continue
                                
                            ayat2 = self.ayat_data[neighbor_idx]
                            similarity = sim_scores[neighbor_idx]
                            
                            # Hanya buat relasi jika di atas threshold
                            if similarity >= self.threshold:
                                batch_relations.append({
                                    "surah_number_1": ayat1['surah_number'],
                                    "ayah_number_1": ayat1['ayah_number'],
                                    "surah_number_2": ayat2['surah_number'],
                                    "ayah_number_2": ayat2['ayah_number'],
                                    "similarity": float(similarity)
                                })
                        
                        # Buat relasi dalam batch untuk kinerja yang lebih baik
                        if batch_relations:
                            query = """
                            UNWIND $batch AS relation
                            MATCH (a:Ayat), (b:Ayat)
                            WHERE a.number = relation.ayah_number_1 AND EXISTS {
                                MATCH (s:Surah)-[:HAS_AYAT]->(a) WHERE s.number = relation.surah_number_1
                            }
                            AND b.number = relation.ayah_number_2 AND EXISTS {
                                MATCH (s:Surah)-[:HAS_AYAT]->(b) WHERE s.number = relation.surah_number_2
                            }
                            MERGE (a)-[:RELATED_TO {similarity: relation.similarity}]->(b)
                            MERGE (b)-[:RELATED_TO {similarity: relation.similarity}]->(a)
                            """
                            session.run(query, {"batch": batch_relations})
                            total_relations += len(batch_relations) * 2  # x2 karena relasi timbal balik
                
                elapsed_time = time.time() - start_time
                print(f"✅ Relasi KNN berhasil dibuat! Total relasi: {total_relations}")
                print(f"Waktu yang dibutuhkan: {elapsed_time:.2f} detik")
                
        except Exception as e:
            print(f"❌ Error saat membuat relasi KNN: {str(e)}")
            import traceback
            traceback.print_exc()

    def cleanup_old_relations(self):
        """Hapus relasi RELATED_TO yang lama sebelum membuat yang baru"""
        try:
            with self.driver.session() as session:
                print("Menghapus relasi lama...")
                session.run("MATCH ()-[r:RELATED_TO]->() DELETE r")
                print("✅ Relasi lama berhasil dihapus")
        except Exception as e:
            print(f"❌ Error saat menghapus relasi lama: {str(e)}")

# Main function to run the class methods
if __name__ == "__main__":
    # Gunakan threshold yang lebih tinggi (0.75) dan batasi maksimal 10 tetangga terdekat
    relator = QuranRelator(driver, threshold=0.75, k=10)
    relator.load_embeddings()  # Memuat embedding ayat
    relator.cleanup_old_relations()  # Hapus relasi lama
    relator.batch_process_knn(batch_size=100)  # Buat relasi baru dengan metode batch