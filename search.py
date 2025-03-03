import json
import re
import traceback
import requests
import os
from neo4j_graphrag.retrievers import VectorRetriever
from groq_embedder import Embedder
from config import driver, INDEX_NAME

# Mapping untuk normalisasi nama surah
surah_mapping = {
    'al baqarah': 'Al-Baqarah',
    'baqarah': 'Al-Baqarah',
    'al-baqoroh': 'Al-Baqarah',
    'ali imran': 'Ali Imran',
    'an nisa': 'An-Nisa',
    # Tambahkan mapping lainnya sesuai kebutuhan
}

def initialize_groq():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=10
        )
        response.raise_for_status()
        return GROQ_API_KEY, GROQ_MODEL
    except Exception as e:
        print(f"❌ Gagal terhubung ke Groq: {str(e)}")
        return None, None

def parse_verse_query(query_text):
    # Regex improved untuk menangkap lebih banyak variasi penulisan
    pattern = r"(?:ayat|verse|surat|surah)?\s*(?:ke|nomor)?\s*(\d+).*?(surah|surat|dalam)?\s+([a-zA-Z-\s]+?)(?:\s*ayat|\s*$|\.)"
    match = re.search(pattern, query_text, re.IGNORECASE)
    
    if match:
        verse_number = int(match.group(1))
        raw_surah = match.group(3).strip().lower()
        
        # Normalisasi nama surah
        surah_name = surah_mapping.get(raw_surah, raw_surah.title())
        
        # Validasi angka ayat
        if not validate_verse_number(surah_name, verse_number):
            print(f"⚠️ Nomor ayat {verse_number} tidak valid untuk {surah_name}")
            return None, None
            
        return surah_name, verse_number
    return None, None

def validate_verse_number(surah, number):
    # Daftar panjang ayat per surah (contoh)
    max_verses = {
        'Al-Baqarah': 286,
        'Ali Imran': 200,
        'An-Nisa': 176,
        # Tambahkan data lengkap sesuai kebutuhan
    }
    
    max_num = max_verses.get(surah, 300)  # Default 300 untuk surah tidak terdaftar
    return 1 <= number <= max_num

def get_specific_verse(surah_name, verse_number):
    try:
        result = driver.execute_query(
            """MATCH (s:Surah {name_latin: $surah_name})-[:HAS_AYAT]->(a:Ayat {number: $verse_number})
            OPTIONAL MATCH (a)-[:HAS_TRANSLATION]->(t:Translation)
            OPTIONAL MATCH (a)-[:HAS_TAFSIR]->(taf:Tafsir)
            RETURN s.name_latin as surah, 
                   a.number as ayat_number, 
                   a.text as arabic, 
                   t.text as translation, 
                   taf.text as tafsir""",
            {"surah_name": surah_name, "verse_number": verse_number}
        )
        
        if result.records:
            print(f"✅ Ditemukan ayat {verse_number} dari surah {surah_name}: {result.records}")
            return result.records
        else:
            print(f"⚠️ Tidak ditemukan ayat {verse_number} di surah {surah_name}")
            return None
        
    except Exception as e:
        print(f"❌ Error query spesifik: {traceback.format_exc()}")
        return None



def process_vector_query(query_text):
    try:
        query_vector = Embedder.embed_text(query_text)
        print(f"🔍 Embedding query dimensi: {len(query_vector)}")  # Tambahkan debugging

        result = driver.execute_query(
            """CALL db.index.vector.queryNodes('ayat_embeddings', 5, $query_vector)
            YIELD node, score
            MATCH (s:Surah)-[:HAS_AYAT]->(node)
            RETURN node.text as text, 
                   s.name_latin as surah,
                   node.number as ayat_number, 
                   score
            ORDER BY score DESC
            LIMIT 3""",
            query_vector=query_vector
        )
        
        if result.records:
            print(f"✅ Hasil pencarian vektor: {result.records}")  # Tambahkan debugging
        else:
            print(f"⚠️ Tidak ada hasil dari vector search.")
        
        return result.records
    
    except Exception as e:
        print(f"❌ Error vector search: {traceback.format_exc()}")
        return []


def build_context(records, is_specific=False):
    context = []
    for record in records:
        try:
            ctx = f"""
            📖 Surah: {record['surah']}
            Ayat {record['ayat_number']}:
            Arab: {record.get('arabic', '')}
            Terjemahan: {record.get('translation', '')}
            Tafsir: {record.get('tafsir', '')}
            """
            if is_specific:
                ctx += "\n🔵 [AYAT SPESIFIK YANG DIMINTA]"
            context.append(ctx)
        except KeyError as e:
            print(f"Error format data: {e}")
    return context

def generate_prompt(context, query_text, is_specific=False):
    base_prompt = f"""**Instruksi Sistem**
Anda adalah AI Asisten ahli tafsir Al-Quran. Berikan jawaban dengan struktur:
1. Pendahuluan
2. Ayat Arab + Terjemahan
3. Tafsir
4. Kontekstualisasi
5. Kesimpulan

**Konteks**:
{"".join(context)}

**Pertanyaan**:
{query_text}"""

    if is_specific:
        base_prompt += "\n\n**CATATAN**: User meminta ayat spesifik. Fokuskan jawaban pada ayat tersebut."

    print(f"🔍 Prompt yang dikirim ke Groq API:\n{base_prompt}")  # Debugging
    return base_prompt

def get_verse_by_text(text):
    """Cari ayat berdasarkan teks jika vector search gagal."""
    try:
        result = driver.execute_query(
            """MATCH (s:Surah)-[:HAS_AYAT]->(a:Ayat)
            WHERE a.text CONTAINS $text
            RETURN s.name_latin as surah, a.number as ayat_number, a.text as arabic""",
            {"text": text}
        )
        
        if result.records:
            print(f"✅ Ditemukan langsung dengan pencarian teks: {result.records}")
            return result.records
        else:
            print(f"⚠️ Tidak ada hasil dari pencarian teks langsung.")
            return None
    except Exception as e:
        print(f"❌ Error pencarian teks: {traceback.format_exc()}")
        return None


def process_query(query_text, retriever, GROQ_API_KEY, GROQ_MODEL):
    try:
        print(f"\n🔍 Memproses query: '{query_text}'")
        
        # Step 1: Cek query spesifik
        surah_name, verse_number = parse_verse_query(query_text)
        specific_records = None
        
        if surah_name and verse_number:
            print(f"🔎 Deteksi query spesifik - Surah: {surah_name}, Ayat: {verse_number}")
            specific_records = get_specific_verse(surah_name, verse_number)
            
        # Step 2: Hybrid handling
        if specific_records:
            context = build_context(specific_records, is_specific=True)
            is_specific = True
        else:
            vector_records = process_vector_query(query_text)
            context = build_context(vector_records)
            is_specific = False

        if not context:
            return "⚠️ Maaf, tidak menemukan data yang relevan."

        # Step 3: Generate prompt
        prompt = generate_prompt(context, query_text, is_specific)
        
        # Step 4: Call Groq API
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 5000
            }
        )

        print("Respons API:", response.text)
        
        # Periksa status code dan struktur respons
        if response.status_code != 200:
            print(f"❌ Gagal memanggil API Groq. Status code: {response.status_code}")
            print(f"Respons error: {response.text}")
            return "⚠️ Maaf, terjadi kesalahan saat memproses permintaan."

        response_data = response.json()
        
        # Pastikan key 'choices' ada dalam respons
        if "choices" not in response_data:
            print(f"❌ Struktur respons tidak valid: {response_data}")
            return "⚠️ Maaf, terjadi kesalahan dalam format respons."

        return response_data["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        return "❌ Terjadi kesalahan dalam memproses permintaan."

def main():
    print("Selamat datang di Chatbot Tafsir Al-Quran")
    GROQ_API_KEY, GROQ_MODEL = initialize_groq()
    
    if not GROQ_API_KEY:
        return

    retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=Embedder,
        return_properties=["id", "text", "surah", "ayat_number"]
    )

    try:
        while True:
            query = input("\n💭 Masukkan pertanyaan: ").strip()
            if query.lower() in ['keluar', 'exit']:
                break
                
            if query:
                answer = process_query(query, retriever, GROQ_API_KEY, GROQ_MODEL)
                print("\n💡 Jawaban:")
                print(answer)
                
    finally:
        driver.close()

if __name__ == "__main__":
    main()