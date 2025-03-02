# chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class QuranTextChunker:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def split_quran_data(self, quran_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks_with_metadata = []
        
        for surah in quran_data:
            print(f"Processing Surah {surah['name_latin']} ({surah['number']})...")
            
            # Process Arabic text
            for verse_num, verse_text in surah['text'].items():
                chunks_with_metadata.append({
                    'text': verse_text,
                    'type': 'arabic_text',
                    'surah_number': surah['number'],
                    'verse_number': verse_num,
                    'chunk_id': f"ar_s{surah['number']}_v{verse_num}",
                    'language': 'ar'
                })
            
            # Process Indonesian translation
            for verse_num, verse_translation in surah['translations']['id']['text'].items():
                chunks_with_metadata.append({
                    'text': verse_translation,
                    'type': 'translation',
                    'surah_number': surah['number'],
                    'verse_number': verse_num,
                    'chunk_id': f"tr_s{surah['number']}_v{verse_num}",
                    'language': 'id',
                    'translation_name': surah['translations']['id']['name']
                })
            
            # Process Kemenag tafsir
            if 'tafsir' in surah and 'id' in surah['tafsir']:
                for verse_num, tafsir_text in surah['tafsir']['id']['kemenag']['text'].items():
                    chunks_with_metadata.append({
                        'text': tafsir_text,
                        'type': 'tafsir',
                        'surah_number': surah['number'],
                        'verse_number': verse_num,
                        'chunk_id': f"tf_s{surah['number']}_v{verse_num}",
                        'language': 'id',
                        'tafsir_source': surah['tafsir']['id']['kemenag']['source']
                    })
        
        return chunks_with_metadata