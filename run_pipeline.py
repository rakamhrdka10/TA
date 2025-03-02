from typing import List, Dict, Any
import json
import asyncio
from langchain_neo4j import Neo4jGraph
import re
import requests
import time
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
class GroqLLM:
    def __init__(self, api_key: str, model: str, max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.last_request_time = None
        self.requests_this_minute = 0
        self.token_usage = 0
        self.token_limit = 6000  # Token per minute limit
        self.reset_time = None

    async def invoke(self, prompt: str) -> str:
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Check if we need to wait before making another request
                await self._handle_rate_limit()
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 2000,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    # Update token usage
                    usage = result.get("usage", {})
                    self.token_usage += usage.get("total_tokens", 0)
                    self.requests_this_minute += 1
                    self.last_request_time = datetime.now()
                    return result["choices"][0]["message"]["content"]
                
                elif response.status_code == 429:  # Rate limit exceeded
                    error_data = response.json()
                    wait_time = float(error_data["error"]["message"].split("try again in ")[1].split("s")[0])
                    print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                    await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
                    retry_count += 1
                    continue
                
                else:
                    raise Exception(f"Error from Groq: {response.status_code}, {response.text}")
                
            except Exception as e:
                print(f"Error during request: {str(e)}")
                retry_count += 1
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise Exception(f"Failed after {self.max_retries} retries")

    async def _handle_rate_limit(self):
        """Handle rate limiting logic"""
        now = datetime.now()
        
        # Reset counters if it's been more than a minute since last request
        if self.last_request_time and (now - self.last_request_time) > timedelta(minutes=1):
            self.token_usage = 0
            self.requests_this_minute = 0
            self.reset_time = None
        
        # If we're close to the token limit, wait until the next minute
        if self.token_usage >= self.token_limit * 0.9:  # 90% of limit
            if not self.reset_time:
                self.reset_time = self.last_request_time + timedelta(minutes=1)
            
            wait_time = (self.reset_time - now).total_seconds()
            if wait_time > 0:
                print(f"Approaching token limit. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                self.token_usage = 0
                self.requests_this_minute = 0
                self.reset_time = None

class SemanticGraphExtractor:
    def __init__(self, llm: GroqLLM):
        self.llm = llm
        
    async def extract_semantic_graph(self, 
                                   arabic_text: str, 
                                   translation: str, 
                                   tafsir: str,
                                   surah_number: int,
                                   verse_number: int) -> Dict:
        prompt = f"""
        Analyze this Quranic verse, its translation, and tafsir (interpretation), then extract the key semantic elements.
        
        Surah Number: {surah_number}
        Verse Number: {verse_number}
        Arabic: {arabic_text}
        Translation: {translation}
        Tafsir: {tafsir}
        
        Create a semantic graph representation with the following elements:
        1. Main concepts/entities as nodes (e.g., Allah, believers, actions, attributes)
        2. Clear relationships between these nodes
        3. Include relevant context from the tafsir
        
        Return ONLY a JSON object in this exact format with no additional text:
        {{
            "nodes": [
                {{
                    "id": "unique_string_id",
                    "label": "CONCEPT_TYPE",
                    "properties": {{
                        "name": "display_name",
                        "arabic_name": "arabic_name_if_available",
                        "description": "brief_description",
                        "source": "verse_or_tafsir"
                    }}
                }}
            ],
            "relationships": [
                {{
                    "from": "source_node_id",
                    "to": "target_node_id",
                    "type": "RELATIONSHIP_TYPE",
                    "properties": {{
                        "description": "brief_description",
                        "source": "verse_or_tafsir"
                    }}
                }}
            ],
            "metadata": {{
                "surah_number": {surah_number},
                "verse_number": {verse_number},
                "context": "brief_context_from_tafsir"
            }}
        }}

        Guidelines:
        - Node IDs should be unique and descriptive (e.g., "ALLAH", "BELIEVERS", "WORSHIP_ACTION")
        - Node labels should be categories (e.g., DEITY, PERSON, ACTION, CONCEPT)
        - Relationship types should be in UPPERCASE and descriptive
        - Include relevant context from both verse and tafsir
        - Use snake_case for IDs and relationship types
        """
        
        try:
            response = await self.llm.invoke(prompt)
            
            # Extract and clean JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                json_str = json_str.strip()
                
                # Parse and validate JSON
                result = json.loads(json_str)
                
                # Validate required fields
                if not self._validate_graph_structure(result):
                    raise ValueError("Invalid graph structure")
                
                return result
            else:
                print(f"No valid JSON found in response. Raw response: {response}")
                return self._create_empty_graph(surah_number, verse_number)
                
        except Exception as e:
            print(f"Error extracting semantic graph: {e}")
            print(f"Raw response: {response}")
            return self._create_empty_graph(surah_number, verse_number)

    def _validate_graph_structure(self, data: Dict) -> bool:
        """Validate the structure of the extracted semantic graph"""
        try:
            # Check basic structure
            required_keys = {"nodes", "relationships", "metadata"}
            if not all(key in data for key in required_keys):
                return False

            # Validate nodes
            for node in data["nodes"]:
                required_node_keys = {"id", "label", "properties"}
                required_prop_keys = {"name", "description", "source"}
                if not (all(key in node for key in required_node_keys) and 
                       all(key in node["properties"] for key in required_prop_keys)):
                    return False

            # Validate relationships
            for rel in data["relationships"]:
                required_rel_keys = {"from", "to", "type", "properties"}
                required_rel_props = {"description", "source"}
                if not (all(key in rel for key in required_rel_keys) and
                       all(key in rel["properties"] for key in required_rel_props)):
                    return False

            # Validate metadata
            required_metadata = {"surah_number", "verse_number", "context"}
            if not all(key in data["metadata"] for key in required_metadata):
                return False

            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def _create_empty_graph(self, surah_number: int, verse_number: int) -> Dict:
        """Create an empty graph structure with required fields"""
        return {
            "nodes": [],
            "relationships": [],
            "metadata": {
                "surah_number": surah_number,
                "verse_number": verse_number,
                "context": "No context available"
            }
        }

class QuranKnowledgeGraph:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password,
            database=database
        )
        self.embedding_generator = EmbeddingGenerator()
        
    def create_schema(self):
        """Create Neo4j schema with constraints, indices, and vector index"""
        constraints = [
            "CREATE CONSTRAINT semantic_node_id IF NOT EXISTS FOR (n:SemanticNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE (v.surah_number, v.verse_number) IS NODE KEY",
            "CREATE CONSTRAINT concept_type IF NOT EXISTS FOR (c:Concept) REQUIRE c.type IS UNIQUE"
        ]
        
        indices = [
            "CREATE INDEX semantic_node_label IF NOT EXISTS FOR (n:SemanticNode) ON (n.label)",
            "CREATE INDEX verse_number IF NOT EXISTS FOR (v:Verse) ON (v.verse_number)"
        ]
        
        # Vector index for embeddings
        vector_index = """
        CALL db.index.vector.createNodeIndex(
            'verse_embeddings',
            'Verse',
            'embedding',
            384,
            'cosine'
        )
        """
        
        for constraint in constraints:
            try:
                self.graph.query(constraint)
            except Exception as e:
                print(f"Warning: Constraint operation failed: {e}")

        for index in indices:
            try:
                self.graph.query(index)
            except Exception as e:
                print(f"Warning: Index operation failed: {e}")
                
        try:
            self.graph.query(vector_index)
        except Exception as e:
            print(f"Warning: Vector index creation failed: {e}")

    def create_semantic_graph(self, semantic_data: Dict, arabic_text: str, translation: str, tafsir: str):
        """Create or update the semantic graph for a verse with embeddings"""
        try:
            # Generate embeddings for verse content
            combined_text = f"{arabic_text} {translation} {tafsir}"
            embedding = self.embedding_generator.generate_embedding(combined_text)
            
            # Create verse node with metadata and embedding
            verse_cypher = """
            MERGE (v:Verse {
                surah_number: $metadata.surah_number, 
                verse_number: $metadata.verse_number
            })
            SET 
                v.context = $metadata.context,
                v.arabic_text = $arabic_text,
                v.translation = $translation,
                v.tafsir = $tafsir,
                v.embedding = $embedding
            """
            self.graph.query(
                verse_cypher, 
                params={
                    **semantic_data,
                    "arabic_text": arabic_text,
                    "translation": translation,
                    "tafsir": tafsir,
                    "embedding": embedding
                }
            )

            # Create semantic nodes with embeddings
            for node in semantic_data["nodes"]:
                node_text = f"{node['properties']['name']} {node['properties']['description']}"
                node_embedding = self.embedding_generator.generate_embedding(node_text)
                
                node_cypher = """
                MERGE (n:SemanticNode {id: $node.id})
                SET 
                    n.label = $node.label,
                    n.name = $node.properties.name,
                    n.arabic_name = $node.properties.arabic_name,
                    n.description = $node.properties.description,
                    n.source = $node.properties.source,
                    n.embedding = $embedding
                WITH n
                MATCH (v:Verse {
                    surah_number: $metadata.surah_number, 
                    verse_number: $metadata.verse_number
                })
                MERGE (v)-[:CONTAINS]->(n)
                """
                self.graph.query(
                    node_cypher, 
                    params={
                        "node": node,
                        "metadata": semantic_data["metadata"],
                        "embedding": node_embedding
                    }
                )

            # Create relationships
            for rel in semantic_data["relationships"]:
                rel_cypher = f"""
                MATCH (from:SemanticNode {{id: $from}})
                MATCH (to:SemanticNode {{id: $to}})
                MERGE (from)-[r:`{rel['type']}`]->(to)
                SET 
                    r.description = $properties.description,
                    r.source = $properties.source
                """
                self.graph.query(
                    rel_cypher,
                    params={
                        "from": rel["from"],
                        "to": rel["to"],
                        "properties": rel["properties"]
                    }
                )

        except Exception as e:
            print(f"Error creating semantic graph: {e}")
            print(f"Semantic data: {json.dumps(semantic_data, indent=2)}")

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
            """Search verses using semantic similarity"""
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            search_query = """
            CALL db.index.vector.queryNodes(
                'verse_embeddings',
                $limit,
                $embedding
            ) YIELD node, score
            RETURN 
                node.surah_number as surah_number,
                node.verse_number as verse_number,
                node.arabic_text as arabic_text,
                node.translation as translation,
                node.tafsir as tafsir,
                score
            ORDER BY score DESC
            """
            
            results = self.graph.query(
                search_query,
                params={"embedding": query_embedding, "limit": limit}
            )
            
            return results

    def get_verse_graph(self, surah_number: int, verse_number: int) -> Dict:
        """Retrieve the complete semantic graph for a verse"""
        query = """
        MATCH (v:Verse {surah_number: $surah_number, verse_number: $verse_number})-[:CONTAINS]->(n:SemanticNode)
        OPTIONAL MATCH (n)-[r]->(m:SemanticNode)
        RETURN 
            v.context as context,
            collect(distinct {
                id: n.id,
                label: n.label,
                properties: {
                    name: n.name,
                    arabic_name: n.arabic_name,
                    description: n.description,
                    source: n.source
                }
            }) as nodes,
            collect(distinct {
                from: startNode(r).id,
                to: endNode(r).id,
                type: type(r),
                properties: {
                    description: r.description,
                    source: r.source
                }
            }) as relationships
        """
        
        result = self.graph.query(query, params={
            "surah_number": surah_number,
            "verse_number": verse_number
        })
        
        if result and result[0]["nodes"]:
            return {
                "nodes": result[0]["nodes"],
                "relationships": [rel for rel in result[0]["relationships"] if rel["from"]],
                "metadata": {
                    "surah_number": surah_number,
                    "verse_number": verse_number,
                    "context": result[0]["context"]
                }
            }
        return self._create_empty_graph(surah_number, verse_number)

    def _create_empty_graph(self, surah_number: int, verse_number: int) -> Dict:
        return {
            "nodes": [],
            "relationships": [],
            "metadata": {
                "surah_number": surah_number,
                "verse_number": verse_number,
                "context": "No context available"
            }
        }

async def main():
    llm = GroqLLM(
        api_key="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",
        model="llama3-70b-8192"
    )
    
    extractor = SemanticGraphExtractor(llm)
    kg = QuranKnowledgeGraph(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="12345678"
    )
    
    print("Creating schema...")
    kg.create_schema()
    
    # Load Quran data
    with open("quran.json", "r", encoding="utf-8") as f:
        quran_data = json.load(f)
    
    # Process each verse
    for surah in quran_data:
        surah_num = int(surah["number"])
        print(f"\nProcessing Surah {surah['name_latin']} ({surah_num})...")
        
        for verse_num, arabic_text in surah["text"].items():
            verse_num = int(verse_num)
            translation = surah["translations"]["id"]["text"][str(verse_num)]
            tafsir = surah["tafsir"]["id"]["kemenag"]["text"].get(str(verse_num), "")
            
            print(f"\nProcessing verse {verse_num}...")
            print(f"Arabic: {arabic_text}")
            print(f"Translation: {translation}")
            print(f"Tafsir: {tafsir[:100]}...")  # Print first 100 chars of tafsir
            
            # Extract semantic elements
            semantic_data = await extractor.extract_semantic_graph(
                arabic_text=arabic_text,
                translation=translation,
                tafsir=tafsir,
                surah_number=surah_num,
                verse_number=verse_num
            )
                # Create knowledge graph with embeddings
            kg.create_semantic_graph(
                semantic_data=semantic_data,
                arabic_text=arabic_text,
                translation=translation,
                tafsir=tafsir
            )
            print("\nTesting semantic search...")
            search_results = kg.semantic_search("concept of mercy and forgiveness", limit=3)
            print("\nSimilar verses:")
            for result in search_results:
                print(f"Surah {result['surah_number']}, Verse {result['verse_number']}")
                print(f"Score: {result['score']}")
                print(f"Translation: {result['translation']}\n")
            
            # Print extracted data for verification
            print("\nExtracted semantic data:")
            print(json.dumps(semantic_data, indent=2, ensure_ascii=False))
            
            # Create knowledge graph
            kg.create_semantic_graph(
                semantic_data=semantic_data,
                arabic_text=arabic_text,
                translation=translation,
                tafsir=tafsir
            )

            
            # Verify the created graph
            verse_graph = kg.get_verse_graph(surah_num, verse_num)
            print("\nVerified graph in database:")
            print(json.dumps(verse_graph, indent=2, ensure_ascii=False))
    
    print("\nKnowledge graph creation completed!")

if __name__ == "__main__":
    asyncio.run(main())