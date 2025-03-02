import asyncio
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from config import driver
from llm_config import llm

# Membuat pipeline untuk membangun Knowledge Graph
async def build_knowledge_graph():
    kg_builder = SimpleKGPipeline(
        llm=llm,  # Menggunakan GroqLLM
        driver=driver,
        from_pdf=False,
        perform_entity_resolution=True  # Menggabungkan entitas yang sama
    )
    await kg_builder.run_async()

# Jalankan pipeline
if __name__ == "__main__":
    asyncio.run(build_knowledge_graph())
