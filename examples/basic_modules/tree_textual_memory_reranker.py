from memos import log
from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.configs.llm import LLMConfigFactory
from memos.embedders.factory import EmbedderFactory
from memos.graph_dbs.factory import GraphStoreFactory
from memos.llms.factory import LLMFactory
from memos.memories.textual.item import TextualMemoryItem, TreeNodeTextualMemoryMetadata
from memos.memories.textual.tree_text_memory.retrieve.reranker import MemoryReranker
from memos.memories.textual.tree_text_memory.retrieve.retrieval_mid_structs import ParsedTaskGoal


logger = log.get_logger(__name__)

embedder_config = EmbedderConfigFactory.model_validate(
    {
        "backend": "ollama",
        "config": {
            "model_name_or_path": "nomic-embed-text:latest",
        },
    }
)
embedder = EmbedderFactory.from_config(embedder_config)

# Step 1: Load LLM config and instantiate
config = LLMConfigFactory.model_validate(
    {
        "backend": "ollama",
        "config": {
            "model_name_or_path": "qwen3:0.6b",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    }
)
llm = LLMFactory.from_config(config)

# Step 1: Prepare a mock ParsedTaskGoal
parsed_goal = ParsedTaskGoal(
    memories=[
        "Caroline's participation in the LGBTQ community",
        "Historical details of her membership",
        "Specific instances of Caroline's involvement in LGBTQ support groups",
        "Information about Caroline's activities in LGBTQ spaces",
        "Accounts of Caroline's role in promoting LGBTQ+ inclusivity",
    ],
    keys=["Family hiking experiences", "LGBTQ support group"],
    goal_type="retrieval",
    tags=["LGBTQ", "support group"],
)

query = "How can multiple UAVs coordinate to maximize coverage while saving energy?"
query_embedding = embedder.embed([query])[0]


# Step 2: Initialize graph store
graph_config = GraphDBConfigFactory(
    backend="neo4j",
    config={
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "12345678",
        "db_name": "user06alice",
        "auto_create": True,
    },
)
graph_store = GraphStoreFactory.from_config(graph_config)

retrieved_results = [
    TextualMemoryItem(
        id="a88db9ce-3c77-4e83-8d61-aa9ef95c957e",
        memory="Coverage performance is measured using CT (Coverage Time) and FT (Fairness Time) metrics.",
        metadata=TreeNodeTextualMemoryMetadata(
            user_id=None,
            session_id=None,
            status="activated",
            type="fact",
            memory_time="2024-01-01",
            source="file",
            confidence=91.0,
            tags=["coverage", "fairness", "metrics"],
            visibility="public",
            updated_at="2025-06-11T11:51:24.438001",
            memory_type="LongTermMemory",
            key="Coverage Metrics",
            value="CT and FT used for long-term area and fairness evaluation",
            sources=["paper://multi-uav-coverage/metrics"],
            embedding=[0.01] * 768,
            background="",
        ),
    )
]

# Step 7: Init memory retriever
reranker = MemoryReranker(llm=llm, embedder=embedder)


# Step 8: Print retrieved memory items before ranking
print("\n=== Retrieved Memory Items (Before Rerank) ===")
for idx, item in enumerate(retrieved_results):
    print(f"[Original #{idx + 1}] ID: {item.id}")
    print(f"Memory: {item.memory[:200]}...\n")

# Step 9: Rerank
ranked_results = reranker.rerank(
    query=query,
    query_embedding=query_embedding,
    graph_results=retrieved_results,
    top_k=10,
    parsed_goal=parsed_goal,
)

# Step 10: Print ranked memory items with original positions
print("\n=== Memory Items After Rerank (Sorted) ===")
id_to_original_rank = {item.id: i + 1 for i, item in enumerate(retrieved_results)}

for idx, ranked_results_i in enumerate(ranked_results):
    item = ranked_results_i[0]
    original_rank = id_to_original_rank.get(item.id, "-")
    print(f"[Ranked #{idx + 1}] ID: {item.id} (Original #{original_rank})")
    print(f"Memory: {item.memory[:200]}...\n")
