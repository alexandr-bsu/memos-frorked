from memos import log
from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.embedders.factory import EmbedderFactory
from memos.graph_dbs.factory import GraphStoreFactory
from memos.memories.textual.item import TextualMemoryItem
from memos.memories.textual.tree_text_memory.retrieve.recall import GraphMemoryRetriever
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

# Step 2: Initialize graph store
graph_config = GraphDBConfigFactory(
    backend="neo4j",
    config={
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "12345678",
        "db_name": "caroline",
        "auto_create": True,
    },
)
graph_store = GraphStoreFactory.from_config(graph_config)

# Step 6: Create embedding for query
query = "When did Caroline go to the LGBTQ support group?"
query_embedding = embedder.embed([query])[0]

# Step 7: Init memory retriever
retriever = GraphMemoryRetriever(graph_store=graph_store, embedder=embedder)

# Step 8: Run hybrid retrieval
retrieved_items: list[TextualMemoryItem] = retriever.retrieve(
    query=query,
    parsed_goal=parsed_goal,
    top_k=10,
    memory_scope="LongTermMemory",
    query_embedding=[query_embedding],
)

# Step 9: Print retrieved memory items
print("\n=== Retrieved Memory Items ===")
for item in retrieved_items:
    print(f"ID: {item.id}")
    print(f"Memory: {item.memory}")
