import uuid

from memos.configs.memory import MemoryConfigFactory
from memos.memories.factory import MemoryFactory


config = MemoryConfigFactory(
    backend="naive_text",
    config={
        "extractor_llm": {
            "backend": "ollama",
            "config": {
                "model_name_or_path": "qwen3:0.6b",
                "temperature": 0.0,
                "remove_think_prefix": True,
            },
        }
    },
)
m = MemoryFactory.from_config(config)


print("===== Extract memories =====")
memories = m.extract(
    [
        {"role": "user", "content": "I love tomatoes."},
        {"role": "assistant", "content": "Great! Tomatoes are delicious."},
    ]
)
print(memories)
print()

print("==== Add memories ====")
m.add(memories)
example_id = str(uuid.uuid4())
m.add([{"id": example_id, "memory": "User is Chinese.", "metadata": {"type": "opinion"}}])
print(m.get_all())
print()

print("==== Search memories ====")
search_results = m.search("Tell me more about the user", top_k=2)
print(search_results)
print()

print("==== Get memories ====")
memories = m.get(example_id)
print(memories)
print()

print("==== Update memories ====")
m.update(
    example_id,
    {
        "id": example_id,
        "memory": "User is Canadian.",
        "metadata": {"type": "opinion", "confidence": 85},
    },
)
print(m.get(example_id))
print()

print("==== Delete memories ====")
m.delete([example_id])
print(m.get_all())
print()

print("==== Delete all memories ====")
m.delete_all()
print(m.get_all())
print()
