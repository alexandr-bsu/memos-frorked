from memos.configs.memory import MemoryConfigFactory
from memos.memories.factory import MemoryFactory


# ===== Example: Use factory and HFLLM to build and manage KVCacheMemory =====

# 1. Create config for KVCacheMemory (using HuggingFace backend)
config = MemoryConfigFactory(
    backend="kv_cache",
    config={
        "extractor_llm": {
            "backend": "huggingface",
            "config": {
                "model_name_or_path": "Qwen/Qwen3-0.6B",  # Use a valid HuggingFace model name
                "max_tokens": 32,
                "add_generation_prompt": True,
                "remove_think_prefix": True,
            },
        },
    },
)

# 2. Instantiate KVCacheMemory using the factory
kv_mem = MemoryFactory.from_config(config)

# 3. Extract a KVCacheItem (DynamicCache) from a prompt (uses HFLLM.build_kv_cache internally)
prompt = [
    {"role": "user", "content": "What is MemOS?"},
    {"role": "assistant", "content": "MemOS is a memory operating system for LLMs."},
]
print("===== Extract KVCacheItem =====")
cache_item = kv_mem.extract(prompt)
print(cache_item)
print()

# 4. Add the extracted KVCacheItem
print("===== Add KVCacheItem =====")
kv_mem.add([cache_item])
print(kv_mem.get_all())
print()

# 5. Get by id
print("===== Get KVCacheItem by id =====")
retrieved = kv_mem.get(cache_item.id)
print(retrieved)
print()

# 6. Merge caches (simulate with two items)
print("===== Merge DynamicCache =====")
item2 = kv_mem.extract([{"role": "user", "content": "Tell me a joke."}])
kv_mem.add([item2])
merged_cache = kv_mem.get_cache([cache_item.id, item2.id])
print(merged_cache)
print()

# 7. Delete one
print("===== Delete one KVCacheItem =====")
kv_mem.delete([cache_item.id])
print(kv_mem.get_all())
print()

# 8. Dump and load
print("===== Dump and Load KVCacheMemory =====")
kv_mem.dump("tmp/kv_mem")
print("Memory dumped to 'tmp/kv_mem'.")
kv_mem.delete_all()
kv_mem.load("tmp/kv_mem")
print("Memory loaded from 'tmp/kv_mem':", kv_mem.get_all())
