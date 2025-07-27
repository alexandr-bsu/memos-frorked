import time

from memos.configs.llm import LLMConfigFactory
from memos.llms.factory import LLMFactory
from memos.memories.textual.tree_text_memory.retrieve.task_goal_parser import TaskGoalParser


# Step 1: Load LLM config and instantiate
config = LLMConfigFactory.model_validate(
    {
        "backend": "ollama",
        "config": {
            "model_name_or_path": "qwen3:0.6b",
            "temperature": 0.7,
            "max_tokens": 1024,
            "remove_think_prefix": True,
        },
    }
)
llm = LLMFactory.from_config(config)

# Task input
task = "When did Caroline go to the LGBTQ support group?"

parser = TaskGoalParser(llm, mode="fast")

time_init = time.time()
# Parse task goal
result = parser.parse(task)

# Print output
print("=== Parsed Result ===")
print("memories:", result.memories)
print("keys:", result.keys)
print("tags:", result.tags)
print("goal_type:", result.goal_type)
print("time:", time.time() - time_init)

parser = TaskGoalParser(llm, mode="fine")

time_init = time.time()

# Parse task goal
result = parser.parse(task)

# Print output
print("=== Parsed Result ===")
print("memories:", result.memories)
print("keys:", result.keys)
print("tags:", result.tags)
print("goal_type:", result.goal_type)
print("time:", time.time() - time_init)
