from memos.configs.mem_chat import MemChatConfigFactory
from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.mem_chat.factory import MemChatFactory
from memos.mem_cube.general import GeneralMemCube


mem_chat_config = MemChatConfigFactory.model_validate(
    {
        "backend": "simple",
        "config": {
            "user_id": "user_123",
            "chat_llm": {
                "backend": "ollama",
                "config": {
                    "model_name_or_path": "qwen3:1.7b",
                    "temperature": 0.0,
                    "remove_think_prefix": True,
                    "max_tokens": 4096,
                },
            },
            "max_turns_window": 20,
            "top_k": 5,
            "enable_textual_memory": True,
            "enable_activation_memory": False,
            "enable_parametric_memory": False,
        },
    }
)
mem_chat = MemChatFactory.from_config(mem_chat_config)

# Initialize_memory_cube
config = GeneralMemCubeConfig.model_validate(
    {
        "user_id": "user03alice",
        "cube_id": "user03alice/mem_cube_tree",
        "text_mem": {
            "backend": "tree_text",
            "config": {
                "extractor_llm": {
                    "backend": "ollama",
                    "config": {
                        "model_name_or_path": "qwen3:1.7b",
                        "temperature": 0.0,
                        "remove_think_prefix": True,
                        "max_tokens": 8192,
                    },
                },
                "dispatcher_llm": {
                    "backend": "ollama",
                    "config": {
                        "model_name_or_path": "qwen3:1.7b",
                        "temperature": 0.0,
                        "remove_think_prefix": True,
                        "max_tokens": 8192,
                    },
                },
                "graph_db": {
                    "backend": "neo4j",
                    "config": {
                        "uri": "bolt://localhost:7687",
                        "user": "neo4j",
                        "password": "12345678",
                        "db_name": "user03alice_mem_cube_3",
                        "auto_create": True,
                    },
                },
                "embedder": {
                    "backend": "ollama",
                    "config": {
                        "model_name_or_path": "nomic-embed-text:latest",
                    },
                },
            },
        },
        "act_mem": {
            "backend": "kv_cache",
            "config": {
                "memory_filename": "activation_memory.pickle",
                "extractor_llm": {
                    "backend": "huggingface",
                    "config": {
                        "model_name_or_path": "Qwen/Qwen3-1.7B",
                        "temperature": 0.8,
                        "max_tokens": 1024,
                        "top_p": 0.9,
                        "top_k": 50,
                        "add_generation_prompt": True,
                        "remove_think_prefix": False,
                    },
                },
            },
        },
        "para_mem": {
            "backend": "lora",
            "config": {
                "memory_filename": "parametric_memory.adapter",
                "extractor_llm": {
                    "backend": "huggingface",
                    "config": {
                        "model_name_or_path": "Qwen/Qwen3-1.7B",
                        "temperature": 0.8,
                        "max_tokens": 1024,
                        "top_p": 0.9,
                        "top_k": 50,
                        "add_generation_prompt": True,
                        "remove_think_prefix": False,
                    },
                },
            },
        },
    },
)

# Initialize the MemCube with the configuration
mem_cube = GeneralMemCube(config)

# TODO: Read memory and prepare data
# Hope to read user docs and save in a file

# TODO: Organize MemoryCube
# Call Tree.add()
# save in memory cube

# chat and search and organize
mem_chat.mem_cube = mem_cube
mem_chat.run()
mem_chat.mem_cube.dump("new_cube_path")
