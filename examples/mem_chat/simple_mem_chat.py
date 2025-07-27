from memos.configs.mem_chat import MemChatConfigFactory
from memos.mem_chat.factory import MemChatFactory
from memos.mem_cube.general import GeneralMemCube


config = MemChatConfigFactory.model_validate(
    {
        "backend": "simple",
        "config": {
            "user_id": "user_123",
            "chat_llm": {
                "backend": "huggingface",
                "config": {
                    "model_name_or_path": "Qwen/Qwen3-1.7B",
                    "temperature": 0.1,
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
mem_chat = MemChatFactory.from_config(config)
mem_chat.mem_cube = GeneralMemCube.init_from_dir("examples/data/mem_cube_2")

mem_chat.run()

mem_chat.mem_cube.dump("tmp/mem_cube")
