import os

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS


# init MOSConfig by deafult user
# note kvcache must at chatllm backend by huggingface
# gpu need set
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
user_id = "root"
cube_id = "root/mem_cube_kv_cache"
tmp_cube_path = "/tmp/default/mem_cube_5"

mos_config = MOSConfig.from_json_file("examples/data/config/simple_treekvcache_memos_config.json")
mos = MOS(mos_config)


# 2. Initialize_memory_cube
cube_config = GeneralMemCubeConfig.from_json_file(
    "examples/data/config/simple_treekvcache_cube_config.json"
)

# 3. Initialize the MemCube with the configuration and dump cube
mem_cube = GeneralMemCube(cube_config)
try:
    mem_cube.dump(tmp_cube_path)
except Exception as e:
    print(e)

# 4. Register the MemCube explicitly
mos.register_mem_cube(tmp_cube_path, mem_cube_id=cube_id, user_id=user_id)

# 5. Extract kv memory and add kv cache_mem
extract_kvmem = mos.mem_cubes[cube_id].act_mem.extract("I like football")
mos.mem_cubes[cube_id].act_mem.add([extract_kvmem])

# .chat
while True:
    user_input = input("👤 [You] ").strip()
    print()
    response = mos.chat(user_input)
    print(f"🤖 [Assistant] {response}\n")
print("📢 [System] MemChat has stopped.")
