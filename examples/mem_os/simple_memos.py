from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS


# init MOSConfig
mos_config = MOSConfig.from_json_file("examples/data/config/simple_memos_config.json")
mos = MOS(mos_config)

# create user id
user_id = "lcy1"
mos.create_user(user_id=user_id)
users = mos.list_users()
print("\nAll users:")
for user in users:
    print(f"  - {user['user_name']} ({user['user_id']}) - Role: {user['role']}")


# load exist mem_cube from local
mos.register_mem_cube("examples/data/mem_cube_2", user_id=user_id)

mos.add(memory_content="I like playing football.", user_id=user_id)

get_all_results = mos.get_all(user_id=user_id)
print(f"Get all results for user : {get_all_results}")

get_results = mos.get(
    mem_cube_id="examples/data/mem_cube_2",
    memory_id=get_all_results["text_mem"][0]["memories"][0].id,
    user_id=user_id,
)
print(f"Get memories for user : {get_results}")

search_results = mos.search(query="my favorite football game", user_id=user_id)
print(f"Search results for user : {search_results}")


while True:
    user_input = input("👤 [You] ").strip()
    print()
    response = mos.chat(user_input, user_id=user_id)
    print(f"🤖 [Assistant] {response}\n")
print("📢 [System] MemChat has stopped.")
