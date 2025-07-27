import uuid

from memos import settings
from memos.configs.vec_db import VectorDBConfigFactory
from memos.vec_dbs.factory import VecDBFactory


config = VectorDBConfigFactory.model_validate(
    {
        "backend": "qdrant",
        "config": {
            "collection_name": "test_collection",
            "vector_dimension": 4,
            "distance_metric": "cosine",
            "path": str(settings.MEMOS_DIR / "qdrant"),
        },
    }
)
vec_db = VecDBFactory.from_config(config)

# ============== BATCH OPERATIONS ==============
print("\n" + "=" * 50)
print("BATCH DOCUMENT ADDITION")

# Add batch data
batch_data = [
    {
        "id": str(uuid.uuid4()),
        "vector": [0.5, 0.6, 0.7, 0.8],
        "payload": {"text": "Document A", "category": "type1"},
    },
    {
        "id": str(uuid.uuid4()),
        "vector": [0.4, 0.5, 0.6, 0.7],
        "payload": {"text": "Document B", "category": "type2"},
    },
    {
        "id": str(uuid.uuid4()),
        "vector": [0.3, 0.4, 0.5, 0.6],
        "payload": {"text": "Document C", "category": "type1"},
    },
]
vec_db.add(batch_data)
print(f"✓ Added {len(batch_data)} documents")

# ============== SEARCH OPERATIONS ==============
print("\n" + "=" * 50)
print("VECTOR SEARCH")

# Search for similar items
query_vector = [5.0, 6.0, 7.0, 8.0]
results = vec_db.search(query_vector, top_k=2)
print(f"Query vector: {query_vector}")
print("\nResults:")
for i, result in enumerate(results, 1):
    print(f"  {i}. ID: {result.id}")
    print(f"     Score: {result.score}")
    print(f"     Payload: {result.payload}")

# ============== COUNT OPERATIONS ==============
print("\n" + "=" * 50)
print("DOCUMENT COUNT")

# Count documents in collection
count = vec_db.count()
print(f"Total documents in collection: {count}")

# Count documents with filter
filtered_count = vec_db.count(filter={"category": "type1"})
print(f"Documents with category 'type1': {filtered_count}")

# ============== SINGLE DOCUMENT OPERATIONS ==============
print("\n" + "=" * 50)
print("DOCUMENT OPERATIONS")

# Add a document
doc_id = str(uuid.uuid4())
vec_db.add(
    [
        {
            "id": doc_id,
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {"text": "Original document", "status": "new"},
        }
    ]
)
print(f"✓ Added document with ID: {doc_id}")

# Update document payload
vec_db.update(doc_id, {"payload": {"text": "Updated document", "status": "updated"}})
print(f"✓ Updated document payload for ID: {doc_id}")

# Retrieve updated document
result = vec_db.get_by_id(doc_id)
print("\nRetrieved updated document:")
print(f"  ID: {doc_id}")
print(f"  Payload: {result.payload if result else 'Not found'}")

# Delete the document
vec_db.delete([doc_id])
print(f"\n✓ Deleted document with ID: {doc_id}")

# Verify deletion
result = vec_db.get_by_id(doc_id)
print("\nDocument after deletion:")
print(f"  Result: {'Not found' if result is None else result}")

# ============== COLLECTION OPERATIONS ==============
print("\n" + "=" * 50)
print("COLLECTION OPERATIONS")

# List all collections in the database
collections = vec_db.list_collections()
print(f"Available collections: {collections}")

# ============== FILTER OPERATIONS ==============
print("\n" + "=" * 50)
print("FILTER OPERATIONS")

# Get documents by filter criteria
filter_results = vec_db.get_by_filter({"category": "type1"})
print("Documents filtered by category 'type1':")
for i, item in enumerate(filter_results, 1):
    print(f"  {i}. ID: {item.id}")
    print(f"     Payload: {item.payload}")

# Get all documents in the collection
all_docs = vec_db.get_all()
print("\nAll documents in the collection:")
for i, item in enumerate(all_docs, 1):
    print(f"  {i}. ID: {item.id}")
    print(f"     Vector: {item.vector}")
    print(f"     Payload: {item.payload}")

# ============== CLEANUP ==============
print("\n" + "=" * 50)
print("CLEANUP")

# Delete the collection
vec_db.delete_collection("test_collection")
print("✓ Collection deleted")
print(f"Available collections after deletion: {vec_db.list_collections()}")
print("\n" + "=" * 50)
