import uuid

import pytest

from pydantic import ValidationError

from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata


class TestTextualMemoryMetadata:
    def test_basic_init_and_defaults(self):
        metadata = TextualMemoryMetadata()
        assert metadata.type is None
        assert metadata.updated_at is not None

    def test_full_init(self):
        item = TextualMemoryMetadata.model_validate(
            {
                "type": "opinion",
                "memory_time": "2025-05-24",
                "source": "conversation",
                "confidence": 100.0,
                "entities": ["rainy days", "the one I love"],
                "tags": ["preferences", "opinions"],
                "visibility": "session",
                "updated_at": "2025-05-24T02:10:16.190683",
            }
        )
        assert item.type == "opinion"
        assert item.memory_time == "2025-05-24"
        assert item.source == "conversation"
        assert item.confidence == 100.0
        assert item.entities == ["rainy days", "the one I love"]
        assert item.tags == ["preferences", "opinions"]
        assert item.visibility == "session"
        assert item.updated_at == "2025-05-24T02:10:16.190683"

    def test_valid_and_invalid_confidence(self):
        assert TextualMemoryMetadata(confidence=85.0).confidence == 85.0
        with pytest.raises(ValidationError):
            TextualMemoryMetadata(confidence=150)

    def test_valid_and_invalid_memory_time(self):
        TextualMemoryMetadata(memory_time="2025-05-24")  # Should pass
        with pytest.raises(ValidationError):
            TextualMemoryMetadata(memory_time="5-24-2025")  # Wrong format

    def test_enum_validation(self):
        TextualMemoryMetadata(type="fact", source="conversation", visibility="private")
        with pytest.raises(ValidationError):
            TextualMemoryMetadata(type="unknown")


class TestTextualMemoryItem:
    def test_full_init(self):
        item = TextualMemoryItem(
            id=str(uuid.uuid4()), memory="test", metadata={"type": "event", "confidence": 90.0}
        )
        assert item.id is not None
        assert item.memory == "test"
        assert isinstance(item.metadata, TextualMemoryMetadata)

    def test_required_fields_and_defaults(self):
        item = TextualMemoryItem(memory="test")
        assert item.memory == "test"
        assert item.id is not None
        assert isinstance(item.metadata, TextualMemoryMetadata)

    def test_id_and_metadata_validation(self):
        valid_id = str(uuid.uuid4())
        metadata = {"type": "event", "confidence": 90.0}
        item = TextualMemoryItem(id=valid_id, memory="test", metadata=metadata)
        assert item.id == valid_id
        assert item.metadata.type == "event"

        with pytest.raises(ValidationError):
            TextualMemoryItem(id="bad-uuid", memory="test")

    def test_dict_conversion(self):
        item = TextualMemoryItem(memory="test")
        as_dict = item.to_dict()
        assert "memory" in as_dict
        reconstructed = TextualMemoryItem.from_dict(as_dict)
        assert reconstructed.memory == item.memory
