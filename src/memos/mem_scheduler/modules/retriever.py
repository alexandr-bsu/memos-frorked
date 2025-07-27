import logging

from memos.configs.mem_scheduler import BaseSchedulerConfig
from memos.dependency import require_python_package
from memos.llms.base import BaseLLM
from memos.log import get_logger
from memos.mem_cube.general import GeneralMemCube
from memos.mem_scheduler.modules.base import BaseSchedulerModule
from memos.mem_scheduler.modules.schemas import (
    TreeTextMemory_SEARCH_METHOD,
)
from memos.mem_scheduler.utils import (
    extract_json_dict,
    is_all_chinese,
    is_all_english,
    transform_name_to_key,
)
from memos.memories.textual.tree import TextualMemoryItem, TreeTextMemory


logger = get_logger(__name__)


class SchedulerRetriever(BaseSchedulerModule):
    def __init__(self, process_llm: BaseLLM, config: BaseSchedulerConfig):
        super().__init__()

        self.config: BaseSchedulerConfig = config
        self.process_llm = process_llm

        # hyper-parameters
        self.filter_similarity_threshold = 0.75
        self.filter_min_length_threshold = 6

        # log function callbacks
        self.log_working_memory_replacement = None

    def search(
        self, query: str, mem_cube: GeneralMemCube, top_k: int, method=TreeTextMemory_SEARCH_METHOD
    ):
        """Search in text memory with the given query.

        Args:
            query: The search query string
            top_k: Number of top results to return
            method: Search method to use

        Returns:
            Search results or None if not implemented
        """
        text_mem_base = mem_cube.text_mem
        try:
            if method == TreeTextMemory_SEARCH_METHOD:
                assert isinstance(text_mem_base, TreeTextMemory)
                results_long_term = text_mem_base.search(
                    query=query, top_k=top_k, memory_type="LongTermMemory"
                )
                results_user = text_mem_base.search(
                    query=query, top_k=top_k, memory_type="UserMemory"
                )
                results = results_long_term + results_user
            else:
                raise NotImplementedError(str(type(text_mem_base)))
        except Exception as e:
            logger.error(f"Fail to search. The exeption is {e}.", exc_info=True)
            results = []
        return results

    @require_python_package(
        import_name="sklearn",
        install_command="pip install scikit-learn",
        install_link="https://scikit-learn.org/stable/install.html",
    )
    def filter_similar_memories(
        self, text_memories: list[str], similarity_threshold: float = 0.75
    ) -> list[str]:
        """
        Filters out low-quality or duplicate memories based on text similarity.

        Args:
            text_memories: List of text memories to filter
            similarity_threshold: Threshold for considering memories duplicates (0.0-1.0)
                                Higher values mean stricter filtering

        Returns:
            List of filtered memories with duplicates removed
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not text_memories:
            logging.warning("Received empty memories list - nothing to filter")
            return []

        for idx in range(len(text_memories)):
            if not isinstance(text_memories[idx], str):
                logger.error(
                    f"{text_memories[idx]} in memories is not a string,"
                    f" and now has been transformed to be a string."
                )
                text_memories[idx] = str(text_memories[idx])

        try:
            # Step 1: Vectorize texts using TF-IDF
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(text_memories)

            # Step 2: Calculate pairwise similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Step 3: Identify duplicates
            to_keep = []
            removal_reasons = {}

            for current_idx in range(len(text_memories)):
                is_duplicate = False

                # Compare with already kept memories
                for kept_idx in to_keep:
                    similarity_score = similarity_matrix[current_idx, kept_idx]

                    if similarity_score > similarity_threshold:
                        is_duplicate = True
                        # Generate removal reason with sample text
                        removal_reasons[current_idx] = (
                            f"Memory too similar (score: {similarity_score:.2f}) to kept memory #{kept_idx}. "
                            f"Kept: '{text_memories[kept_idx][:100]}...' | "
                            f"Removed: '{text_memories[current_idx][:100]}...'"
                        )
                        logger.info(removal_reasons)
                        break

                if not is_duplicate:
                    to_keep.append(current_idx)

            # Return filtered memories
            return [text_memories[i] for i in sorted(to_keep)]

        except Exception as e:
            logging.error(f"Error filtering memories: {e!s}")
            return text_memories  # Return original list if error occurs

    def filter_too_short_memories(
        self, text_memories: list[str], min_length_threshold: int = 20
    ) -> list[str]:
        """
        Filters out text memories that fall below the minimum length requirement.
        Handles both English (word count) and Chinese (character count) differently.

        Args:
            text_memories: List of text memories to be filtered
            min_length_threshold: Minimum length required to keep a memory.
                                For English: word count, for Chinese: character count.

        Returns:
            List of filtered memories meeting the length requirement
        """
        if not text_memories:
            logging.debug("Empty memories list received in short memory filter")
            return []

        filtered_memories = []
        removed_count = 0

        for memory in text_memories:
            stripped_memory = memory.strip()
            if not stripped_memory:  # Skip empty/whitespace memories
                removed_count += 1
                continue

            # Determine measurement method based on language
            if is_all_english(stripped_memory):
                length = len(stripped_memory.split())  # Word count for English
            elif is_all_chinese(stripped_memory):
                length = len(stripped_memory)  # Character count for Chinese
            else:
                logger.debug(
                    f"Mixed-language memory, using character count: {stripped_memory[:50]}..."
                )
                length = len(stripped_memory)  # Default to character count

            if length >= min_length_threshold:
                filtered_memories.append(memory)
            else:
                removed_count += 1

        if removed_count > 0:
            logger.info(
                f"Filtered out {removed_count} short memories "
                f"(below {min_length_threshold} units). "
                f"Total remaining: {len(filtered_memories)}"
            )

        return filtered_memories

    def replace_working_memory(
        self,
        queries: list[str],
        user_id: str,
        mem_cube_id: str,
        mem_cube: GeneralMemCube,
        original_memory: list[TextualMemoryItem],
        new_memory: list[TextualMemoryItem],
        top_k: int = 10,
    ) -> None | list[TextualMemoryItem]:
        """Replace working memory with new memories after reranking."""
        memories_with_new_order = None
        text_mem_base = mem_cube.text_mem
        if isinstance(text_mem_base, TreeTextMemory):
            text_mem_base: TreeTextMemory = text_mem_base
            combined_memory = original_memory + new_memory
            memory_map = {
                transform_name_to_key(name=mem_obj.memory): mem_obj for mem_obj in combined_memory
            }
            combined_text_memory = [transform_name_to_key(name=m.memory) for m in combined_memory]

            # apply filters
            filtered_combined_text_memory = self.filter_similar_memories(
                text_memories=combined_text_memory,
                similarity_threshold=self.filter_similarity_threshold,
            )

            filtered_combined_text_memory = self.filter_too_short_memories(
                text_memories=filtered_combined_text_memory,
                min_length_threshold=self.filter_min_length_threshold,
            )

            unique_memory = list(dict.fromkeys(filtered_combined_text_memory))

            try:
                prompt = self.build_prompt(
                    "memory_reranking",
                    queries=queries,
                    current_order=unique_memory,
                    staging_buffer=[],
                )
                response = self.process_llm.generate([{"role": "user", "content": prompt}])
                response = extract_json_dict(response)
                text_memories_with_new_order = response.get("new_order", [])[:top_k]
            except Exception as e:
                logger.error(f"Fail to rerank with LLM, Exeption: {e}.", exc_info=True)
                text_memories_with_new_order = unique_memory[:top_k]

            memories_with_new_order = []
            for text in text_memories_with_new_order:
                normalized_text = transform_name_to_key(name=text)
                if text in memory_map:
                    memories_with_new_order.append(memory_map[normalized_text])
                else:
                    logger.warning(
                        f"Memory text not found in memory map. text: {text}; keys of memory_map: {memory_map.keys()}"
                    )

            text_mem_base.replace_working_memory(memories_with_new_order)
            logger.info(
                f"The working memory has been replaced with {len(memories_with_new_order)} new memories."
            )
            self.log_working_memory_replacement(
                original_memory=original_memory,
                new_memory=memories_with_new_order,
                user_id=user_id,
                mem_cube_id=mem_cube_id,
                mem_cube=mem_cube,
            )
        else:
            logger.error("memory_base is not supported")

        return memories_with_new_order
