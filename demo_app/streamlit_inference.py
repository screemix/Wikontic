from __future__ import annotations

from typing import Any

from streamlit_app_config import USE_ONTOLOGY


def extract_triplets_for_demo(
    inference: Any,
    *,
    text: str,
    sample_id: str,
    source_text_id: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if USE_ONTOLOGY:
        return inference.extract_triplets_with_ontology_filtering_and_add_to_db(
            text=text,
            sample_id=sample_id,
            source_text_id=source_text_id,
        )

    initial_triplets, final_triplets, filtered_triplets = (
        inference.extract_triplets_and_add_to_db(
            text=text,
            sample_id=sample_id,
            source_text_id=source_text_id,
        )
    )
    return initial_triplets, final_triplets, filtered_triplets, []
