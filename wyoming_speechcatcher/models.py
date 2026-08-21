from __future__ import annotations
from typing import Dict, List

# Short tag → full HuggingFace model ID (as used by espnet_model_zoo)
TAGS: Dict[str, str] = {
    "de_streaming_transformer_m": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_m_raw_de_bpe1024",
    "de_streaming_transformer_l": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_13k_train_size_l_raw_de_bpe1024",
    "de_streaming_transformer_xl": "speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    "en_streaming_transformer_m": "speechcatcher/wordcab_speechcatcher_english_espnet_streaming_transformer_35k_train_size_m_raw_en_bpe1024",
    "en_streaming_transformer_l": "speechcatcher/wordcab_speechcatcher_english_espnet_streaming_transformer_35k_train_size_l_raw_en_bpe1024",
    "es_streaming_transformer_m": "speechcatcher/wordcab_speechcatcher_spanish_espnet_streaming_transformer_35k_train_size_m_raw_es_bpe1024",
    "es_streaming_transformer_l": "speechcatcher/wordcab_speechcatcher_spanish_espnet_streaming_transformer_35k_train_size_l_raw_es_bpe1024",
}

# Derive language code from short tag (first 2 characters)
LANG_FOR_TAG: Dict[str, str] = {tag: tag[:2] for tag in TAGS}

# Language code → list of available short tags for that language
MODELS: Dict[str, List[str]] = {
    "de": [
        "de_streaming_transformer_m",
        "de_streaming_transformer_l",
        "de_streaming_transformer_xl",
    ],
    "en": [
        "en_streaming_transformer_m",
        "en_streaming_transformer_l",
    ],
    "es": [
        "es_streaming_transformer_m",
        "es_streaming_transformer_l",
    ],
}

def model_choices() -> List[str]:
    return list(TAGS.keys())

def language_choices() -> List[str]:
    return list(MODELS.keys())

def get_language_for_tag(tag: str) -> str:
    return LANG_FOR_TAG[tag]

def get_models_for_language(language: str) -> List[str]:
    return MODELS[language]

def get_full_model_id(tag: str) -> str:
    return TAGS[tag]
