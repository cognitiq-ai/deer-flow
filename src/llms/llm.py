# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Any, Dict, List, get_args

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings

from src.config import load_yaml_config
from src.config.agents import LLMType
from src.llms.providers.dashscope import ChatDashscope

# Cache for LLM instances
_llm_cache: dict[LLMType, BaseChatModel] = {}

# Cache for embedding instance
_embedding_cache: OpenAIEmbeddings | GoogleGenerativeAIEmbeddings | None = None


def _get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())


def _get_llm_type_config_keys() -> dict[str, str]:
    """Get mapping of LLM types to their configuration keys."""
    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
        "code": "CODE_MODEL",
    }


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    Environment variables should follow the format: {LLM_TYPE}__{KEY}
    e.g., BASIC_MODEL__api_key, BASIC_MODEL__base_url
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value
    return conf


def _get_env_embedding_conf() -> Dict[str, Any]:
    """
    Get embedding configuration from environment variables.
    Environment variables should follow the format: EMBEDDING_MODEL__{KEY}
    e.g., EMBEDDING_MODEL__api_key, EMBEDDING_MODEL__base_url
    """
    prefix = "EMBEDDING_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value

    return conf


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> BaseChatModel:
    """Create LLM instance using configuration."""
    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    if not config_key:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    llm_conf = conf.get(config_key, {})
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM configuration for {llm_type}: {llm_conf}")

    # Get configuration from environment variables
    env_conf = _get_env_llm_conf(llm_type)

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**llm_conf, **env_conf}

    if not merged_conf:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Add max_retries to handle rate limit errors
    if "max_retries" not in merged_conf:
        merged_conf["max_retries"] = 3

    # Handle SSL verification settings
    verify_ssl = merged_conf.pop("verify_ssl", True)

    # Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        merged_conf["http_client"] = http_client
        merged_conf["http_async_client"] = http_async_client

    # Check if it's Google AI Studio platform based on configuration
    platform = merged_conf.get("platform", "").lower()
    is_google_aistudio = platform == "google_aistudio" or platform == "google-aistudio"

    if is_google_aistudio:
        # Handle Google AI Studio specific configuration
        gemini_conf = merged_conf.copy()

        # Map common keys to Google AI Studio specific keys
        if "api_key" in gemini_conf:
            gemini_conf["google_api_key"] = gemini_conf.pop("api_key")

        # Remove base_url and platform since Google AI Studio doesn't use them
        gemini_conf.pop("base_url", None)
        gemini_conf.pop("platform", None)

        # Remove unsupported parameters for Google AI Studio
        gemini_conf.pop("http_client", None)
        gemini_conf.pop("http_async_client", None)

        return ChatGoogleGenerativeAI(**gemini_conf)

    if "azure_endpoint" in merged_conf or os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(**merged_conf)

    # Check if base_url is dashscope endpoint
    if "base_url" in merged_conf and "dashscope." in merged_conf["base_url"]:
        if llm_type == "reasoning":
            merged_conf["extra_body"] = {"enable_thinking": True}
        else:
            merged_conf["extra_body"] = {"enable_thinking": False}
        return ChatDashscope(**merged_conf)

    if llm_type == "reasoning":
        merged_conf["api_base"] = merged_conf.pop("base_url", None)
        return ChatDeepSeek(**merged_conf)
    else:
        return ChatOpenAI(**merged_conf)


def _create_embedding_use_conf(
    conf: Dict[str, Any],
) -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings:
    """Create embedding instance using configuration."""
    config_key = "EMBEDDING_MODEL"
    embedding_conf = conf.get(config_key, {})
    if not isinstance(embedding_conf, dict):
        raise ValueError(
            f"Invalid embedding configuration for {config_key}: {embedding_conf}"
        )

    # Get configuration from environment variables
    env_conf = _get_env_embedding_conf()

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**embedding_conf, **env_conf}

    if not merged_conf:
        raise ValueError(f"No configuration found for {config_key}")

    # Check if it's Google AI Studio platform based on configuration
    platform = merged_conf.get("platform", "").lower()
    is_google_aistudio = platform == "google_aistudio" or platform == "google-aistudio"

    if is_google_aistudio:
        # Handle Google AI Studio specific configuration
        gemini_conf = merged_conf.copy()

        # Map common keys to Google AI Studio specific keys
        if "api_key" in gemini_conf:
            gemini_conf["google_api_key"] = gemini_conf.pop("api_key")

        # Remove base_url and platform since Google AI Studio doesn't use them
        gemini_conf.pop("base_url", None)
        gemini_conf.pop("platform", None)

        # Remove unsupported parameters for Google AI Studio
        gemini_conf.pop("http_client", None)
        gemini_conf.pop("http_async_client", None)

        return GoogleGenerativeAIEmbeddings(**gemini_conf)

    else:
        return OpenAIEmbeddings(**merged_conf)


def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(_get_config_file_path())
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


def get_embedding_model() -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings:
    """
    Get embedding instance. Returns cached instance if available.
    """
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    conf = load_yaml_config(_get_config_file_path())
    embedding = _create_embedding_use_conf(conf)
    _embedding_cache = embedding
    return embedding


def get_embedding_dimension() -> int:
    """Get the embedding dimension for the configured model."""
    # Load configuration to get model details
    conf = load_yaml_config(_get_config_file_path())

    embedding_conf = conf.get("EMBEDDING_MODEL", {})
    model_name = embedding_conf.get("model", "")
    platform = embedding_conf.get("platform", "").lower()

    # Common embedding model dimensions
    model_dimensions = {
        # OpenAI models
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        # Google Gemini models
        "gemini-embedding-001": 3072,
    }

    # Check if we know the dimension for this model
    if model_name in model_dimensions:
        return model_dimensions[model_name]
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")


def get_configured_llm_models() -> dict[str, list[str]]:
    """
    Get all configured LLM models grouped by type.

    Returns:
        Dictionary mapping LLM type to list of configured model names.
    """
    try:
        conf = load_yaml_config(_get_config_file_path())
        llm_type_config_keys = _get_llm_type_config_keys()

        configured_models: dict[str, list[str]] = {}

        for llm_type in get_args(LLMType):
            # Get configuration from YAML file
            config_key = llm_type_config_keys.get(llm_type, "")
            yaml_conf = conf.get(config_key, {}) if config_key else {}

            # Get configuration from environment variables
            env_conf = _get_env_llm_conf(llm_type)

            # Merge configurations, with environment variables taking precedence
            merged_conf = {**yaml_conf, **env_conf}

            # Check if model is configured
            model_name = merged_conf.get("model")
            if model_name:
                configured_models.setdefault(llm_type, []).append(model_name)

        return configured_models

    except Exception as e:
        # Log error and return empty dict to avoid breaking the application
        print(f"Warning: Failed to load LLM configuration: {e}")
        return {}


def get_configured_embedding_models() -> dict[str, list[str]]:
    """
    Get configured embedding model.

    Returns:
        Dictionary with embedding configuration if available.
    """
    try:
        conf = load_yaml_config(_get_config_file_path())
        config_key = "EMBEDDING_MODEL"

        # Get configuration from YAML file
        yaml_conf = conf.get(config_key, {})

        # Get configuration from environment variables
        env_conf = _get_env_embedding_conf()

        # Merge configurations, with environment variables taking precedence
        merged_conf = {**yaml_conf, **env_conf}

        configured_models: dict[str, list[str]] = {}

        # Check if model is configured
        model_name = merged_conf.get("model")
        if model_name:
            configured_models["embedding"] = [model_name]

        return configured_models

    except Exception as e:
        # Log error and return empty dict to avoid breaking the application
        print(f"Warning: Failed to load embedding configuration: {e}")
        return {}


# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")


async def generate_embedding(
    text: str, provider: str = "auto", normalize: bool = True
) -> List[float] | List[List[float]]:
    """
    Generate an embedding vector for the given text.

    Args:
        text: The text to generate embeddings for
        provider: Kept for backwards compatibility, provider is auto-detected
        normalize: Whether to normalize the embedding vector

    Returns:
        A list of floats representing the embedding vector

    Raises:
        Exception: If embedding generation fails
    """
    if not text or not text.strip():
        raise Exception("Cannot generate embedding for empty text")

    try:
        embedding_model = get_embedding_model()

        # Generate embedding
        embeddings = await embedding_model.aembed_documents([text.strip()])

        if not embeddings or len(embeddings) == 0:
            raise Exception("Failed to generate embedding - empty result")

        embedding = embeddings[0]

        # Normalize if requested
        if normalize:
            # Calculate magnitude
            magnitude = sum(x * x for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]

        return embedding

    except Exception as e:
        raise Exception(f"Failed to generate embedding: {str(e)}")
