# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict, List
import os
import httpx

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from typing import get_args

from src.config import load_yaml_config
from src.config.agents import LLMType

# Cache for LLM instances
_llm_cache: dict[LLMType, BaseChatModel] = {}


def _get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())


def _get_llm_type_config_keys() -> dict[str, str]:
    """Get mapping of LLM types to their configuration keys."""
    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
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

    if llm_type == "reasoning":
        merged_conf["api_base"] = merged_conf.pop("base_url", None)

    # Handle SSL verification settings
    verify_ssl = merged_conf.pop("verify_ssl", True)

    # Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        merged_conf["http_client"] = http_client
        merged_conf["http_async_client"] = http_async_client

    if "azure_endpoint" in merged_conf or os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(**merged_conf)
    if llm_type == "reasoning":
        return ChatDeepSeek(**merged_conf)
    else:
        return ChatOpenAI(**merged_conf)


def get_llm_by_type(
    llm_type: LLMType,
) -> BaseChatModel:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(_get_config_file_path())
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


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


# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")


def get_embedding_model(
    provider: str = "openai",
) -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings:
    """
    Get a LangChain embedding model based on the specified provider.

    Args:
        provider: The embedding provider to use ("openai" or "anthropic")

    Returns:
        A configured LangChain embedding model

    Raises:
        EmbeddingClientError: If the specified provider is not available
    """
    conf = load_yaml_config(_get_config_file_path())
    config_key = "EMBEDDING_MODEL"
    embedding_conf = conf.get(config_key, {})
    if not isinstance(embedding_conf, dict):
        raise ValueError(f"Invalid configuration for {config_key}: {embedding_conf}")

    # Get configuration from environment variables
    env_conf = _get_env_llm_conf("embedding")

    # Merge configurations, with environment variables taking precedence
    conf = {**embedding_conf, **env_conf}

    if not conf:
        raise ValueError(f"No configuration found for {config_key}")

    # For better testability, check if the settings exist first
    openai_available = bool(conf.get("OPENAI_API_KEY", ""))
    gemini_available = bool(conf.get("GEMINI_API_KEY", ""))

    # Try preferred provider first
    if provider == "openai" and openai_available:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=conf.get("OPENAI_API_KEY", ""),
        )
    elif provider == "gemini" and gemini_available:
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="SEMANTIC_SIMILARITY",
            google_api_key=conf.get("GEMINI_API_KEY", ""),
        )

    raise Exception(
        "No embedding providers are available. Please configure the API keys in .env file."
    )


async def generate_embedding(
    text: str, provider: str = "gemini", normalize: bool = True
) -> List[float] | List[List[float]]:
    """
    Generate an embedding vector for the given text.

    Args:
        text: The text to generate embeddings for
        provider: The embedding provider to use
        normalize: Whether to normalize the embedding vector

    Returns:
        A list of floats representing the embedding vector

    Raises:
        EmbeddingClientError: If embedding generation fails
    """
    if not text or not text.strip():
        raise Exception("Cannot generate embedding for empty text")

    try:
        embedding_model = get_embedding_model(provider)

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
