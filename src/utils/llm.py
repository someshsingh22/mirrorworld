"""Utilities for LLM initialization and configuration."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
# Try to find .env file automatically, fallback to explicit path
dotenv_path = find_dotenv()
if not dotenv_path:
    dotenv_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


def create_azure_llm(
    model: str = "gpt-4o",
    temperature: float = 0.7,
    deployment_name: str = None,
) -> AzureChatOpenAI:
    """Initialize Azure OpenAI client.

    Args:
        model: Model name (default: gpt-4o)
        temperature: Sampling temperature (default: 0.7)
        deployment_name: Azure deployment name (optional, defaults to model name)

    Returns:
        Configured AzureChatOpenAI instance

    Raises:
        AssertionError: If required environment variables are not set
    """
    required_env_vars = [
        "AZURE_API_BASE",
        "AZURE_API_KEY",
        "AZURE_API_VERSION",
    ]

    for var in required_env_vars:
        assert var in os.environ, f"Missing required environment variable: {var}"

    # Use deployment_name if provided, otherwise use model name
    deployment = deployment_name or model

    return AzureChatOpenAI(
        model=model,
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
        azure_deployment=deployment,
        temperature=temperature,
    )
