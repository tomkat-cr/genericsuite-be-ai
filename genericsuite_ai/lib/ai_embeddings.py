"""
Embeddings engine
"""
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import ClarifaiEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from genericsuite.util.app_context import AppContext
from genericsuite.util.utilities import log_debug

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.clarifai import (
    get_model_config,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities

DEBUG = False


def get_embeddings_engine(
    app_context: AppContext,
    embedding_engine: str = None
) -> dict:
    """
    Get the embeddings engine.

    Returns:
        Embeddings: The embeddings engine.
    """
    settings = Config(app_context)
    response = {
        "engine": None,
        "error": None,
    }
    billing = BillingUtilities(app_context)
    openai_api_key = billing.get_openai_api_key()
    if billing.is_free_plan():
        if not openai_api_key:
            response["error"] = \
                "OpenAI API key is not configured [AI-GEE-E010]"
        response["engine"] = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=settings.OPENAI_EMBEDDINGS_MODEL,
        )
        return response
    if not embedding_engine:
        embedding_engine: str = settings.EMBEDDINGS_ENGINE
    if DEBUG:
        log_debug("GET_EMBEDDINGS_ENGINE |" +
                  f" embedding_engine: {embedding_engine}")

    if embedding_engine == "huggingface":
        # https://python.langchain.com/docs/integrations/platforms/huggingface
        # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
        response["engine"] = HuggingFaceEmbeddings(
            # api_key=settings.HUGGINGFACE_API_KEY,
            model_name=settings.HUGGINGFACE_EMBEDDINGS_MODEL,
            model_kwargs=settings.HUGGINGFACE_EMBEDDINGS_MODEL_KWARGS,
            encode_kwargs=settings.HUGGINGFACE_EMBEDDINGS_ENCODE_KWARGS,
        )
    elif embedding_engine == "clarifai":
        # https://python.langchain.com/docs/integrations/text_embedding/clarifai
        model_config = \
            get_model_config(settings.AI_CLARIFAI_DEFAULT_TEXT_EMBEDDING_MODEL,
                             include_all=False)
        if DEBUG:
            log_debug("GET_EMBEDDINGS_ENGINE | Clarifai /" +
                      f" model_config: {model_config}")
        if "error" in model_config:
            response["error"] = model_config["error"]
        else:
            response["engine"] = ClarifaiEmbeddings(
                pat=settings.CLARIFAI_PAT,
                **model_config,
            )
    elif embedding_engine == "bedrock":
        # https://python.langchain.com/docs/integrations/text_embedding/bedrock
        model_config = {}
        model_config["region_name"] = settings.AWS_REGION
        model_config["model_id"] = settings.AWS_BEDROCK_EMBEDDINGS_MODEL_ID
        if settings.AWS_BEDROCK_EMBEDDINGS_PROFILE:
            model_config["credentials_profile_name"] = \
                settings.AWS_BEDROCK_EMBEDDINGS_PROFILE
        response["engine"] = BedrockEmbeddings(**model_config)

    elif embedding_engine == "cohere":
        # https://python.langchain.com/docs/integrations/text_embedding/cohere
        response["engine"] = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_EMBEDDINGS_MODEL,
        )

    elif embedding_engine == "ollama":
        # https://python.langchain.com/docs/integrations/text_embedding/ollama
        response["engine"] = OllamaEmbeddings(
            # https://ollama.ai/library
            model=settings.OLLAMA_EMBEDDINGS_MODEL,
        )

    elif embedding_engine == "openai":
        # https://python.langchain.com/docs/integrations/text_embedding/openai
        response["engine"] = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=settings.OPENAI_EMBEDDINGS_MODEL,
        )

    log_debug("GET_EMBEDDINGS_ENGINE |" +
              f" response: {response}")
    return response


def messages_to_embeddings(
    app_context: AppContext,
    conversation: list
) -> List[List[float]]:
    """
    Convert the conversation messages into embeddings using the embeddings
    object.

    Args:
        messages (list): The conversation messages in OpenAI format.
        e.g.
        [
            {"role": "system", "content": "You have access to the following "
                                          "tools: Search, Calculator"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Let me search for that..."}
        ]

    Returns:
        list: The embeddings of the conversation messages.
    """
    embeddings = get_embeddings_engine(app_context)
    if "error" in embeddings:
        raise Exception(embeddings["error"])
    return embeddings["engine"].embed_documents(
        [f'{message["role"]}: {message["content"]}'
         for message in conversation]
    )
