
"""
Langchain models
"""
from typing import Union, Optional, Any
import json

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain_community.llms import Clarifai

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables.utils import ConfigurableField
from langchain_core.runnables.base import RunnableSerializable
from langchain_google_genai import ChatGoogleGenerativeAI

from genericsuite.util.app_context import AppContext
from genericsuite.util.app_logger import log_debug, log_error
from genericsuite.util.utilities import (
    is_under_test,
    get_default_resultset,
)

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.clarifai import (
    get_model_config,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities
from genericsuite_ai.lib.huggingface_endpoint import (
    GsHuggingFaceEndpoint,
)
from genericsuite_ai.lib.huggingface_chat_model import (
    GsChatHuggingFace,
)
from genericsuite_ai.lib.ibm import IbmWatsonx
from genericsuite_ai.lib.gcp import get_gcp_vertexai_credentials

DEBUG = True


class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler that can be used to handle callbacks from langchain.
    """

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        reason = kwargs.get("reason")
        if reason == "GUARDRAIL_INTERVENED":
            log_error(f"Guardrails: {kwargs}")


def get_system_msg_permitted(app_context: AppContext, model_name: str) -> bool:
    """
    Check if the model allows system messages
    """
    settings = Config(app_context)
    if settings.AI_MODEL_ALLOW_SYSTEM_MSG != "":
        return settings.AI_MODEL_ALLOW_SYSTEM_MSG == "1"
    return model_name not in [
        "o1-mini",
        "o1-preview",
    ]


def get_tools_permitted(app_context: AppContext, model_name: str) -> bool:
    """
    Check if the model allows Tools / GPT functions
    """
    settings = Config(app_context)
    if settings.AI_MODEL_ALLOW_TOOLS != "":
        return settings.AI_MODEL_ALLOW_TOOLS == "1"
    return model_name not in [
        "o1-mini",
        "o1-preview",
    ]


def get_need_preamble(app_context: AppContext, model_name: str) -> bool:
    """
    Check if the model needs another model as a preamble to handle tools
    and system message.
    """
    settings = Config(app_context)
    if settings.AI_MODEL_NEED_PREAMBLE != "":
        return settings.AI_MODEL_NEED_PREAMBLE == "1"
    return model_name in [
        "o1-mini",
        "o1-preview",
    ]


def get_preamble_model(app_context: AppContext, model_name: str) -> bool:
    settings = Config(app_context)
    preamble_model_configs: dict = \
        json.loads(settings.AI_PREAMBLE_MODEL_BASE_CONF)
    if settings.AI_PREAMBLE_MODEL_CUSTOM_CONF:
        preamble_model_configs.update(
            json.loads(settings.AI_PREAMBLE_MODEL_CUSTOM_CONF)
        )
    return preamble_model_configs.get(model_name, {
        "model_type": settings.AI_PREAMBLE_MODEL_DEFAULT_TYPE,
        "model_name": settings.AI_PREAMBLE_MODEL_DEFAULT_MODEL,
    })


def get_openai_api(model_params: dict) -> ChatOpenAI:
    """
    Get OpenAI API

    Args:
        model_config (dict): model configuration

    Returns:
        ChatOpenAI: OpenAI API object
    """
    response = get_default_resultset()
    openai_api_key = model_params.get("api_key")
    model_name = model_params.get("model_name")
    model_config = {
        "model": model_name,
        "openai_api_key": openai_api_key,
    }
    for key in ["base_url", "stop"]:
        if model_params.get(key):
            model_config[key] = model_params[key]
    for key in ["temperature"]:
        if model_params.get(key):
            model_config[key] = float(model_params[key])
    for key in ["top_p", "max_tokens"]:
        if model_params.get(key):
            model_config[key] = int(model_params[key])
    if model_params.get("streaming", "0") == "1":
        model_config["streaming"] = True
        model_config["n"] = 1
    model_object = ChatOpenAI(**model_config)
    if not model_object:
        response['error'] = True
        response['error_message'] = \
            "ERROR [GET_MODEL-OAI-020] - ChatOpenAI cannot" + \
            f" be initialized for {model_params.get('provider', 'N/A')}"
    else:
        response['model_object'] = model_object
    return response


def get_model(
    app_context: AppContext,
    model_type: str,
    model_params: Optional[dict] = None,
) -> dict:
    """
    Get model object.

    Args:
        app_context (AppContext): application context.
        model_type (str): the model type. e.g. "chat_openai",
            "gemini", "ollama", "anthropic", "clarifai"
        model_params (dict, optional): model parameters. Defaults to None.

    Returns:
        dict: a dictionary with the attributes:
            "model_type" (str): model type
            "model_object" (Any): model object
            "manufacturer" (str): manufacturer name
            "error" (str): error message
    """
    self_debug = DEBUG or is_under_test()
    settings = Config(app_context)
    if not model_params:
        model_params = {}
    model_object: Union[
        None,
        ChatOpenAI,
        ChatGoogleGenerativeAI,
        ChatOllama,
        ChatAnthropic,
        Clarifai,
    ] = None
    error = None
    manufacturer = None
    model_name = None
    tools_permitted = None
    need_preamble = None
    pref_agent_type = 'lcel'
    other_data = {}
    try:
        # OpenAI ChatGPT
        if model_type == "chat_openai" or model_type == "openai":
            # https://python.langchain.com/docs/integrations/chat/openai/
            other_data["user_plan"] = model_params.get(
                "user_plan",
                "Unknown or N/A")
            manufacturer = "OpenAI"
            model_name = model_params.get("model_name", settings.OPENAI_MODEL)
            openai_model = get_openai_api({
                "provider": manufacturer,
                "api_key": model_params.get(
                    "api_key", settings.OPENAI_API_KEY),
                "model_name": model_name,
                "temperature": settings.OPENAI_TEMPERATURE,
                "top_p": settings.OPENAI_TOP_P,
                "max_tokens": settings.OPENAI_MAX_TOKENS,
                "streaming": settings.AI_STREAMING,
            })
            if openai_model["error"]:
                error = openai_model["error_message"]
            else:
                model_object = openai_model["model_object"]

        # Google Gemini
        if model_type == "gemini" and settings.GOOGLE_API_KEY:
            # Google Generative AI Chatbot : Gemini
            # https://python.langchain.com/docs/integrations/chat/google_generative_ai
            manufacturer = "Google"
            model_name = settings.GOOGLE_MODEL
            model_object = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=float(settings.OPENAI_TEMPERATURE),
                google_api_key=settings.GOOGLE_API_KEY,
                convert_system_message_to_human=True,
            )

        # Google VertexAI
        if model_type == "vertexai":
            # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
            # https://python.langchain.com/api_reference/google_vertexai/index.html
            # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
            # https://console.cloud.google.com/vertex-ai/studio/freeform
            # https://cloud.google.com/docs/authentication/application-default-credentials#GAC
            # https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
            from langchain_google_vertexai import ChatVertexAI

            manufacturer = "Google Vertex AI"
            model_name = settings.VERTEXAI_MODEL
            model_config = {
                'model': model_name,
            }

            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                model_config["credentials"] = get_gcp_vertexai_credentials(
                    settings.GOOGLE_APPLICATION_CREDENTIALS)
            model_config["project"] = settings.GOOGLE_CLOUD_PROJECT
            model_config["location"] = settings.GOOGLE_CLOUD_LOCATION

            model_config["max_tokens"] = model_params.get(
                "max_tokens", settings.VERTEXAI_MAX_TOKENS)
            if model_config["max_tokens"]:
                model_config["max_tokens"] = int(model_config["max_tokens"])
            else:
                del model_config["max_tokens"]

            model_config["max_retries"] = model_params.get(
                "max_retries", settings.VERTEXAI_MAX_RETRIES)
            if model_config["max_retries"]:
                model_config["max_retries"] = int(model_config["max_retries"])
            else:
                del model_config["max_retries"]

            model_config["temperature"] = model_params.get(
                "temperature", settings.VERTEXAI_TEMPERATURE)
            if model_config["temperature"]:
                model_config["temperature"] = float(
                    model_config["temperature"])
            else:
                del model_config["temperature"]

            _ = DEBUG and log_debug(f"VERTEXAI | model_config: {model_config}")
            model_object = ChatVertexAI(**model_config)

        # Ollama
        if model_type == "ollama":
            # https://python.langchain.com/docs/integrations/chat/ollama/
            manufacturer = "Ollama"
            model_name = settings.OLLAMA_MODEL
            model_config = {
                'model': model_name,
                'temperature': float(settings.OLLAMA_TEMPERATURE),
            }
            if settings.OLLAMA_BASE_URL:
                model_config['base_url'] = settings.OLLAMA_BASE_URL
            model_object = ChatOllama(**model_config)

        # Genericsuite's Hugging Face lightweight Inference API
        if model_type == "huggingface_remote" or \
           model_type == "gs_huggingface":
            manufacturer = "GS Hugging Face"
            model_name = settings.HUGGINGFACE_DEFAULT_CHAT_MODEL
            if 'model_name' in model_params:
                model_name = model_params['model_name']
            model_object = GsHuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                do_sample=False,
                max_new_tokens=int(settings.HUGGINGFACE_MAX_NEW_TOKENS),
                top_k=int(settings.HUGGINGFACE_TOP_K),
                temperature=float(settings.HUGGINGFACE_TEMPERATURE),
                repetition_penalty=float(
                    settings.HUGGINGFACE_REPETITION_PENALTY),
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                timeout=float(settings.HUGGINGFACE_TIMEOUT),
            )
            # 0 = use HuggingFaceEndpoint + LangChain Agent
            # 1 = use ChatHuggingFace + LangChain LCEL
            other_data["HUGGINGFACE_USE_CHAT_HF"] = \
                settings.HUGGINGFACE_USE_CHAT_HF

            if settings.HUGGINGFACE_USE_CHAT_HF == "1":
                model_object = GsChatHuggingFace(
                    llm=model_object,
                    verbose=settings.HUGGINGFACE_VERBOSE == "1",
                )
            else:
                # Instruct models or pure LLMs Doesn't work with LCEL
                pref_agent_type = 'react_chat_agent'

        if model_type == "huggingface":
            # https://python.langchain.com/docs/integrations/platforms/huggingface/
            # https://python.langchain.com/docs/integrations/llms/huggingface_endpoint/
            # https://python.langchain.com/docs/integrations/chat/huggingface/
            #
            from langchain_huggingface \
                import HuggingFaceEndpoint  # type: ignore[import]
            from langchain_huggingface \
                import ChatHuggingFace  # type: ignore[import]
            #
            manufacturer = "Hugging Face"
            model_name = settings.HUGGINGFACE_DEFAULT_CHAT_MODEL
            # if 'url' in model_params:
            #     model_name = model_params['url']
            if 'model_name' in model_params:
                model_name = model_params['model_name']
            model_object = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                do_sample=False,
                max_new_tokens=int(settings.HUGGINGFACE_MAX_NEW_TOKENS),
                top_k=int(settings.HUGGINGFACE_TOP_K),
                temperature=float(settings.HUGGINGFACE_TEMPERATURE),
                repetition_penalty=float(
                    settings.HUGGINGFACE_REPETITION_PENALTY),
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                timeout=float(settings.HUGGINGFACE_TIMEOUT),
            )
            # 0 = use HuggingFaceEndpoint + LangChain Agent
            # 1 = use ChatHuggingFace + LangChain LCEL
            other_data["HUGGINGFACE_USE_CHAT_HF"] = \
                settings.HUGGINGFACE_USE_CHAT_HF

            if settings.HUGGINGFACE_USE_CHAT_HF == "1":
                model_object = ChatHuggingFace(
                    llm=model_object,
                    verbose=settings.HUGGINGFACE_VERBOSE == "1",
                )
            else:
                # Instruct models or pure LLMs Doesn't work with LCEL
                pref_agent_type = 'react_chat_agent'

        # Hugging Face Pipelines
        if model_type == "huggingface_pipeline":
            # https://python.langchain.com/v0.2/docs/integrations/llms/huggingface_pipelines/
            #
            from langchain_huggingface.llms import \
                HuggingFacePipeline  # type: ignore[import]
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline)  # type: ignore[import]
            from langchain_huggingface import \
                ChatHuggingFace  # type: ignore[import]
            #
            manufacturer = "Hugging Face (Pipeline)"
            model_name = settings.HUGGINGFACE_DEFAULT_CHAT_MODEL
            if 'model_name' in model_params:
                model_name = model_params['model_name']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model_config = {
                'model': model,
                'tokenizer': tokenizer,
                'max_new_tokens': int(settings.HUGGINGFACE_MAX_NEW_TOKENS),
                'huggingfacehub_api_token': settings.HUGGINGFACE_API_KEY,
            }
            if settings.HUGGINGFACE_PIPELINE_DEVICE != "":
                model_config["device"] = settings.HUGGINGFACE_PIPELINE_DEVICE
            # Pipeline() reference:
            # https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html
            pipe = pipeline("text-generation", **model_config)
            model_object = HuggingFacePipeline(
                pipeline=pipe,
                # timeout=float(settings.HUGGINGFACE_TIMEOUT),
            )
            other_data["HUGGINGFACE_USE_CHAT_HF"] = \
                settings.HUGGINGFACE_USE_CHAT_HF
            if settings.HUGGINGFACE_USE_CHAT_HF == "1":
                model_object = ChatHuggingFace(
                    llm=model_object,
                    verbose=settings.HUGGINGFACE_VERBOSE == "1",
                )
            else:
                # Instruct models or pure LLMs Doesn't work with LCEL
                pref_agent_type = 'react_chat_agent'

        # Anthropic Claude
        if model_type == "anthropic":
            # https://python.langchain.com/docs/integrations/chat/anthropic/
            manufacturer = "Anthropic"
            model_name = settings.ANTHROPIC_MODEL
            if not settings.ANTHROPIC_API_KEY:
                error = "ERROR [GET_MODEL-AT-010] - Missing ANTHROPIC_API_KEY"
            else:
                model_object = ChatAnthropic(
                    # model="claude-2"
                    model=model_name,
                    anthropic_api_key=settings.ANTHROPIC_API_KEY,
                )

        # Groq
        if model_type == "groq":
            # https://python.langchain.com/docs/integrations/chat/groq/
            manufacturer = "Groq"
            model_name = settings.GROQ_MODEL
            if 'model_name' in model_params:
                model_name = model_params['model_name']
            if not settings.GROQ_API_KEY:
                error = "ERROR [GET_MODEL-GROQ-010] - Missing GROQ_API_KEY"
            else:
                model_object = ChatGroq(
                    model=model_name,
                    api_key=settings.GROQ_API_KEY,
                    temperature=float(settings.GROQ_TEMPERATURE),
                    max_tokens=int(settings.GROQ_MAX_TOKENS)
                    if settings.GROQ_MAX_TOKENS else None,
                    timeout=float(settings.GROQ_TIMEOUT)
                    if settings.GROQ_TIMEOUT else None,
                    max_retries=int(settings.GROQ_MAX_RETRIES),
                )

        # Amazon Bedrock
        if model_type == "bedrock":
            # https://python.langchain.com/docs/integrations/llms/bedrock/
            # https://python.langchain.com/docs/how_to/response_metadata/#bedrock-anthropic
            manufacturer = "AWS"
            model_name = settings.AWS_BEDROCK_MODEL_ID
            if 'model_name' in model_params:
                model_name = model_params['model_name']
            model_config = {}
            model_config["model_id"] = model_name
            if settings.AWS_BEDROCK_CREDENTIALS_PROFILE:
                model_config["credentials_profile_name"] = \
                    settings.AWS_BEDROCK_CREDENTIALS_PROFILE
            if settings.AWS_BEDROCK_GUARDRAIL_ID:
                model_config["callbacks"] = [BedrockAsyncCallbackHandler()]
                model_config["guardrails"] = {
                    "id": settings.AWS_BEDROCK_GUARDRAIL_ID,
                    "version": settings.AWS_BEDROCK_GUARDRAIL_VERSION,
                    "trace": settings.AWS_BEDROCK_GUARDRAIL_TRACE == "1",
                }
            model_object = ChatBedrock(
                **model_config,
            )

        # Clarifai Platform
        if model_type == "clarifai":
            if not settings.CLARIFAI_PAT:
                error = "ERROR [GET_MODEL-CF-010] - Missing CLARIFAI_PAT"
            else:
                # https://python.langchain.com/docs/integrations/providers/clarifai#llms
                model_name = settings.AI_CLARIFAI_DEFAULT_CHAT_MODEL
                all_model_config = \
                    get_model_config(model_name)
                if "error" in all_model_config:
                    error = f"ERROR [GET_MODEL-CF-020] - {all_model_config}"
                else:
                    manufacturer = all_model_config["manufacturer"]
                    model_config = {}
                    # The "protected_namespaces" entry was added to fix the
                    # runtime warning "UserWarning: Field "model_id" has
                    # conflict with protected namespace "model_"
                    model_config['protected_namespaces'] = ()
                    model_config["user_id"] = all_model_config["user_id"]
                    model_config["app_id"] = all_model_config["app_id"]
                    model_config["model_id"] = all_model_config["model_id"]
                    model_config["model_version_id"] = \
                        all_model_config["model_version_id"]
                    model_object = Clarifai(
                        pat=settings.CLARIFAI_PAT,
                        **model_config,
                    )

        # AI/ML API
        if model_type == "aimlapi":
            # https://lablab.ai/blog/how-to-access-o1-models
            # https://python.langchain.com/api_reference/openai/llms/langchain_openai.llms.base.OpenAI.html
            # https://docs.aimlapi.com/api-overview/model-database/text-models
            manufacturer = "AI/ML API"
            model_name = model_params.get(
                "model_name", settings.AIMLAPI_MODEL_NAME)
            openai_model = get_openai_api({
                "provider": manufacturer,
                "base_url": settings.AIMLAPI_BASE_URL,
                "api_key": model_params.get(
                    "api_key", settings.AIMLAPI_API_KEY),
                "model_name": model_name,
                "temperature": settings.AIMLAPI_TEMPERATURE,
                "top_p": settings.AIMLAPI_TOP_P,
                "max_tokens": settings.AIMLAPI_MAX_TOKENS,
                "streaming": settings.AI_STREAMING,
            })
            if openai_model["error"]:
                error = openai_model["error_message"]
            else:
                model_object = openai_model["model_object"]

        # Openrouter
        if model_type == "openrouter":
            # https://openrouter.ai/docs/quickstart
            # https://openrouter.ai/models
            manufacturer = "OpenRouter"
            model_name = model_params.get(
                "model_name", settings.OPENROUTER_MODEL_NAME)
            openai_model = get_openai_api({
                "provider": manufacturer,
                "base_url": settings.OPENROUTER_BASE_URL,
                "api_key": model_params.get(
                    "api_key", settings.OPENROUTER_API_KEY),
                "model_name": model_name,
                "temperature": settings.OPENAI_TEMPERATURE,
                "top_p": settings.OPENAI_TOP_P,
                "max_tokens": settings.OPENAI_MAX_TOKENS,
                "streaming": settings.AI_STREAMING,
            })
            if openai_model["error"]:
                error = openai_model["error_message"]
            else:
                model_object = openai_model["model_object"]

        # Nvidia
        if model_type == "nvidia":
            # https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct
            manufacturer = "Nvidia"
            model_name = model_params.get(
                "model_name", settings.NVIDIA_MODEL_NAME)
            openai_model = get_openai_api({
                "provider": manufacturer,
                "base_url": settings.NVIDIA_BASE_URL,
                "api_key": model_params.get(
                    "api_key", settings.NVIDIA_API_KEY),
                "model_name": model_name,
                "temperature": settings.NVIDIA_TEMPERATURE,
                "top_p": settings.NVIDIA_TOP_P,
                "max_tokens": settings.NVIDIA_MAX_TOKENS,
                "streaming": settings.AI_STREAMING,
            })
            if openai_model["error"]:
                error = openai_model["error_message"]
            else:
                model_object = openai_model["model_object"]

        # Rhymes.ai
        if model_type == "rhymes":
            # https://lablab.ai/t/aria-api-tutorial
            manufacturer = "Rhymes.ai"
            model_name = model_params.get(
                "model_name", settings.RHYMES_CHAT_MODEL_NAME)
            openai_model = get_openai_api({
                "provider": manufacturer,
                "base_url": settings.RHYMES_CHAT_BASE_URL,
                "api_key": model_params.get(
                    "api_key", settings.RHYMES_CHAT_API_KEY),
                "model_name": model_name,
                "temperature": settings.RHYMES_CHAT_TEMPERATURE,
                "top_p": settings.RHYMES_CHAT_TOP_P,
                "max_tokens": settings.RHYMES_CHAT_MAX_TOKENS,
                "stop": ["<|im_end|>"],
                "streaming": settings.AI_STREAMING,
            })
            if openai_model["error"]:
                error = openai_model["error_message"]
            else:
                model_object = openai_model["model_object"]

        # xAI (ex-Twitter) Grok
        if model_type == "xai":
            # https://docs.x.ai/api/integrations#openai-sdk
            manufacturer = "xAI"
            model_name = model_params.get(
                "model_name", settings.XAI_MODEL_NAME)
            model_config = {
                "provider": manufacturer,
                "base_url": settings.XAI_BASE_URL,
                "api_key": model_params.get(
                    "api_key", settings.XAI_API_KEY),
                "model_name": model_name,
                "temperature": settings.XAI_TEMPERATURE,
                "top_p": settings.XAI_TOP_P,
                "max_tokens": settings.XAI_MAX_TOKENS,
                "stop": ["<|im_end|>"],
                "streaming": settings.AI_STREAMING,
            }
            _ = DEBUG and \
                log_debug(f"GET_MODEL | xAI model_config: {model_config}")
            openai_model = get_openai_api(model_config)
            if openai_model["error"]:
                error = openai_model["error_message"]
            else:
                model_object = openai_model["model_object"]

        # IBM watsonx
        if model_type == "ibm":
            manufacturer = "IBM watsonx"
            tools_permitted = False
            model_name = settings.IBM_WATSONX_MODEL_NAME
            api_key = settings.IBM_WATSONX_API_KEY
            project_id = settings.IBM_WATSONX_PROJECT_ID
            model_url = settings.IBM_WATSONX_URL
            identity_token_url = settings.IBM_WATSONX_IDENTITY_TOKEN_URL
            # n = int(settings.IBM_WATSONX_N)
            if not api_key:
                error = "ERROR [GET_MODEL-IBM-010] - Missing" \
                    " IBM_WATSONX_API_KEY"
            elif not project_id:
                error = "ERROR [GET_MODEL-IBM-020] - Missing" \
                    " IBM_WATSONX_PROJECT_ID"
            elif not model_url:
                error = "ERROR [GET_MODEL-IBM-030] - Missing" \
                    " IBM_WATSONX_URL"
            elif not model_name:
                error = "ERROR [GET_MODEL-IBM-040] - Missing" \
                    " IBM_WATSONX_MODEL_NAME"
            else:
                model_object = IbmWatsonx(
                    model_name=model_name,
                    api_key=api_key,
                    project_id=project_id,
                    model_url=model_url,
                    identity_token_url=identity_token_url,
                    # n=n,
                    app_context=app_context,
                )
        # Together
        if model_type == "together":
            # https://python.langchain.com/docs/integrations/chat/together/
            manufacturer = "Together.ai"
            model_name = model_params.get("model_name") \
                or settings.TOGETHER_MODEL_NAME
            api_key = model_params.get("api_key") or settings.TOGETHER_API_KEY
            if not api_key:
                error = \
                    "ERROR [GET_MODEL-TOGETHER-010] - Missing TOGETHER_API_KEY"
            else:
                model_config = {
                    "together_api_key": api_key,
                    "model": model_name,
                    "stop": ["<|eot_id|>", "<|eom_id|>"],
                }
                if settings.TOGETHER_TEMPERATURE:
                    model_config["temperature"] = \
                        float(settings.TOGETHER_TEMPERATURE)
                if settings.TOGETHER_TOP_P:
                    model_config["top_p"] = float(settings.TOGETHER_TOP_P)
                if settings.TOGETHER_MAX_TOKENS:
                    model_config["max_tokens"] = \
                        int(settings.TOGETHER_MAX_TOKENS)
                model_object = ChatTogether(**model_config)

    except Exception as err:
        error = f"[GET_MODEL-GENEX-010] - {err}"

    _ = error and log_error(error)

    need_preamble = need_preamble if need_preamble is not None else \
        get_need_preamble(app_context, model_name)
    result = {
        "model_type": model_type,
        "model_name": model_name,
        "model_object": model_object,
        "manufacturer": manufacturer,
        "model_params": model_params,
        "pref_agent_type": pref_agent_type,
        "system_msg_permitted": get_system_msg_permitted(
            app_context, model_name),
        "tools_permitted":
            tools_permitted if tools_permitted is not None else
            get_tools_permitted(app_context, model_name),
        "need_preamble": need_preamble,
        "preamble_model":
            get_preamble_model(app_context, model_name)
            if need_preamble else None,
        "other_data": other_data,
        "error": error,
    }
    _ = self_debug and log_debug(f"GM-2) GET_MODEL: result: {result}")
    return result


def get_model_middleware(
    app_context: AppContext,
    model_type: str,
    model_params: Optional[dict] = None,
) -> dict:
    """
    Get model object, verifying the billing plan.

    Args:
        app_context (AppContext): application context.
        model_type (str): the model type. e.g. "chat_openai",
            "gemini", "ollama", "anthropic", "clarifai"
        model_params (dict, optional): model parameters. Defaults to None.

    Returns:
        dict: a dictionary with the attributes:
            "model_type" (str): model type
            "model_object" (Any): model object
            "manufacturer" (str): manufacturer name
            "error" (str): error message
    """
    if not model_params:
        model_params = {}
    billing = BillingUtilities(app_context)
    model_params["user_plan"] = billing.get_user_plan()
    if not billing.is_free_plan():
        if model_type == "chat_openai":
            model_params["api_key"] = billing.get_openai_api_key()
            model_params["model_name"] = billing.get_openai_chat_model()
            _ = DEBUG and log_debug(
                f"GET_MODEL_MIDDLEWARE | model_params: {model_params}")
        return get_model(app_context, model_type, model_params)
    # Free plan only allows GPT with the user's OpenAI API key and user's
    # configured model or small GPT
    model_type = "chat_openai"
    model_params["api_key"] = billing.get_openai_api_key()
    model_params["model_name"] = billing.get_openai_chat_model()
    if not model_params["api_key"]:
        error = "ERROR [GET_MODEL-OAI-010] Missing OpenAI API Key"
        result = {
            "model_type": None,
            "model_name": None,
            "model_object": None,
            "manufacturer": None,
            "model_params": None,
            "pref_agent_type": None,
            "system_msg_permitted": False,
            "tools_permitted": False,
            "need_preamble": False,
            "other_data": {},
            "error": error,
        }
        _ = DEBUG and \
            log_debug(
                'GM-E1: Error getting model' +
                f'\n | error: {error}'
                f'\n | user_plan: {model_params["user_plan"]}')
        return result
    return get_model(app_context, model_type, model_params)


MODEL_ATTR_NAME_REPLACE = {
    "manufacturer": "model_manufacturer",
    "error": "model_load_error",
}


def get_model_obj(
    app_context: AppContext,
    model_type: Optional[Union[str, None]] = None,
    model_params: Optional[dict] = None,
) -> Union[RunnableSerializable, None]:
    """
    Get model object with alternative models and fallback.
    The alternative models will be returned only if
    settings.AI_ADDITIONAL_MODELS is "1"

    Args:
        app_context (AppContext): application context.
        model_type (str): the model type. e.g. "chat_openai",
            "gemini", "ollama", "anthropic", "clarifai".
            If None, assumes settings.LANGCHAIN_DEFAULT_MODEL.
            Defaults to None.

    Returns:
        Any: the model object. If something goes wrong getting the model
            object, stores the following app_context.set_other_data() elements:
            "model_type" (str): model type
            "model_manufacturer" (str): manufacturer name
            "model_load_error" (str): error message
    """
    self_debug = DEBUG or is_under_test()
    settings = Config(app_context)
    selected_model_type: str = settings.LANGCHAIN_DEFAULT_MODEL \
        if model_type is None else model_type

    if self_debug:
        log_debug("GET_MODEL_OBJ | Model (selected_model_type): >> " +
                  f'{selected_model_type} <<')

    model_response = get_model_middleware(
        app_context,
        selected_model_type,
        model_params,
    )

    # Add the last retrieved model type to the app_context
    app_context.set_other_data(
        "model_type",
        model_response["model_type"])

    # Add last model attributes to the app_context under its type
    app_context.set_other_data(
        model_response["model_type"],
        {
            MODEL_ATTR_NAME_REPLACE.get(key, key):
            model_response[key]
            for key in model_response
        }
    )

    # app_context.set_other_data("model_name",
    #                            model_response["model_name"])
    # app_context.set_other_data("pref_agent_type",
    #                            model_response["pref_agent_type"])
    # app_context.set_other_data("system_msg_permitted",
    #                            model_response["system_msg_permitted"])
    # app_context.set_other_data("tools_permitted",
    #                            model_response["tools_permitted"])
    # app_context.set_other_data("need_preamble",
    #                            model_response["need_preamble"])
    # app_context.set_other_data("model_manufacturer",
    #                            model_response["manufacturer"])
    # app_context.set_other_data("model_load_error",
    #                            model_response["error"])

    if model_response["error"]:
        log_error(f"ERROR [AI-GMO-E010] - {model_response}")
        return None
    default_model = model_response["model_object"]

    if settings.AI_ADDITIONAL_MODELS != "1":
        return default_model

    add_models = ["chat_openai", "anthropic", "gemini"]
    openai = OpenAI(
        # model="gpt-3.5-turbo-instruct"
        model=settings.OPENAI_MODEL_INSTRUCT,
        temperature=float(settings.OPENAI_TEMPERATURE),
    )
    additional_pars = {
        "which": ConfigurableField(id="model"),
        "default_key": selected_model_type,
        "openai": openai,
    }
    alternative_models = []
    for model_name in add_models:
        if model_name == selected_model_type:
            continue
        model_response = get_model_middleware(app_context, model_name)
        if model_response["error"]:
            log_error(f"ERROR [AI-GMO-E020] - {model_response}")
            additional_model = None
        else:
            additional_model = model_response["model_object"]
        if additional_model:
            additional_pars[model_name] = additional_model
            alternative_models.append(additional_pars[model_name])

    if self_debug:
        log_debug("GET_MODEL_OBJ\n" +
                  f"default_model: {default_model}\n" +
                  "alternative_models: " +
                  str(alternative_models) + "\n" +
                  "additional_pars: " +
                  str(additional_pars) + "\n")

    model = default_model \
        .with_fallbacks(alternative_models) \
        .configurable_alternatives(
            **additional_pars,
        )

    return model


def get_model_load_error(
    app_context: AppContext,
    error_code: str
) -> str:
    """
    Returns a standard model load error

    Returns:
        str: standard model load error based on the data stored
            in the app context by the get_model_obj() method:
            model_load_error, model_manufacturer, and model_type.
    """
    model_type = app_context.get_other_data("model_type")
    model_data = app_context.get_other_data(model_type)
    return "ERROR " + \
        str(model_data["model_load_error"]) + \
        ", loading model: " + \
        str(model_data["model_manufacturer"]) + \
        " / " + \
        str(model_type) + \
        f" [{error_code}]"
