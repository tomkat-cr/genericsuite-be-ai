
"""
Langchain models
"""
from typing import Union, Optional

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.llms import Clarifai
from langchain_core.runnables.utils import ConfigurableField
from langchain_core.runnables.base import RunnableSerializable
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from genericsuite.util.app_context import AppContext
from genericsuite.util.app_logger import log_debug, log_error
from genericsuite.util.utilities import is_under_test

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.clarifai import (
    get_model_config,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities

DEBUG = False


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

    Returns:
        dict: a resultset with the attributes:
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
    billing = BillingUtilities(app_context)
    manufacturer = None
    model_name = None
    pref_agent_type = 'lcel'
    try:
        # OpenAI ChatGPT
        if model_type == "chat_openai":
            manufacturer = "OpenAI"
            openai_api_key = billing.get_openai_api_key()
            model_name = billing.get_openai_chat_model()
            if not openai_api_key:
                error = "ERROR [GET_MODEL-OAI-010] Missing OpenAI API Key"
                _ = DEBUG and \
                    log_debug('GM-E1: Error getting model' +
                              f'\n | error: {error}'
                              f'\n | user_plan: {billing.get_user_plan()}')
            else:
                model_object = ChatOpenAI(
                    # model="gpt-3.5-turbo"
                    model=model_name,
                    temperature=float(settings.OPENAI_TEMPERATURE),
                    openai_api_key=openai_api_key,
                )
                if not model_object:
                    error = "ERROR [GET_MODEL-OAI-020] - ChatOpenAI cannot" + \
                            " be initialized"

        # Google Gemini
        if model_type == "gemini" and settings.GOOGLE_API_KEY:
            # Google Generative AI Chatbot : Gemini
            # https://python.langchain.com/docs/integrations/chat/google_generative_ai
            manufacturer = "Google"
            model_name = settings.GOOGLE_MODEL
            model_object = ChatGoogleGenerativeAI(
                # model="gemini-pro",
                model=model_name,
                temperature=float(settings.OPENAI_TEMPERATURE),
                google_api_key=settings.GOOGLE_API_KEY,
                convert_system_message_to_human=True,
            )

        # Ollama
        if model_type == "ollama":
            manufacturer = "Ollama"
            model_name = settings.OLLAMA_MODEL
            model_object = ChatOllama(
                # model="llama:7b",
                model=model_name,
            )

        # Hugging Face Inference API
        if model_type == "huggingface":
            # https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface
            # https://python.langchain.com/v0.2/docs/integrations/llms/huggingface_endpoint/
            # https://python.langchain.com/v0.2/docs/integrations/chat/huggingface/
            pref_agent_type = 'react_chat_agent'    # Doesn't work with LCEL
            manufacturer = "Hugging Face"
            model_name = settings.HUGGINGFACE_DEFAULT_CHAT_MODEL
            # if 'url' in model_params:
            #     model_name = model_params['url']
            if 'repo_id' in model_params:
                model_name = model_params['repo_id']
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
            )
            # model_object = ChatHuggingFace(llm=llm)

        # Hugging Face Pipelines
        if model_type == "huggingface_pipeline":
            # https://python.langchain.com/v0.2/docs/integrations/llms/huggingface_pipelines/
            pref_agent_type = 'react_chat_agent'    # Doesn't work with LCEL
            manufacturer = "Hugging Face (Pipeline)"
            model_name = settings.HUGGINGFACE_DEFAULT_CHAT_MODEL
            if 'repo_id' in model_params:
                model_name = model_params['repo_id']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=int(settings.HUGGINGFACE_MAX_NEW_TOKENS),
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
            )
            model_object = HuggingFacePipeline(pipeline=pipe)

        # Anthropic Claude
        if model_type == "anthropic":
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
            # https://python.langchain.com/v0.2/docs/integrations/chat/groq/
            manufacturer = "Groq"
            model_name = settings.GROQ_MODEL
            if 'model' in model_params:
                model_name = model_params['model']
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

    except Exception as err:
        error = f"[GET_MODEL-GENEX-010] - {err}"

    _ = error and log_error(error)
    result = {
        "model_type": model_type,
        "model_name": model_name,
        "model_object": model_object,
        "manufacturer": manufacturer,
        "model_params": model_params,
        "pref_agent_type": pref_agent_type,
        "error": error,
    }
    _ = self_debug and log_debug(f"GM-2) GET_MODEL: result: {result}")
    return result


def get_model_obj(
    app_context: AppContext,
    model_type: Optional[Union[str, None]] = None,
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

    get_model_response = get_model(app_context, selected_model_type)
    app_context.set_other_data("model_type",
                               get_model_response["model_type"])
    app_context.set_other_data("model_name",
                               get_model_response["model_name"])
    app_context.set_other_data("pref_agent_type",
                               get_model_response["pref_agent_type"])
    app_context.set_other_data("model_manufacturer",
                               get_model_response["manufacturer"])
    app_context.set_other_data("model_load_error",
                               get_model_response["error"])
    if get_model_response["error"]:
        log_error(f"ERROR [AI-GMO-E010] - {get_model_response}")
        return None
    default_model = get_model_response["model_object"]

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
        get_model_response = get_model(app_context, model_name)
        if get_model_response["error"]:
            log_error(f"ERROR [AI-GMO-E020] - {get_model_response}")
            additional_model = None
        else:
            additional_model = get_model_response["model_object"]
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
    return "ERROR " + \
        str(app_context.get_other_data("model_load_error")) + \
        ", loading model: " + \
        str(app_context.get_other_data("model_manufacturer")) + \
        " / " + \
        str(app_context.get_other_data("model_type")) + \
        f" [{error_code}]"
