"""
General configuration module for AI projects.
"""
# C0103 | Disable "name doesn't conform to naming rules..." (snake_case)
# pylint: disable=C0103
# R0902 | Disable "too-many-instance-attributes"
# pylint: disable=R0902
# R0903 | Disable "too-few-public-methods"
# pylint: disable=R0903
# R0915 | Disable "too-many-statements "
# pylint: disable=R0915
# W0105 | Disable "pointless-string-statement" (for """ comments)
# pylint: disable=W0105
# C0301: | Disable "line-too-long"
# pylint: disable=C0301

from typing import Any
import os

from genericsuite.config.config import (
    Config as ConfigSuperClass,
    # config_log_debug as log_debug,
    text_to_dict,
)


class Config(ConfigSuperClass):
    """
    General configuration parameters for AI projects
    """

    def __init__(self, app_context: Any = None) -> None:
        super().__init__(app_context)

        """
        AI general configuration - BEGIN
        --------------------------------
        """

        self.AI_ASSISTANT_NAME = os.environ.get('AI_ASSISTANT_NAME',
                                                'AI BOT')

        self.AI_TECHNOLOGY = self.get_env(
            'AI_TECHNOLOGY', 'langchain'
            # 'AI_TECHNOLOGY', 'openai'
        )

        self.EMBEDDINGS_ENGINE = self.get_env(
            'EMBEDDINGS_ENGINE', 'openai'
            # 'EMBEDDINGS_ENGINE', 'clarifai'
            # 'EMBEDDINGS_ENGINE', 'bedrock'
            # 'EMBEDDINGS_ENGINE', 'huggingface'
            # 'EMBEDDINGS_ENGINE', 'cohere'
            # 'EMBEDDINGS_ENGINE', 'ollama'
        )

        self.VECTOR_STORE_ENGINE = self.get_env(
            'VECTOR_STORE_ENGINE', 'FAISS'
            # 'VECTOR_STORE_ENGINE', 'clarifai'
            # 'VECTOR_STORE_ENGINE', 'mongo'
            # 'VECTOR_STORE_ENGINE', 'vectara'
        )

        self.LANGCHAIN_DEFAULT_MODEL = self.get_env(
            'LANGCHAIN_DEFAULT_MODEL', 'chat_openai'
            # 'LANGCHAIN_DEFAULT_MODEL', 'anthropic'
            # 'LANGCHAIN_DEFAULT_MODEL', 'huggingface_remote' # gs_huggingface
            # 'LANGCHAIN_DEFAULT_MODEL', 'huggingface'
            # 'LANGCHAIN_DEFAULT_MODEL', 'huggingface_pipeline'
            # 'LANGCHAIN_DEFAULT_MODEL', 'groq'
            # 'LANGCHAIN_DEFAULT_MODEL', 'gemini'
            # 'LANGCHAIN_DEFAULT_MODEL', 'clarifai'
            # 'LANGCHAIN_DEFAULT_MODEL', 'bedrock'
        )

        self.AI_VISION_TECHNOLOGY = self.get_env(
            'AI_VISION_TECHNOLOGY', 'openai'
            # 'AI_VISION_TECHNOLOGY', 'gemini'
            # 'AI_VISION_TECHNOLOGY', 'clarifai'
        )

        self.AI_IMG_GEN_TECHNOLOGY = self.get_env(
            # 'AI_IMG_GEN_TECHNOLOGY', 'openai'
            'AI_IMG_GEN_TECHNOLOGY', 'huggingface'
            # 'AI_IMG_GEN_TECHNOLOGY', 'gemini'
            # 'AI_IMG_GEN_TECHNOLOGY', 'clarifai'
            # 'AI_IMG_GEN_TECHNOLOGY', 'bedrock'
        )

        self.AI_AUDIO_TO_TEXT_TECHNOLOGY = self.get_env(
            'AI_AUDIO_TO_TEXT_TECHNOLOGY', 'openai'
            # 'AI_AUDIO_TO_TEXT_TECHNOLOGY', 'clarifai'
            # 'AI_AUDIO_TO_TEXT_TECHNOLOGY', 'google'
            # # TODO implement... 'google' or 'gemini' ???
        )

        self.AI_TEXT_TO_AUDIO_TECHNOLOGY = self.get_env(
            'AI_TEXT_TO_AUDIO_TECHNOLOGY', 'openai'
            # 'AI_TEXT_TO_AUDIO_TECHNOLOGY', 'clarifai'
        )

        self.AI_ADDITIONAL_MODELS = self.get_env(
            # 'AI_ADDITIONAL_MODELS', '1'   # Add aditional models to the LLM
            'AI_ADDITIONAL_MODELS', '0'
        )

        self.WEBSEARCH_DEFAULT_PROVIDER = self.get_env(
            'WEBSEARCH_DEFAULT_PROVIDER', ''    # First DDG, if error, Google
            # 'WEBSEARCH_DEFAULT_PROVIDER', 'ddg'   # DuckDuckGo
            # 'WEBSEARCH_DEFAULT_PROVIDER', 'google'   # Google
        )

        # Allows run the Assistant withhout Tools calling in case the LLM
        # model doesn't sopport bind_tools / bind_functions methods.
        self.AI_ALLOW_INFERENCE_WITH_NO_TOOLS = self.get_env(
            'AI_ALLOW_INFERENCE_WITH_NO_TOOLS', '0'
            # 'AI_ALLOW_INFERENCE_WITH_NO_TOOLS', '1'
        )

        """
        --------------------------------
        AI general configuration - END
        """

        # Langchain credentials and other parameters

        # Langsmith
        self.LANGCHAIN_API_KEY = self.get_env('LANGCHAIN_API_KEY', '')
        self.LANGCHAIN_PROJECT = self.get_env('LANGCHAIN_PROJECT', '')
        self.LANGCHAIN_ENDPOINT = self.get_env(
            'LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com'
        )
        self.LANGCHAIN_TRACING_V2 = self.get_env(
            'LANGCHAIN_TRACING_V2', 'true'
        )

        # Agent configuration

        self.LANGCHAIN_AGENT_TYPE = self.get_env(
            'LANGCHAIN_AGENT_TYPE', "lcel"
            # 'LANGCHAIN_AGENT_TYPE', "react_chat_agent"
            # 'LANGCHAIN_AGENT_TYPE', "react_agent"
            # 'LANGCHAIN_AGENT_TYPE', "structured_chat_agent"
            # 'LANGCHAIN_AGENT_TYPE', "LLMSingleActionAgent"
        )

        self.LANGCHAIN_MAX_CONV_MESSAGES = self.get_env(
            'LANGCHAIN_MAX_CONV_MESSAGES', '30'
            # 'LANGCHAIN_MAX_CONV_MESSAGES', '-1'     # Default: preserve all
        )

        self.LANGCHAIN_MAX_ITERATIONS = self.get_env(
            'LANGCHAIN_MAX_ITERATIONS', '8'     # Default: 15
        )
        self.LANGCHAIN_EARLY_STOPPING_METHOD = self.get_env(
            'LANGCHAIN_EARLY_STOPPING_METHOD', 'force'
            # 'LANGCHAIN_EARLY_STOPPING_METHOD', 'generate'
        )
        self.LANGCHAIN_HANDLE_PARSING_ERR = self.get_env(
            'LANGCHAIN_HANDLE_PARSING_ERR', '1'
        )

        self.LANGCHAIN_TRANSLATE_USING = self.get_env(
            'LANGCHAIN_TRANSLATE_USING', 'google_translate'
            # 'LANGCHAIN_TRANSLATE_USING', 'initial_prompt'
            # 'LANGCHAIN_TRANSLATE_USING', 'same_model'
            # 'LANGCHAIN_TRANSLATE_USING', ''
        )

        self.LANGCHAIN_USE_LANGSMITH_HUB = self.get_env(
            'LANGCHAIN_USE_LANGSMITH_HUB', '0'
        )

        # Google credentials and other parameters

        self.GOOGLE_API_KEY = self.get_env('GOOGLE_API_KEY', '')
        self.GOOGLE_CSE_ID = self.get_env('GOOGLE_CSE_ID', '')
        self.GOOGLE_MODEL = self.get_env('GOOGLE_MODEL', 'gemini-pro')
        self.GOOGLE_VISION_MODEL = self.get_env(
            'GOOGLE_VISION_MODEL', 'gemini-pro-vision'
        )
        self.GOOGLE_IMG_GEN_MODEL = self.get_env(
            # 'GOOGLE_IMG_GEN_MODEL', 'gemini-pro-vision'
            'GOOGLE_IMG_GEN_MODEL', 'imagegeneration@005'
        )

        # OpenAI credentials and other parameters

        self.OPENAI_API_KEY = self.get_env('OPENAI_API_KEY', '')

        self.OPENAI_MODEL = self.get_env(
            'OPENAI_MODEL_NAME', self.get_env('OPENAI_MODEL', 'gpt-4o-mini')
            # 'OPENAI_MODEL', 'gpt-3.5-turbo'
        )
        self.OPENAI_MODEL_PREMIUM = self.get_env(
            'OPENAI_MODEL_PREMIUM', 'gpt-4o'
            # 'OPENAI_MODEL_PREMIUM', 'gpt-4-turbo'
        )
        self.OPENAI_MODEL_INSTRUCT = self.get_env(
            'OPENAI_MODEL_INSTRUCT', 'gpt-3.5-turbo-instruct'
        )
        self.OPENAI_VISION_MODEL = self.get_env(
            'OPENAI_VISION_MODEL', 'gpt-4o'
            # 'OPENAI_VISION_MODEL', 'gpt-4-turbo'
            # 'OPENAI_VISION_MODEL', 'gpt-4-vision-preview'
        )
        self.OPENAI_IMAGE_GEN_MODEL = self.get_env(
            'OPENAI_IMAGE_GEN_MODEL', 'dall-e-3'
        )
        self.OPENAI_VOICE_MODEL = self.get_env(
            'OPENAI_VOICE_MODEL', 'whisper-1'
        )
        self.OPENAI_TEXT_TO_AUDIO_MODEL = self.get_env(
            'OPENAI_TEXT_TO_AUDIO_MODEL', 'tts-1'
        )
        self.OPENAI_TEXT_TO_AUDIO_VOICE = self.get_env(
            'OPENAI_TEXT_TO_AUDIO_VOICE', 'onyx'
            # 'OPENAI_TEXT_TO_AUDIO_VOICE', 'alloy' | 'echo', 'fable', 'onyx',
            # 'nova', 'shimmer'
        )
        self.OPENAI_EMBEDDINGS_MODEL = self.get_env(
            'OPENAI_EMBEDDINGS_MODEL', 'text-embedding-3-small'
            # 'OPENAI_EMBEDDINGS_MODEL', 'text-embedding-ada-002'
        )
        self.OPENAI_EMBEDDINGS_MODEL_PREMIUM = self.get_env(
            'OPENAI_EMBEDDINGS_MODEL_PREMIUM', 'text-embedding-3-large'
            # TODO implement this param
        )

        self.OPENAI_TEMPERATURE = self.get_env('OPENAI_TEMPERATURE', '0.7')
        self.OPENAI_MAX_TOKENS = self.get_env('OPENAI_MAX_TOKENS', '500')

        # Anthropic credentials and other parameters

        self.ANTHROPIC_MODEL = self.get_env(
            'ANTHROPIC_MODEL', 'claude-3-5-sonnet-20240620'
            # 'ANTHROPIC_MODEL', 'claude-3-sonnet'
        )
        self.ANTHROPIC_API_KEY = self.get_env('ANTHROPIC_API_KEY', '')

        # AWS credentials and other parameters

        self.AWS_REGION = self.get_env('AWS_REGION', 'us-east-1')
        self.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET = \
            self.get_env('AWS_S3_CHATBOT_ATTACHMENTS_BUCKET')

        # AWS Bedrock credentials and other parameters

        self.AWS_BEDROCK_MODEL_ID = self.get_env(
            'AWS_BEDROCK_MODEL_ID',
            "amazon.titan-text-premier-v1:0"
            "amazon.titan-text-express-v1"
            "ai21.jamba-instruct-v1:0"
            "anthropic.claude-3-haiku-20240307-v1:0"
            "anthropic.claude-3-opus-20240229-v1:0"
            "anthropic.claude-3-sonnet-20240229-v1:0"
            "anthropic.claude-3-5-sonnet-20240229-v1:0"
        )

        self.AWS_BEDROCK_IMAGE_GEN_MODEL_ID = self.get_env(
            'AWS_BEDROCK_IMAGE_GEN_MODEL_ID',
            "stability.stable-diffusion-xl-v1"
        )

        self.AWS_BEDROCK_CREDENTIALS_PROFILE = self.get_env(
            'AWS_BEDROCK_CREDENTIALS_PROFILE', ""
            # 'AWS_BEDROCK_CREDENTIALS_PROFILE', "bedrock-admin"
        )

        self.AWS_BEDROCK_GUARDRAIL_ID = self.get_env(
            'AWS_BEDROCK_GUARDRAIL_ID', ""
        )
        self.AWS_BEDROCK_GUARDRAIL_VERSION = self.get_env(
            'AWS_BEDROCK_GUARDRAIL_VERSION', ""
        )
        self.AWS_BEDROCK_GUARDRAIL_TRACE = self.get_env(
            'AWS_BEDROCK_GUARDRAIL_TRACE', "1"
        )

        self.AWS_BEDROCK_EMBEDDINGS_MODEL_ID = self.get_env(
            'AWS_BEDROCK_EMBEDDINGS_MODEL_ID', "amazon.titan-embed-text-v1"
        )
        self.AWS_BEDROCK_EMBEDDINGS_PROFILE = self.get_env(
            'AWS_BEDROCK_EMBEDDINGS_PROFILE', ""
            # 'AWS_BEDROCK_EMBEDDINGS_PROFILE', "bedrock-admin"
        )

        # HuggingFace (HF) credentials and other parameters

        self.HUGGINGFACE_API_KEY = self.get_env('HUGGINGFACE_API_KEY', "")

        # HF Models

        self.HUGGINGFACE_DEFAULT_CHAT_MODEL = self.get_env(
            'HUGGINGFACE_DEFAULT_CHAT_MODEL',
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
            # "meta-llama/Llama-2-7b-chat-hf"
            # NOTE: instruct models must be configured to run with
            # the HUGGINGFACE_USE_CHAT_HF = "0"
            # "meta-llama/Meta-Llama-3.1-8B-Instruct"
            # NOTE: Big models work with huggingface_pipeline only
            # "meta-llama/Meta-Llama-3.1-8B"
            # "meta-llama/Meta-Llama-3.1-405B-Instruct"
            # "tiiuae/falcon-mamba-7b"
        )

        # 0 = use HuggingFaceEndpoint + LangChain Agent
        # 1 = use ChatHuggingFace + LangChain LCEL
        # By default, it's set to "0" for instruct models
        self.HUGGINGFACE_USE_CHAT_HF = self.get_env(
            'HUGGINGFACE_USE_CHAT_HF',
            "0" if "-Instruct" in self.HUGGINGFACE_DEFAULT_CHAT_MODEL else "1"
        )

        self.HUGGINGFACE_DEFAULT_IMG_GEN_MODEL = self.get_env(
            'HUGGINGFACE_DEFAULT_IMG_GEN_MODEL',
            # "black-forest-labs/FLUX.1-schnell"
            "black-forest-labs/FLUX.1-dev"
        )

        # HF Embeddings

        # IMPORTANT: about "sentence-transformers" lib. Be careful, because
        # when it's included, the package size increase by 5 Gb. and if the
        # app runs in a AWS Lambda Function, it exceeds the package size
        # deployment limit.

        self.HUGGINGFACE_EMBEDDINGS_MODEL = self.get_env(
            'HUGGINGFACE_EMBEDDINGS_MODEL',
            # "sentence-transformers/all-mpnet-base-v2"
            "BAAI/bge-base-en-v1.5"
        )
        self.HUGGINGFACE_EMBEDDINGS_MODEL_KWARGS = self.get_env(
            'HUGGINGFACE_EMBEDDINGS_MODEL_KWARGS',
            text_to_dict('{"device":"cpu"}')
        )
        self.HUGGINGFACE_EMBEDDINGS_ENCODE_KWARGS = self.get_env(
            'HUGGINGFACE_EMBEDDINGS_ENCODE_KWARGS',
            text_to_dict('{"normalize_embeddings": true}')
        )

        # HF Options and general parameters

        self.HUGGINGFACE_ENDPOINT_URL = self.get_env(
            'HUGGINGFACE_ENDPOINT_URL',
            "https://api-inference.huggingface.co/models"
        )
        self.HUGGINGFACE_VERBOSE = self.get_env(
            "HUGGINGFACE_VERBOSE", "0")

        self.HUGGINGFACE_MAX_NEW_TOKENS = self.get_env(
            "HUGGINGFACE_MAX_NEW_TOKENS", "512")

        self.HUGGINGFACE_TOP_K = self.get_env(
            "HUGGINGFACE_TOP_K", "50")

        self.HUGGINGFACE_TEMPERATURE = self.get_env(
            "HUGGINGFACE_TEMPERATURE", "0.1")

        self.HUGGINGFACE_REPETITION_PENALTY = self.get_env(
            "HUGGINGFACE_REPETITION_PENALTY", "1.03")

        self.HUGGINGFACE_TIMEOUT = self.get_env(
            "HUGGINGFACE_TIMEOUT", "60")

        # Reference: from python3.xx > transformers > def pipeline() comment:
        # device (`int` or `str` or `torch.device`):
        #   Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU
        #   ordinal rank like `1`) on which this pipeline will be allocated.
        self.HUGGINGFACE_PIPELINE_DEVICE = self.get_env(
            "HUGGINGFACE_PIPELINE_DEVICE",
            ""
            # "0"
            # "cuda"
        )

        # Groq

        self.GROQ_API_KEY = self.get_env('GROQ_API_KEY', '')
        self.GROQ_TEMPERATURE = self.get_env('GROQ_TEMPERATURE', '0')
        self.GROQ_MAX_TOKENS = self.get_env('GROQ_MAX_TOKENS', '')
        self.GROQ_TIMEOUT = self.get_env('GROQ_TIMEOUT', '')
        self.GROQ_MAX_RETRIES = self.get_env('GROQ_MAX_RETRIES', '2')
        # https://console.groq.com/docs/models
        self.GROQ_MODEL = self.get_env(
            'GROQ_MODEL',
            'mixtral-8x7b-32768'
            # 'llama-3.1-70b-versatile'
            # 'llama-3.1-8b-instant'
        )

        # Clarifai credentials and other parameters

        # PAT (Personal API Token): https://clarifai.com/settings/security
        self.CLARIFAI_PAT = self.get_env('CLARIFAI_PAT')
        self.CLARIFAI_USER_ID = self.get_env('CLARIFAI_USER_ID')
        self.CLARIFAI_APP_ID = self.get_env('CLARIFAI_APP_ID')

        self.AI_CLARIFAI_DEFAULT_CHAT_MODEL = self.get_env(
            'AI_CLARIFAI_DEFAULT_CHAT_MODEL', 'GPT-4'
            # 'AI_CLARIFAI_DEFAULT_CHAT_MODEL', 'claude-v2'
            # 'AI_CLARIFAI_DEFAULT_CHAT_MODEL', 'mixtral-8x7B-Instruct-v0_1'
            # 'AI_CLARIFAI_DEFAULT_CHAT_MODEL', 'llama2-70b-chat'
        )

        self.AI_CLARIFAI_DEFAULT_TEXT_EMBEDDING_MODEL = self.get_env(
            'AI_CLARIFAI_DEFAULT_TEXT_EMBEDDING_MODEL',
            'text-embedding-ada'
            # 'BAAI-bge-base-en-v15'
        )

        self.AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL = self.get_env(
            'AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL', 'whisper'
            # 'AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL', 'whisper-large-v2'
        )

        self.AI_CLARIFAI_DEFAULT_TEXT_TO_AUDIO_MODEL = self.get_env(
            'AI_CLARIFAI_DEFAULT_TEXT_TO_AUDIO_MODEL', 'speech-synthesis'
        )

        self.AI_CLARIFAI_AUDIO_TO_TEXT_SDK_TYPE = self.get_env(
            'AI_CLARIFAI_AUDIO_TO_TEXT_SDK_TYPE', 'python_sdk'
            # 'AI_CLARIFAI_AUDIO_TO_TEXT_SDK_TYPE', 'clarifai_grpc'
        )

        self.AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL = self.get_env(
            'AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL', 'stable-diffusion-xl'
            # 'AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL', 'dall-e-3'
        )

        self.AI_CLARIFAI_DEFAULT_VISION_MODEL = self.get_env(
            'AI_CLARIFAI_DEFAULT_VISION_MODEL', 'openai-gpt-4-vision'
            # 'AI_CLARIFAI_DEFAULT_VISION_MODEL', 'food-item-recognition'
        )

        # AI/ML API

        self.AIMLAPI_API_KEY = self.get_env('AIMLAPI_API_KEY', "")
        self.AIMLAPI_MODEL_NAME = self.get_env(
            'AIMLAPI_MODEL_NAME',
            "o1-mini"
            # "o1-preview"
        )
        self.AIMLAPI_TEMPERATURE = self.get_env('AIMLAPI_TEMPERATURE', '1')

        self.AIMLAPI_BASE_URL = self.get_env(
            'AIMLAPI_BASE_URL',
            "https://api.aimlapi.com/"
        )

        # ElevenLabs

        self.ELEVENLABS_API_KEY = self.get_env('ELEVENLABS_API_KEY', "")

        self.ELEVENLABS_VOICE_ID_FEMALE = self.get_env(
            'ELEVENLABS_VOICE_ID_FEMALE',
            "EXAVITQu4vr4xnSDxMaL")  # Sarah
        self.ELEVENLABS_VOICE_ID_MALE = self.get_env(
            'ELEVENLABS_VOICE_ID_FEMALE',
            "29vD33N1CtxCmqQRPOHJ")  # Drew

        self.ELEVENLABS_MODEL_ID = self.get_env(
            'ELEVENLABS_MODEL_ID',
            "eleven_multilingual_v2")
        self.ELEVENLABS_STABILITY = self.get_env('ELEVENLABS_STABILITY', "0.5")
        self.ELEVENLABS_SIMILARITY_BOOST = self.get_env(
            'ELEVENLABS_SIMILARITY_BOOST', "0.5")
        self.ELEVENLABS_STYLE = self.get_env('ELEVENLABS_STYLE', "0")
        self.ELEVENLABS_USE_SPEAKER_BOOST = self.get_env(
            'ELEVENLABS_USE_SPEAKER_BOOST', "1")

        # Cohere credentials and other parameters

        self.COHERE_API_KEY = self.get_env('COHERE_API_KEY', "")
        self.COHERE_EMBEDDINGS_MODEL = self.get_env(
            'COHERE_EMBEDDINGS_MODEL', "embed-english-light-v3.0"
        )

        # Ollama credentials and other parameters

        self.OLLAMA_MODEL = self.get_env('OLLAMA_MODEL', "llama:7b")
        self.OLLAMA_EMBEDDINGS_MODEL = self.get_env(
            'OLLAMA_EMBEDDINGS_MODEL', "llama:7b"
        )

        # MongoDB credentials and other parameters

        # For MongooDB embeddings
        self.MONGODB_VS_COLLECTION = self.get_env(
            'MONGODB_VS_COLLECTION', ""
        )
        self.MONGODB_VS_INDEX_NAME = self.get_env(
            'MONGODB_VS_INDEX_NAME', ""
        )

        # Pinecone credentials and other parameters
        # PINECONE_API_KEY = self.get_env('PINECONE_API_KEY', "")
        # PINECONE_ENV = self.get_env('PINECONE_ENV', "")

        # Vectara credentials and other parameters

        self.VECTARA_CUSTOMER_ID = self.get_env('VECTARA_CUSTOMER_ID', "")
        self.VECTARA_CORPUS_ID = self.get_env('VECTARA_CORPUS_ID', "")
        self.VECTARA_API_KEY = self.get_env('VECTARA_API_KEY', "")

        # Weaviate credentials and other parameters

        self.WEAVIATE_URL = self.get_env('WEAVIATE_URL', "")
        self.WEAVIATE_API_KEY = self.get_env('WEAVIATE_API_KEY', "")
