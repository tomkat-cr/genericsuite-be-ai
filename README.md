# The GenericSuite AI for Python (backend version)

![GenericSuite AI Logo](https://github.com/tomkat-cr/genericsuite-fe-ai/blob/main/src/lib/images/gs_ai_logo_circle.png)

[GenericSuite AI](https://www.carlosjramirez.com/genericsuite/) is a versatile backend solution, designed to provide a comprehensive suite of features, tools and functionalities for AI oriented Python APIs.

It's bassed on [The Generic Suite (backend version)](https://github.com/tomkat-cr/genericsuite-be), so its features are inherited.

The perfect companion for this backend solution is [The GenericSuite AI (frontend version)](https://github.com/tomkat-cr/genericsuite-fe-ai)

## Pre-requisites

- [Python](https://www.python.org/downloads/) >= 3.9 and < 4.0
- [Git](https://www.atlassian.com/git/tutorials/install-git)
- Make: [Mac](https://formulae.brew.sh/formula/make) | [Windows](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows)
- Node version 18+, installed via [NVM (Node Package Manager)](https://nodejs.org/en/download/package-manager) or [NPM and Node](https://nodejs.org/en/download) install.

### AWS account and credentials

* AWS account, see [free tier](https://aws.amazon.com/free).
* AWS Token, see [Access Keys](https://us-east-1.console.aws.amazon.com/iamv2/home?region=us-east-1#/security_credentials?section=IAM_credentials).
* AWS Command-line interface, see [awscli](https://formulae.brew.sh/formula/awscli).
* API Framework and Serverless Deployment, see [Chalice](https://github.com/aws/chalice).

## Installation

First check the [Getting Started](https://github.com/tomkat-cr/genericsuite-be/blob/main/README.md#getting-started) section in the [GenericSuite backend version documentation](https://github.com/tomkat-cr/genericsuite-be/blob/main/README.md#getting-started).

To use GenericSuite AI in your project, install it with the following command(s):

### From Pypi

#### Pip
```bash
pip install genericsuite-ai
```

#### Pipenv
```bash
pipenv install genericsuite-ai
```

#### Poetry
```bash
poetry add genericsuite-ai
```

### From a specific branch in the repository, e.g. "branch_x"

#### Pip
```bash
pip install git+https://github.com/tomkat-cr/genericsuite-be-ai@branch_x
```

#### Pipenv
```bash
pipenv install git+https://github.com/tomkat-cr/genericsuite-be-ai@branch_x
```

#### Poetry
```bash
poetry add git+https://github.com/tomkat-cr/genericsuite-be-ai@branch_x
```

### Test dependencies

To execute the unit and integration test, install `pytest` and `coverage`:

#### Pip
```bash
pip install pytest coverage
```

#### Pipenv
```bash
pipenv install --dev pytest coverage
```

#### Poetry
```bash
poetry add --dev pytest coverage
```

### Development scripts installation

[The GenericSuite backend development scripts](https://github.com/tomkat-cr/genericsuite-be-scripts?tab=readme-ov-file#the-genericsuite-scripts-backend-version) contains utilities to build and deploy APIs made by The GenericSuite.

```bash
npm install -D genericsuite-be-scripts
```

## Features

- `ai_chatbot` endpoint to implement NLP conversations based on OpenAI or Langchain APIs.
- OpenAPI, Google Gemini, Anthropic, Ollama, and Hugging Face models handling.
- Clarifai models and embeddings handling.
- Computer vision (OpenAPI GPT4 Vision, Google Gemini Vision, Clarifai Vision).
- Speech-to-text processing (OpenAPI Whisper, Clarifai Audio Models).
- Text-to-speech (OpenAI TTS-1, Clarifai Audio Models).
- Image generator (OpenAI DALL-E 3, Google Gemini Image, Clarifai Image Models).
- Vector indexers (FAISS, Chroma, Clarifai, Vectara, Weaviate, MongoDBAtlasVectorSearch)
- Embedders (OpenAI, Hugging Face, Clarifai, Bedrock, Cohere, Ollama)
- Web search tool.
- Webpage scrapping and analyzing tool.
- JSON, PDF, Git and Youtube readers.
- Language translation tools.
- Chats stored in the Database.
- Plan attribute, OpenAi API key and model name in the user profile, to allow free plan users to use Models at their own expenses.

## Configuration

Configure your application by setting up the necessary environment variables. Refer to the [.env.example](https://github.com/tomkat-cr/genericsuite-be-ai/blob/main/.env.example) and [config.py](https://github.com/tomkat-cr/genericsuite-be-ai/blob/main/genericsuite_ai/config/config.py) files for the available options.

Please check the [GenericSuite backend version configuration section](https://github.com/tomkat-cr/genericsuite-be/blob/main/README.md#configuration) for more details about general environment variables.

For GenericSuite AI, there are these additional environment variables:

1. Chabot configuration
```
# Aplicacion AI assistant name
AI_ASSISTANT_NAME=ExampleBot
```

2. Google configuration
```
GOOGLE_API_KEY=google_console_api_key
GOOGLE_CSE_ID=google_console_cse_key
```

3. OpenAI configuration
```
OPENAI_API_KEY=openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
```

4. Langchain/LangSmith configuration
```
LANGCHAIN_API_KEY=langchain_api_key
LANGCHAIN_PROJECT=langchain_project
```

5. Hugging Face credentials and model URL
```
HUGGINGFACE_API_KEY=huggingface_api_key
HUGGINGFACE_ENDPOINT_URL=huggingface_endpoint_url
```

6. AWS Configuration
```
AWS_S3_CHATBOT_ATTACHMENTS_BUCKET_DEV=aws-s3-bucket-name
AWS_S3_CHATBOT_ATTACHMENTS_BUCKET_QA=aws-s3-bucket-name
AWS_S3_CHATBOT_ATTACHMENTS_BUCKET_STAGING=aws-s3-bucket-name
AWS_S3_CHATBOT_ATTACHMENTS_BUCKET_PROD=aws-s3-bucket-name
```

7. Other AI configurations<br/>(not included in the original AWS Lambda deployment. Configurable via Configuration Parameters)
```
EMBEDDINGS_ENGINE=openai
# EMBEDDINGS_ENGINE=clarifai

VECTOR_STORE_ENGINE=FAISS
# VECTOR_STORE_ENGINE=clarifai
# VECTOR_STORE_ENGINE=mongo
# VECTOR_STORE_ENGINE=vectara

`LANGCHAIN_DEFAULT_MODEL=chat_openai
# LANGCHAIN_DEFAULT_MODEL=gemini
# LANGCHAIN_DEFAULT_MODEL=huggingface
# LANGCHAIN_DEFAULT_MODEL=clarifai

AI_VISION_TECHNOLOGY=openai
# AI_VISION_TECHNOLOGY=gemini
# AI_VISION_TECHNOLOGY=clarifai

AI_IMG_GEN_TECHNOLOGY=openai
# AI_IMG_GEN_TECHNOLOGY=gemini
# AI_IMG_GEN_TECHNOLOGY=clarifai

AI_AUDIO_TO_TEXT_TECHNOLOGY=openai
# AI_AUDIO_TO_TEXT_TECHNOLOGY=google
# AI_AUDIO_TO_TEXT_TECHNOLOGY=clarifai

AI_TEXT_TO_AUDIO_TECHNOLOGY=openai
# AI_TEXT_TO_AUDIO_TECHNOLOGY=clarifai

# Add aditional models to the LLM
AI_ADDITIONAL_MODELS=0
# AI_ADDITIONAL_MODELS=1

# Langchain credentials and other parameters

# Langsmith
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true

# Agent configuration
LANGCHAIN_AGENT_TYPE=react_chat_agent
# LANGCHAIN_AGENT_TYPE=react_agent
# LANGCHAIN_AGENT_TYPE=structured_chat_agent
# LANGCHAIN_AGENT_TYPE=LLMSingleActionAgent

LANGCHAIN_MAX_ITERATIONS=8
LANGCHAIN_EARLY_STOPPING_METHOD=force
# LANGCHAIN_EARLY_STOPPING_METHOD=generate
LANGCHAIN_HANDLE_PARSING_ERR=1

# Translate final Chatbot response to the user in case user's language is not english
LANGCHAIN_TRANSLATE_USING=google_translate
# LANGCHAIN_TRANSLATE_USING=initial_prompt
# LANGCHAIN_TRANSLATE_USING=same_model
# LANGCHAIN_TRANSLATE_USING=

LANGCHAIN_USE_LANGSMITH_HUB=0
# LANGCHAIN_USE_LANGSMITH_HUB=1

# Google other parameters

GOOGLE_MODEL=gemini-pro
GOOGLE_VISION_MODEL=gemini-pro-vision
# GOOGLE_IMG_GEN_MODEL=gemini-pro-vision
GOOGLE_IMG_GEN_MODEL=imagegeneration@005

# OpenAI other parameters
OPENAI_MAX_TOKENS=500
# Addicional NLP model
OPENAI_MODEL_PREMIUM=gpt-4-turbo-preview
OPENAI_MODEL_INSTRUCT=gpt-3.5-turbo-instruct
# Computer Vision model
OPENAI_VISION_MODEL=gpt-4-vision-preview
# Image neration model
OPENAI_IMAGE_GEN_MODEL=dall-e-3
# Speech-to-text model
OPENAI_VOICE_MODEL=whisper-1
# Text-to-speech model
OPENAI_TEXT_TO_AUDIO_MODEL=tts-1
OPENAI_TEXT_TO_AUDIO_VOICE=onyx
# OPENAI_TEXT_TO_AUDIO_VOICE=alloy
# OPENAI_TEXT_TO_AUDIO_VOICE=echo
# OPENAI_TEXT_TO_AUDIO_VOICE=fable
# OPENAI_TEXT_TO_AUDIO_VOICE=nova
# OPENAI_TEXT_TO_AUDIO_VOICE=shimmer
# Embeddings model
OPENAI_EMBEDDINGS_MODEL=text-embedding-ada-002
# OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
# Embeddings premium model
OPENAI_EMBEDDINGS_MODEL_PREMIUM=text-embedding-3-large'

# Anthropic credentials and other parameters

ANTHROPIC_MODEL=claude-2
ANTHROPIC_API_KEY=

# AWS Bedrock credentials and other parameters

AWS_BEDROCK_EMBEDDINGS_MODEL_ID=amazon.titan-embed-text-v1
AWS_BEDROCK_EMBEDDINGS_PROFILE=bedrock-admin

# HuggingFace other parameters

HUGGINGFACE_MAX_NEW_TOKENS=512
HUGGINGFACE_TOP_K=50
HUGGINGFACE_TEMPERATURE=1
HUGGINGFACE_REPETITION_PENALTY=03

# IMPORTANT: about "sentence-transformers" lib. Be careful, because
# when it's included, the package size increase by 5 Gb. and if the
# app runs in a AWS Lambda Function, it overpass the package size
# deployment limit.

HUGGINGFACE_EMBEDDINGS_MODEL="BAAI/bge-base-en-v1.5"
# HUGGINGFACE_EMBEDDINGS_MODEL="sentence-transformers/all-mpnet-base-v2"

HUGGINGFACE_EMBEDDINGS_MODEL_KWARGS='{"device":"cpu"}'
HUGGINGFACE_EMBEDDINGS_ENCODE_KWARGS='{"normalize_embeddings": true}'

# Clarifai credentials and other parameters

# PAT (Personal API Token): https://clarifai.com/settings/security
CLARIFAI_PAT=
CLARIFAI_USER_ID=
CLARIFAI_APP_ID=

AI_CLARIFAI_DEFAULT_CHAT_MODEL=GPT-4
# AI_CLARIFAI_DEFAULT_CHAT_MODEL=claude-v2
# AI_CLARIFAI_DEFAULT_CHAT_MODEL=mixtral-8x7B-Instruct-v0_1
# AI_CLARIFAI_DEFAULT_CHAT_MODEL=llama2-70b-chat

AI_CLARIFAI_DEFAULT_TEXT_EMBEDDING_MODEL==text-embedding-ada
# AI_CLARIFAI_DEFAULT_TEXT_EMBEDDING_MODEL==BAAI-bge-base-en-v15

AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL=whisper
# AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL=whisper-large-v2

AI_CLARIFAI_DEFAULT_TEXT_TO_AUDIO_MODEL=speech-synthesis

AI_CLARIFAI_AUDIO_TO_TEXT_SDK_TYPE=python_sdk
# AI_CLARIFAI_AUDIO_TO_TEXT_SDK_TYPE=clarifai_grpc

AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL=stable-diffusion-xl
# AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL=dall-e-3

AI_CLARIFAI_DEFAULT_VISION_MODEL=openai-gpt-4-vision
# AI_CLARIFAI_DEFAULT_VISION_MODEL=food-item-recognition

# ElevenLabs

ELEVENLABS_API_KEY=

# Sarah
ELEVENLABS_VOICE_ID_FEMALE=EXAVITQu4vr4xnSDxMaL
# Drew
ELEVENLABS_VOICE_ID_MALE=29vD33N1CtxCmqQRPOHJ

ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_STABILITY=0.5
ELEVENLABS_SIMILARITY_BOOST=0.5
ELEVENLABS_STYLE=0
ELEVENLABS_USE_SPEAKER_BOOST=1

# Cohere credentials and other parameters

COHERE_API_KEY=
COHERE_EMBEDDINGS_MODEL=embed-english-light-v3.0

# Ollama credentials and other parameters

OLLAMA_MODEL=llama:7b
OLLAMA_EMBEDDINGS_MODEL=llama:7b

# MongooDB embeddings
MONGODB_VS_COLLECTION=
MONGODB_VS_INDEX_NAME=

# Pinecone credentials and other parameters
# PINECONE_API_KEY=
# PINECONE_ENV=

# Vectara credentials and other parameters

VECTARA_CUSTOMER_ID=
VECTARA_CORPUS_ID=
VECTARA_API_KEY=

# Weaviate credentials and other parameters

WEAVIATE_URL=
WEAVIATE_API_KEY=
```

## Code examples and JSON configuration files

The main menu, API endpoints and CRUD editor configurations are defined in the JSON configuration files.

You can find examples about configurations and how to code an App in the [GenericSuite App Creation and Configuration guide](https://github.com/tomkat-cr/genericsuite-fe/blob/main/src/configs/README.md).

## Usage

Check the [The GenericSuite backend development scripts](https://github.com/tomkat-cr/genericsuite-be-scripts?tab=readme-ov-file#the-genericsuite-scripts-backend-version) for more details.

## License

This project is licensed under the ISC License - see the [LICENSE](https://github.com/tomkat-cr/genericsuite-be-ai/blob/main/LICENSE) file for details.

## Credits

This project is developed and maintained by Carlos J. Ramirez. For more information or to contribute to the project, visit [GenericSuite AI on GitHub](https://github.com/tomkat-cr/genericsuite-be-ai).

Happy Coding!
