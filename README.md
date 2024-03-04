# genericsuite-be-ai
The GenericSuite AI for Python (backend version).

Welcome to the repository for the GenericSuite AI for Python (backend version). This project is designed to provide a comprehensive suite of tools and functionalities for AI development, specifically tailored for backend applications in Python.

## Pre-requisites

Python 3.9+, 3.10+, 3.11

## Installation

To use GenericSuite AI in your project, install it with the following command(s):

### Pip
```bash
pip install git+https://github.com/tomkat-cr/genericsuite-be-ai
```

### Pipenv
```bash
pipenv install git+https://github.com/tomkat-cr/genericsuite-be-ai
```

### Poetry
```bash
poetry add git+https://github.com/tomkat-cr/genericsuite-be-ai
```

## Contents

This repository includes the following features:

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

Configure your application by setting up the necessary environment variables. Refer to the example `.env.example` file for the required variables.

## Documentation

For detailed documentation on each feature and module, please refer to the inline comments and docstrings within the codebase.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Carlos J. Ramirez, for initiating and maintaining this project.
- All contributors who help in enhancing and expanding the capabilities of GenericSuite AI.

Thank you for your interest in the GenericSuite AI for Python.
