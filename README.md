# GenericSuite AI (backend version)
The GenericSuite AI for Python (backend version).

![GenericSuite AI Logo](https://github.com/tomkat-cr/genericsuite-fe-ai/blob/main/src/lib/images/gs_ai_logo_circle.svg)

GenericSuite AI is a versatile backend solution, designed to provide a comprehensive suite of features, tools and functionalities for AI oriented Python APIs.

It's bassed on [The Generic Suite (backend version)](https://github.com/tomkat-cr/genericsuite-be), so its features are inherited.

The perfect companion for this backend solution is [The GenericSuite AI (frontend version)](https://github.com/tomkat-cr/genericsuite-fe-ai)


## Pre-requisites

- Python >= 3.10+ and < 4.0

## Installation

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


## Credits

This project is developed and maintained by Carlos J. Ramirez. For more information or to contribute to the project, visit [GenericSuite AI on GitHub](https://github.com/tomkat-cr/genericsuite-be-ai).

Happy Coding!