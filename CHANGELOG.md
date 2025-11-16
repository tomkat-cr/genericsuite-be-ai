# CHANGELOG

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Changelog](http://keepachangelog.com/).



## [Unreleased]

### Added

### Changed

### Fixed

### Removed

### Security


## [0.2.0] - 2025-11-15

### Added
- Add "install" command to Makefile for easier dependency management.  
- Add PEM_TOOL envvar to .env.example, to select the Python package and dependency management tool (uv, pipenv, and poetry), default to "uv" [GS-77].
- Add AUTO_RELOAD envvar to .env.example, to fix some issues with the "--auto-reload" / "--reload" option running the app in "run_aws.sh", Turborepo and "uv", default to "1" [GS-77].
- Add DEBUG envvars to all AI modules to enable debug logging, default to "0" (they must be set on the .env file) [GS-230].

### Changed
- Update README for clarity and accuracy.
- Update CHANGELOG format to be more semantic.
- Update author email in pyproject.toml and setup.py.
- Bump urllib3 to 2.5.0 and numpy to 2.0.2 to have compatibility with GS BE Core.
- Refactor imports in ai_langchain_models.py to include ChatBedrock (AWS Bedrock) and langchain-aws conditionally.
- Enhance HuggingFace text query functions ("huggingface_remote" model type) to support OpenAI API integration.

### Fixed
- Fix the "langchain_community not found" by adding the "langchain-community" dependency. This addition required the upgrade of dependencies "langchain" to "0.3.26" and "faiss-cpu" to "^1.11.0.post1".
- Use Poetry to run build and publish commands in Makefile to fix a "make publish" error. 
- Add new development dependencies including build and twine to fix a "make publish" error.
- Fix the issues with the new langchain 1.0.x migrating some langchain.* imports to langchain_classic.* imports.

### Security
- Update "transformers" to "^4.57.1" to fix security vulnerabilities [GS-219]:
    * "Regular Expression Denial of Service (ReDoS)", CVE-2025-5197 and CWE-1333
    * "Hugging Face Transformers Regular Expression Denial of Service (ReDoS) vulnerability"
    * "Hugging Face Transformers vulnerable to Regular Expression Denial of Service (ReDoS) in the AdamWeightDecay optimizer"
    * "Hugging Face Transformers library has Regular Expression Denial of Service"
    * "Hugging Face Transformers is vulnerable to ReDoS through its MarianTokenizer "
    * "Transformers is vulnerable to ReDoS attack through its DonutProcessor class"
    * "Transformers's Improper Input Validation vulnerability can be exploited through username injection"
- Update "urllib3" to "^2.5.0" to fix security vulnerabilities [GS-219]:
    * "Catastrophic backtracking in URL authority parser when passed URL containing many @ characters"
    * "`Cookie` HTTP header isn't stripped on cross-origin redirects"
    * "urllib3 redirects are not disabled when retries are disabled on PoolManager instantiation"
    * "urllib3's Proxy-Authorization request header isn't stripped during cross-origin redirects"
    * "urllib3's request body not stripped after redirect from 303 status changes request method to GET"
    * "Using default SSLContext for HTTPS requests in an HTTPS proxy doesn't verify certificate hostname for proxy connection"
    * "`Cookie` HTTP header isn't stripped on cross-origin redirects"
- Update "setuptools" to "^78.1.1" to fix the security vulnerability [GS-219]:
    * "setuptools has a path traversal vulnerability in PackageIndex.download that leads to Arbitrary File Write"
- Remove "text-generation" to fix the security vulnerability (also it's not used in this project) [GS-219]:
    * "Code injection vulnerability exists in the huggingface/text-generation-inference repository"
- Update "clarifai" to "^11.9.0" to fix security vulnerabilities ("llama-index-core" is not longer a dependency of clarifi) [GS-219]:
    * "llama-index-core insecurely handles temporary files"
    * "LlamaIndex affected by a Denial of Service (DOS) in JSONReader"
    * "LlamaIndex vulnerable to Path Traversal attack through its encode_image function"
    * "LlamaIndex vulnerable to DoS attack through uncontrolled recursive JSON parsing"
    * "LlamaIndex has Incomplete Documentation of Program Execution related to JsonPickleSerializer "
- Install "pypdf" latest version "^6.1.3" to fix security vulnerabilities (it was a dependency of clarifai) [GS-219]:
    * "pypdf possibly loops infinitely when reading DCT inline images without EOF marker"
    * "pypdf can exhaust RAM via manipulated LZWDecode streams"
    * "PyPDF's Manipulated FlateDecode streams can exhaust RAM"
- Update all langchain dependencies to latest versions to fix security vulnerabilities [GS-219]:
    * "LangChain Text Splitters is vulnerable to XML External Entity (XXE) attacks due to unsafe XSLT parsing"

### Removed
- Remove "langchain-together" to enable upgrade of all other langchain dependencies, and routes Together AI calls through the OpenAI API.
- Remove "clarifai" to make it optional by default [GS-219].


## [0.1.14] - 2025-07-12

### Fixed
- Fix DuckDuckGo Search "202 Ratelimit error" [GS-224].
- Fix Google Search API requests from referer empty are blocked error [GS-223].


## [0.1.13] - 2025-07-08

### Added
- Implement OpenRouter AI provider and models [GS-182].
- Implement Vertex AI provider and models [GS-183].
- Add AI Endpoints for Flask [GS-15].
- Add envvar GET_MOCKS_DEBUG to test the text_to_audio_generator tool and save money in the sound AI API bill by using existing audio files already generated in the /tmp directory (GET_MOCKS_DEBUG="1") or a specific file path (GET_MOCKS_DEBUG="/path/to/file.mp3") [GS-185].

### Changed
- Tool web_search() updated to use DEFAULT_MAX_RESULTS=30 [GS-87].
- Add envvars to configure various parameters of the IBM WatsonX provider (IBM_WATSONX_REGION, IBM_WATSONX_TEMPERATURE, IBM_WATSONX_REPETITION_PENALTY, IBM_WATSONX_MAX_NEW_TOKENS, IBM_WATSONX_MIN_NEW_TOKENS, IBM_WATSONX_DECODING_METHOD, IBM_WATSONX_MODERATION_HAP_THRESHOLD) [GS-184].
- Implement calls to AppContext.get_env_var() when AppContext is passed to the CustomLLM class, otherwise it calls to os.environ.get() [GD-184].
- Change "web_search.py" to use min() instead of max() to enforce a minimum between the DEFAULT_MAX_RESULTS and the number of results requested by the user/model. If the request is higher than the DEFAULT_MAX_RESULTS, it will use that value.
- Change "ADDITIONAL GUIDELINES:\n" to "REQUIREMENTS:\n" and add "* " (bulletpoints) to each line in build_gs_prompt (ai_chatbot_main_langchain) to have a better system prompt for AI model.
- Change "DEBUG" to use Config().DEBUG == "1" in web_search [GS-185].

### Fixed
- Fix "ERROR: Failed building wheel for pyreqwest-impersonate" error running "sam build" (with "make deploy_run_local_qa") when "duckduckgo-search" were updated to version "6.1.1" [GS-87].
- Fix text-to-audio generation with text_to_audio_response() because sometimes with some models it never generate a new audio.
- Fix the missing model_name parameter calling get_openai_api().
- Set the DUCKDUCKGO_MAX_RESULTS to 5 to avoid the error "error: https://lite.duckduckgo.com/lite/ 202 Ratelimit" [GS-87].
- Set the DUCKDUCKGO_RATE_LIMIT_TOKEN to check the "202 Ratelimit" and switch to google search in case WEBSEARCH_DEFAULT_PROVIDER is not "ddg" or "google" [GS-87].


## [0.1.12] - 2025-02-19

### Added
- Implement Together AI provider and models [GS-158].
- Implement xAI Grok model [GS-157].
- Implement IBM WatsonX provider [GS-155] [GS-184].
- Implement generic Langchain model abstract interface [GS-155].
- Implement Nvidia API / NIM / Nemotron [GS-93].
- Implement Rhymes.ai Aria chat model [GS-152].
- Implement Rhymes.ai Allegro video generation model (only configuration) [GS-153].
- Add AI_STREAMING envvar to configure the Streaming response method [GS-32].
- Add NVIDIA_API_KEY, NVIDIA_MODEL_NAME, NVIDIA_TEMPERATURE, NVIDIA_MAX_TOKENS, NVIDIA_TOP_P, and NVIDIA_BASE_URL envvars [GS-93].
- Add RHYMES_CHAT_API_KEY, RHYMES_CHAT_MODEL_NAME, RHYMES_CHAT_TEMPERATURE, RHYMES_CHAT_MAX_TOKENS, RHYMES_CHAT_TOP_P, RHYMES_CHAT_BASE_URL, RHYMES_VIDEO_API_KEY, RHYMES_VIDEO_MODEL_NAME, RHYMES_VIDEO_BASE_URL, RHYMES_VIDEO_NUM_STEP, and RHYMES_VIDEO_CFG_SCALE envvars [GS-152].
- Add AIMLAPI_TOP_P to configure top_p parameter in AI/ML API [GS-138].
- Add OPENAI_TOP_P to configure top_p parameter in OpenAI chat model.
- Add CustomLLM abstract class for LangChain to implement new LLM providers not implemented in LangChain yet [GS-155]. 

### Changed
- Change "black-forest-labs/FLUX.1-schnell" image generation model by default.
- Change OPENAI_MAX_TOKENS and AIMLAPI_MAX_TOKENS to have '' by default to get the maximum tokens possible [GS-157].
- Change "grok-beta" changed to "grok-2" as default model for xAI [GS-157].
- Change "openai" instead of "openai_chat" to get the default OpenAI provider in the LANGCHAIN_DEFAULT_MODEL parameter.
- "get_openai_api()" function was added to standardize LLM client creation for providers compatible with the OpenAI completions API [GS-157].

### Fixed
- Fix the "ValueError: invalid literal for int() with base 10: ''" error in get_vision_response() when OPENAI_MAX_TOKENS is empty [GS-152].
- Fix poetry 2.x "The option --no-update does not exist" error message [FA-84].
- Fix "'License :: OSI Approved :: ISC License' is not a valid classifier" error running "python3 -m twine upload dist/*" [FA-84].


## [0.1.11] - 2024-10-17

### Added
- Implement preamble model to run OpenAI o1-mini/o1-preview models with Tools and System messages [GS-140].
- Add AI_PREAMBLE_MODEL_DEFAULT_TYPE, AI_PREAMBLE_MODEL_DEFAULT_MODEL, AI_PREAMBLE_MODEL_BASE_CONF, AI_PREAMBLE_MODEL_CUSTOM_CONF to customize the preamble model [GS-140].
- Implement ollama server [GS-139].
- Add AI_MODEL_ALLOW_SYSTEM_MSG, AI_MODEL_ALLOW_TOOLS, and AI_MODEL_NEED_PREAMBLE to manage models like Ollama "llava" that doesn't accept Tools [GS-140].

### Changed
- Update ChatOllama adding the "langchain-ollama" dependency [GS-139].

### Fixed
- Fix Tools make Agent returns empty responses in LCEL chains. Now Agent returns the result when there're no more Tools to call [GS-143].


## [0.1.10] - 2024-10-07

### Added
- Implement HF (HuggingFace) local pipelines [GS-59].
- Implement HF (HuggingFace) image generator [GS-117].
- Implement Flux.1 image generator [GS-117].
- Implement Anthropic Claude 3.5 Sonnet [GS-33].
- Implement Groq chat model [GS-92].
- Implement Amazon Bedrock chat and image generator [GS-131].
- Add HUGGINGFACE_PIPELINE_DEVICE to configure the "device" pipeline() parameter [FA-233].
- Implement o1-mini/o1-preview models use through AI/ML API aimlapi.com [GS-138].
- Implement GS Huggingface lightweight model, identified by model_types "huggingface_remote" or "gs_huggingface". The model_types "huggingface" and "huggingface_pipeline" use the "langchain_hugginface" dependency that required "sentence-transformers", making imposible to deploy the project AWS Lambda Functions [GS-136].
- Implement Falcon Mamba with HF [GS-118].
- Implement Meta Llama 3.1 with HF [GS-119].
- Implement HuggingFaceChatModel langchain model class that uses the HF fetch/post endpoint [GS-59].
- Add HUGGINGFACE_TEXT_TO_TEXT_ENDPOINT envvar [GS-59].


### Changed
- Langchain upgraded to "^0.3.0" [GS-131].
- Replace "gpt-3.5-turbo" with "gpt-4o-mini" as default OpenAI model [GS-109].
- HUGGINGFACE_ENDPOINT_URL replaced by HUGGINGFACE_DEFAULT_CHAT_MODEL [GS-59].
- Config class accepts both OPENAI_MODEL_NAME and OPENAI_MODEL envvars [GS-128].
- get_model() "billing" verification logic moved to get_model_middleware() [GS-128].
- The user with free plan can only use the "gpt-4o-mini" model with their own API Key, regardless of what is configured in LANGCHAIN_DEFAULT_MODEL [FA-233].
- Rename HUGGINGFACE_ENDPOINT_URL to HUGGINGFACE_TEXT_TO_IMAGE_ENDPOINT [GS-59].
- Add app_context to the IbmWatsonx langchain model class.

### Fixed
- Fix Anthropic Claude2 API Error since large prompt change, replacing Claude2 with Claude 3.5 Sonnet [GS-33].
- Fix the "Warning: deprecated HuggingFaceTextGenInference-use HuggingFaceEnpoint instead" [GS-59].
- Fix dependency incompatibility between GS BE Core and GS BE AI fixing the "urllib3" version to "1.26" (and clarifai to "^10.1.0" in consecuence) because GS BE Core's Boto3 use "urllib3" versions less then "<2" [GS-128].

### Removed
- The "langchain_hugginface" dependency is not longer included in this package. It must be imported in the App's project [GS-136].


## [0.1.9] - 2024-07-27

### Added
- Add: ".nvmrc" file to set the repo default node version.

### Fixed
- Fix: typing in create_app() parameters.


## [0.1.8] - 2024-07-18

### Added
- Add "langchain-google-community" due to a deprecation notice about GoogleSearchAPIWrapper [GS-66].
- Add Langchain Tools description length validation, to avoid descriptions > 1024 chars.

### Changed
- Update dependecies "langchain" (from ^0.1.20 to ^0.2.3), "langchain-core", "langchain-openai" (from ^0.1.6 to ^0.1.8), "tiktoken" (from 0.6 to ^0.7.0) to be able to add "langchain-google-community" [GS-66].
- Update env.example. to have the GS BE Core latest updates.
- AWS_API_GATEWAY_STAGE env. var. removed from env.example.
- All DEBUGs turned off to save logs/AWS Cloudwatch space.

### Fixed
- Fix audio processing issues in FastAPI due to AWS API Gateway limitations, sending the base64 encoded files back [GS-95].
- Change: minor linting changes.


## [0.1.7] - 2024-06-06

### Added
- Add AI Endpoints and create_app for FastAPI [FA-246].
- Add 'setup.cfg' and 'setup.py' to enable 'sam build' and 'sam local start-api' [GS-90].

### Changed
- Change 'OPENAI_VISION_MODEL' default value to 'gpt-4o'. Previously was 'gpt-4-vision-preview' [GS-78].
- Remove the "dist" from the git repo.
- Separate "messages_to_langchain_fmt_text" and "messages_to_langchain_fmt" so the latter always returns list [GS-78].
- get_functions_dict() returns the exact function name when LanchChain Tools are required [GS-78].
- Image generator save_image() returns "public_url" instead of "attachment_url", to be aligned to AWS save_file_from_url().
- Add "sendfile_callable" and "background_tasks" parameters to ai_chatbot_endpoint() to allow FastAPI send file back [GS-66].

### Fixed
- Downgrade duckduckgo-search==5.3.1b1 to remove "pyreqwest_impersonate" and fix the error building the docker image.
- Fix "Cannot instantiate typing.Union, <class 'TypeError'>" error in messages_to_langchain_fmt() avoiding AnyMessage use.
- Fix LCEL non-agent compatibility by removing "args_schema" from text_to_audio_response() Tool decorator [GS-66].


## [0.1.6] - 2024-05-28

### Added
- Implement new OpenAI model gpt-4o [GS-78].
- Add file upload on FastAPI [GS-68].
- Add STORAGE_URL_SEED and APP_HOST_NAME env. vars. to mask the S3 URL and avoid AWS over-billing attacks [GS-72].
- Conversations conversion [GS-72].
- Add requirements.txt generation to Makefile on publish.
- Add ".PHONY" entries for all labels in Makefile.

### Changed
Tiktoken and langchain-openai upgraded to use 'text-embedding-3-small' as default OPENAI_EMBEDDINGS_MODEL [GS-65].
- ANTHROPIC_MODEL defaults to 'claude-3-sonnet'.
- OpenAI vision model defaults to 'gpt-4o' [GS-78].
- Remove the GenericSuite AI library dependency from GenericSuite Core [GS-74].
- Redirect README instructions to the GenericSuite Documentation [GS-73].
- "blueprint" as mandatory parameter to GenericDbHelper, AppContext and app_context_and_set_env(), to make posible the specific functions to GenericDbHelper [GS-79].

### Fixed
- Fix DuckDuckGo & Google Search issues [GS-87].
- Implement non-agent LCEL chains to solve issue getting AI assistant responses in deployed environments [GS-66].


## [0.1.5] - 2024-04-20

### Changed
- Updated genericsuite = "0.1.5" with "mangum" to make FastAPI work on AWS Lambda [FA-246].


## [0.1.4] - 2024-04-20

### Changed
- Updated genericsuite = "0.1.4" with FastAPI enhanced support [FA-246].
- Change: README with clearer AWS_S3_CHATBOT_ATTACHMENTS_BUCKET_* example values and main image taken from the official documentation site [FA-246].
- Change: Homepage pointed to "https://genericsuite.carlosjramirez.com/Backend-Development/GenericSuite-AI/" [FA-257].
- License changed to ISC in "pyproject.toml" [FA-244].


## [0.1.3] - 2024-04-09

### Changed
- Updated genericsuite = "0.1.3".
- Add links to https://www.carlosjramirez.com/genericsuite/ in the README.
- Remove deprecated FRONTEND_AUDIENCE.


## [0.1.2] - 2024-04-01

### Added
- Add stage "demo" to APP_DB_ENGINE, APP_DB_NAME, APP_DB_URI, APP_CORS_ORIGIN, and AWS_S3_CHATBOT_ATTACHMENTS_BUCKET [FA-213].

### Changed
- Updated genericsuite = "0.1.2".
- ".env-example" renamed to ".env.example".
- AI configuration environment variables added to README.
- The GenericSuite backend development scripts added to README.
- License changed to ISC [FA-244].


## [0.1.1] - 2024-03-19

### Added
- Add Makefile with `build, `publish` and `publish-test` options.

### Changed
- README enhanced instructions.
- Updated genericsuite = "0.1.1"


## [0.1.0] - 2024-03-14

### Added
- Publish to Pypi


## [0.0.4] - 2024-03-03

### Added
- Separate BE Generic Suite to publish on PyPi [FA-219].
- Initial commit as an independent repository.


## [0.0.3] - 2024-02-19

### Added
- Add LC HuggingFace chat models [FA-233].
- Add Web scrapping module.


## [0.0.2] - 2024-02-18

### Added
- Implement GPT4 Vision [FA-144].
- Implement Audio processing with OpenAPI whisper [FA-145].
- Implement TTS-1 text-to-speech OpenAI Model [FA-210].
- Implement image generator using DALL-E 3 [FA-165].
- Implement Google Gemini models [FA-172].
- Implement Clarifai models and embeddings [FA-182].
- Add web search capability to the AI Asistant [FA-159].
- Chats stored in the DB [FA-119].
- Add the Billing Plan ("plan" attribute) to the user profile [FA-200].
- Add the OpenAI API key and model name ("openai_api_key" and "openai_model" attributes) to allow free plan users to use the AI Asistant at their own expense [FA-201].
- Add double version of GPT Functions and LC Tools [FA-211].


## [0.0.1] - 2023-07-21

### Added
- Add `ai_chatbot` endpoint to handle the AI Assistant based on OpenAI functions call [FA-93].
