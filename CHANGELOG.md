# CHANGELOG

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Changelog](http://keepachangelog.com/).



## Unreleased
---

### New

### Changes

### Fixes

### Breaks


## 0.1.8 (2024-07-18)

### New
Add "langchain-google-community" due to a deprecation notice about GoogleSearchAPIWrapper [GS-66].
Add Langchain Tools description length validation, to avoid descriptions > 1024 chars.

### Changes
Update dependecies "langchain" (from ^0.1.20 to ^0.2.3), "langchain-core", "langchain-openai" (from ^0.1.6 to ^0.1.8), "tiktoken" (from 0.6 to ^0.7.0) to be able to add "langchain-google-community" [GS-66].
Update env.example. to have the GS BE Core latest updates.
AWS_API_GATEWAY_STAGE env. var. removed from env.example.
All DEBUGs turned off to save logs/AWS Cloudwatch space.

### Fixes
Fix audio processing issues in FastAPI due to AWS API Gateway limitations, sending the base64 encoded files back [GS-95].
Change: minor linting changes.


## 0.1.7 (2024-06-06)

### New
Add AI Endpoints and create_app for FastAPI [FA-246].
Add 'setup.cfg' and 'setup.py' to enable 'sam build' and 'sam local start-api' [GS-90].

### Changes
Change 'OPENAI_VISION_MODEL' default value to 'gpt-4o'. Previously was 'gpt-4-vision-preview' [GS-78].
Remove the "dist" from the git repo.
Separate "messages_to_langchain_fmt_text" and "messages_to_langchain_fmt" so the latter always returns list [GS-78].
get_functions_dict() returns the exact function name when LanchChain Tools are required [GS-78].
Image generator save_image() returns "public_url" instead of "attachment_url", to be aligned to AWS save_file_from_url().
Add "sendfile_callable" and "background_tasks" parameters to ai_chatbot_endpoint() to allow FastAPI send file back [GS-66].

### Fixes
Downgrade duckduckgo-search==5.3.1b1 to remove "pyreqwest_impersonate" and fix the error building the docker image.
Fix "Cannot instantiate typing.Union, <class 'TypeError'>" error in messages_to_langchain_fmt() avoiding AnyMessage use.
Fix LCEL non-agent compatibility by removing "args_schema" from text_to_audio_response() Tool decorator [GS-66].


## 0.1.6 (2024-05-28)
---

### New
Implement new OpenAI model gpt-4o [GS-78].
Add file upload on FastAPI [GS-68].
Add STORAGE_URL_SEED and APP_HOST_NAME env. vars. to mask the S3 URL and avoid AWS over-billing attacks [GS-72].
Conversations conversion [GS-72].
Add requirements.txt generation to Makefile on publish.
Add ".PHONY" entries for all labels in Makefile.

### Changes
Tiktoken and langchain-openai upgraded to use 'text-embedding-3-small' as default OPENAI_EMBEDDINGS_MODEL [GS-65].
ANTHROPIC_MODEL defaults to 'claude-3-sonnet'.
OpenAI vision model defaults to 'gpt-4o' [GS-78].
Remove the GenericSuite AI library dependency from GenericSuite Core [GS-74].
Redirect README instructions to the GenericSuite Documentation [GS-73].
"blueprint" as mandatory parameter to GenericDbHelper, AppContext and app_context_and_set_env(), to make posible the specific functions to GenericDbHelper [GS-79].

### Fixes
Fix DuckDuckGo & Google Search issues [GS-87].
Implement non-agent LCEL chains to solve issue getting AI assistant responses in deployed environments [GS-66].


## 0.1.5 (2024-04-20)
---

### Changes
Updated genericsuite = "0.1.5" with "mangum" to make FastAPI work on AWS Lambda [FA-246].


## 0.1.4 (2024-04-20)
---

### Changes
Updated genericsuite = "0.1.4" with FastAPI enhanced support [FA-246].
Change: README with clearer AWS_S3_CHATBOT_ATTACHMENTS_BUCKET_* example values and main image taken from the official documentation site [FA-246].
Change: Homepage pointed to "https://genericsuite.carlosjramirez.com/Backend-Development/GenericSuite-AI/" [FA-257].
License changed to ISC in "pyproject.toml" [FA-244].


## 0.1.3 (2024-04-09)
---

### Changes
Updated genericsuite = "0.1.3".
Add links to https://www.carlosjramirez.com/genericsuite/ in the README.
Remove deprecated FRONTEND_AUDIENCE.


## 0.1.2 (2024-04-01)
---

### New
Add stage "demo" to APP_DB_ENGINE, APP_DB_NAME, APP_DB_URI, APP_CORS_ORIGIN, and AWS_S3_CHATBOT_ATTACHMENTS_BUCKET [FA-213].

### Changes
Updated genericsuite = "0.1.2".
".env-example" renamed to ".env.example".
AI configuration environment variables added to README.
The GenericSuite backend development scripts added to README.
License changed to ISC [FA-244].


## 0.1.1 (2024-03-19)
---

### New
Add Makefile with `build, `publish` and `publish-test` options.

### Changes
README enhanced instructions.
Updated genericsuite = "0.1.1"


## 0.1.0 (2024-03-14)
---

### New
Publish to Pypi


## 0.0.4 (2024-03-03)
---

### New
Separate BE Generic Suite to publish on PyPi [FA-219].
Initial commit as an independent repository.


## 0.0.3 (2024-02-19)
---

### New
Add LC HuggingFace chat models [FA-233].
Add Web scrapping module.


## 0.0.2 (2024-02-18)
---

### New
Implement GPT4 Vision [FA-144].
Implement Audio processing with OpenAPI whisper [FA-145].
Implement TTS-1 text-to-speech OpenAI Model [FA-210].
Implement image generator using DALL-E 3 [FA-165].
Implement Google Gemini models [FA-172].
Implement Clarifai models and embeddings [FA-182].
Add web search capability to FynBot [FA-159].
Chats stored in the DB [FA-119].
Add the "plan" attribute to the user profile [FA-200].
Add the OpenAi API key to allow free plan users to use FynBot at their own expenses [FA-201].
Add double version of GPT Functions and LC Tools [FA-211].


## 0.0.1 (2023-07-21)
---

### New
Add `ai_chatbot` endpoint to handle the AI chatbot based on OpenAI functions call [FA-93].
