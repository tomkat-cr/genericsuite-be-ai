# CHANGELOG

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/) and [Keep a Changelog](http://keepachangelog.com/).



## Unreleased
---

### New

### Changes

### Fixes

### Breaks


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
