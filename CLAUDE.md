# CLAUDE.md

This file provides guidance to AI Coding Assistants (Claude Code, Gemini CLI, Cursor, Antigravity, etc.) when working with code in this repository.

## Project Overview

GenericSuite AI is a backend Python library providing AI-oriented features (chatbot, vision, audio, image generation, embeddings, vector search) for APIs built with FastAPI, Flask, or AWS Chalice. It extends [genericsuite-be](https://github.com/tomkat-cr/genericsuite-be) and is published to PyPI as `genericsuite_ai`.

It is part of a larger ecosystem of GenericSuite projects, including web frontends (genericsuite-fe, genericsuite-fe-ai) and mobile packages (genericsuite-mobile). For more information about the GenericSuite ecosystem, see the [GenericSuite Basecamp](https://github.com/tomkat-cr/genericsuite-basecamp).

## Commands

```bash
# Install dependencies
make install        # production deps via poetry
make install-dev    # dev deps

# Test
make test           # runs pytest with required env vars pre-set

# Single test
APP_DB_URI=fake_db_uri APP_DB_ENGINE=MONGODB APP_DB_NAME=mongo APP_NAME=test_app APP_STAGE=test APP_HOST_NAME=localhost APP_SECRET_KEY=fake_secret_key  STORAGE_URL_SEED=xyz APP_SUPERADMIN_EMAIL=fake_email GIT_SUBMODULE_LOCAL_PATH=fake_path CLOUD_PROVIDER=aws AWS_REGION=us-east-1 GET_SECRETS_ENABLED=0 CURRENT_FRAMEWORK=fastapi poetry run pytest tests/path/to/test_file.py::test_name -v

# Lint / type check
poetry run pylint genericsuite_ai/
poetry run mypy genericsuite_ai/
poetry run yapf --diff -r genericsuite_ai/  # format check

# Build & publish
make build          # creates dist/
make publish-test   # push to TestPyPI
make publish        # push to PyPI

# Dependency management
make lock           # poetry lock
make update         # poetry update
make requirements   # export requirements.txt

# SAST testing
make sast-test             # Run SAST testing
```

### Required environment variables for `make test`

`APP_DB_URI`, `APP_DB_ENGINE`, `APP_DB_NAME`, `APP_NAME`, `APP_STAGE`, `APP_HOST_NAME`, `APP_SECRET_KEY`, `STORAGE_URL_SEED`, `APP_SUPERADMIN_EMAIL`, `GIT_SUBMODULE_LOCAL_PATH`, `CLOUD_PROVIDER`, `AWS_REGION`, `GET_SECRETS_ENABLED`, `CURRENT_FRAMEWORK` — all pre-set in the Makefile's `test` target.

## Architecture

### Framework abstraction layer

The library supports FastAPI, Flask, and AWS Chalice through a three-layer pattern:

1. **Generic handler** — `genericsuite_ai/lib/ai_chatbot_endpoint.py` — framework-agnostic business logic.
2. **Framework wrappers** — `fastapilib/`, `flasklib/`, `chalicelib/` — thin adapters that call the generic handler. Each exposes the same routes (`/chatbot`, `/image_to_text`, `/voice_to_text`).
3. **Framework abstraction primitives** — `genericsuite.util.framework_abs_layer` (from the base library) provides `Response`, `BlueprintOne`, etc.

### AI execution flow

```
HTTP request
  → framework endpoint (fastapilib|flasklib|chalicelib)
    → lib/ai_chatbot_endpoint.py (auth, param parsing, AppContext)
      → lib/ai_conversations.py (load/save conversation from DB)
        → lib/ai_chatbot_main_langchain.py  (default, LangChain LCEL + ReAct)
          OR lib/ai_chatbot_main_openai.py  (direct OpenAI API)
```

### Model/provider system (`lib/ai_langchain_models.py`)

A factory that returns the appropriate LangChain chat model object based on `Config.LANGCHAIN_DEFAULT_MODEL`. Supports 30+ models across OpenAI, Anthropic, HuggingFace, Groq, Gemini, VertexAI, Bedrock, Ollama, Clarifai, IBM Watson X, and others. Model capabilities (system messages, tool use, preamble) are declared per-model and checked before use.

### Embeddings (`lib/ai_embeddings.py`)

Seven interchangeable embedding providers: OpenAI, HuggingFace, Clarifai, Bedrock, Cohere, Ollama, and a default fallback. Selected via configuration.

### Configuration (`config/config.py`)

Extends `genericsuite.config.ConfigSuperClass`. All AI provider selection, default models, language settings, and feature flags live here as environment-backed properties (e.g., `AI_TECHNOLOGY`, `LANGCHAIN_DEFAULT_MODEL`, `AI_VISION_TECHNOLOGY`, `AI_IMG_GEN_TECHNOLOGY`, `DEFAULT_LANG`).

### AppContext

A central context object threaded through all calls, carrying the authenticated user, app metadata, and `Config`. Use it to access settings and user identity rather than reading environment variables directly.

### Result pattern

All internal functions return:
```python
{"error": bool, "error_message": str, "resultset": Any}
```

### Security (`lib/ai_utilities.py`)

- `is_safe_url()` — SSRF guard: rejects private/loopback IPs, non-http(s) schemes.
- `is_safe_local_path()` — LFI guard: restricts file access to allowed directories.
Both must be called before any outbound HTTP request or local file read that uses user-supplied input.

### Billing (`models/billing/billing_utilities.py`)

Subscription/plan gating is handled here. Used to validate API key access and feature availability per user.

### Tests (`tests/`)

`tests/conftest.py` installs stubs to break circular imports that arise when FastAPI is present. Security-focused tests are under `tests/`. Coverage is reported via `pytest-cov`.

## Code style guidelines

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module-level constants, leading underscore for private methods (`_quote_identifier`, `_escape_sql_string_literal`).
- **Imports**: `typing` → stdlib → third-party → local `genericsuite.*`. No wildcard imports.
- **Type hints**: Required on all function signatures. Use `Optional`, `Union`, `Any` from `typing`. Return types always annotated.
- **Docstrings**: Triple-quoted with a one-line summary; include `Returns:` block when the shape is non-obvious.
- **Error returns**: All functions return a result dict `{'error': bool, 'error_message': str|None, 'resultset': dict|list}`. Use `get_default_resultset()` from `utilities.py` as the base. Never raise exceptions across module boundaries — catch and convert to error dict.
- **Error codes**: Tag every error message with a short positional code in brackets, e.g. `"_id is invalid [FUL3]"`. This makes log-grep easy.
- **Logging**: Use `log_debug` / `log_info` / `log_warning` / `log_error` from `app_logger.py`. Guard expensive debug calls with the module-level `DEBUG = False` flag using the walrus-operator idiom: `_ = DEBUG and log_debug(...)`.
- **Broad exceptions**: When catching `Exception` is unavoidable, suppress pylint with `# pylint: disable=broad-except` on the same line.
- **String formatting**: F-strings throughout. Multi-line strings use parenthesised concatenation, not backslash continuation.
- **Linting**: `pylint`, `flake8`, and `mypy` are all enforced. Per-file or per-line pylint disables (`C0103`, `R0902`, etc.) are acceptable when the rule conflicts with framework requirements; add a comment explaining why.

## Security considerations

### Mandatory guards before user-controlled I/O

Two functions in `lib/ai_utilities.py` **must** be called before any outbound HTTP request or local file read that uses user-supplied input:

- **`is_safe_url(url)`** — SSRF guard. Rejects non-http(s) schemes, private/loopback/link-local/multicast IPs, and unresolvable hostnames. Returns `False` on DNS failure. Must be called before any `urllib.request.urlopen()` or equivalent with user input.
- **`is_safe_local_path(path, allowed_dirs=None)`** — LFI/path-traversal guard. Resolves symlinks via `os.path.realpath()` and checks the real path against `allowed_dirs` (defaults to `/tmp` and `os.getcwd()`). Must be called before any `open(path)` with user input. Pass explicit `allowed_dirs` to restrict further.

Both functions log rejections and return `False` on error. Callers must treat `False` as a hard stop and return an error resultset.

### Where the guards are enforced

- `lib/ai_vision.py` — image URL and local path validation in `encode_image()`, `get_vision_image_url()`, `clarifai_vision_raw()`.
- `lib/ai_audio_processing.py` — URL and path validation in `process_audio_file()`, `process_audio_url()`.
- `lib/clarifai.py` — URL and path validation throughout embedding, vision, and audio helpers.

When adding new code that fetches a URL or opens a file from user input, follow the same pattern as the functions above.

### LangChain tool parameter validation

All LangChain tool calls pass parameters through `interpret_tool_params()` in `lib/ai_langchain_tools.py`, which normalises malformed JSON from LLM output and validates against a Pydantic `BaseModel` schema. Define a typed schema for every new tool; never pass raw LLM strings directly to tool logic.

### Authentication and billing gates

- Every endpoint requires a valid JWT via `AuthorizedRequest` (from `genericsuite.util.jwt`). The `AppContext` carries the authenticated user through the entire call chain.
- Feature availability per user is enforced in `models/billing/billing_utilities.py`. Check the billing gate before exposing new AI features.

### Known limitations

- **DNS rebinding**: `is_safe_url()` resolves DNS once at validation time. A time-of-check/time-of-use race or a compromised DNS record could bypass the SSRF guard. No secondary validation on connection.
- **No rate limiting on tool invocation**: LangChain ReAct agents can loop. Rely on framework/infrastructure timeouts.
- **No input size limits**: Large audio/image payloads can cause OOM. Enforce limits at the API gateway or framework layer.
- **Log leakage**: `log_debug` calls sometimes include full request/response payloads. Avoid logging raw API keys or tokens.
- **Default `allowed_dirs` includes `os.getcwd()`**: If the working directory is writable by an attacker, `is_safe_local_path()` can be bypassed. Pass a restricted explicit list when possible.

## Important Notes

- The files `AGENTS.md`, `GEMINI.md`, etc. (if present) have only a referece to `@CLAUDE.md` — edit only `CLAUDE.md`.
- Skills live in `.ai/skills/` (source of truth); symlinked under `.agents/skills/`, `.claude/skills/`, `.codex/skills/`, `.gemini/skills/`, and `.devin/skills/`.
