"""
AI Endpoints (generic)
"""
from typing import Optional, Callable, Any

# from chalice.app import Response
from genericsuite.util.framework_abs_layer import Response, BlueprintOne

from genericsuite.util.app_logger import log_debug
from genericsuite.util.jwt import AuthorizedRequest
from genericsuite.util.parse_multipart import file_upload_handler
from genericsuite.util.utilities import (
    return_resultset_jsonified_or_exception,
    get_default_resultset,
    get_query_params,
    get_file_extension,
    send_file,
)
from genericsuite.config.config_from_db import app_context_and_set_env

from genericsuite_ai.config.config import Config

from genericsuite_ai.lib.ai_chatbot_main_openai import (
    run_conversation as run_conversation_openai,
)
from genericsuite_ai.lib.ai_chatbot_main_langchain import (
    run_conversation as run_conversation_langchain,
)
from genericsuite_ai.lib.ai_audio_processing import (
    cac as cac_audio,
    audio_to_text_transcript
)
from genericsuite_ai.lib.ai_vision import (
    cac as cac_vision,
    vision_image_analyzer
)
from genericsuite_ai.lib.ai_utilities import (
    get_user_lang_code,
)
from genericsuite_ai.lib.clarifai import (
    cac as cac_clarifai,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities

DEBUG = False


def ai_chatbot_endpoint(
    request: AuthorizedRequest,
    blueprint: BlueprintOne,
    other_params: Optional[dict] = None,
    additional_callable: Optional[Callable] = None,
    sendfile_callable: Optional[Callable] = None,
    background_tasks: Optional[Any] = None,
) -> Any:
    """
    This function is the endpoint for the AI chatbot.
    It takes in a request and other parameters,
    logs the request, retrieves the user data, and runs the
    conversation with the AI chatbot.
    It then returns the AI chatbot's response.

    :param request: The request from the user.
    :param other_params: Other parameters that may be needed.
    :return: The response from the AI chatbot.
    """
    if DEBUG:
        log_debug(f'AICBEP-1) AI_CHATBOT - request: {request.to_dict()}')

    if other_params is None:
        other_params = {}

    # Set environment variables from the database configurations.
    app_context = app_context_and_set_env(request=request, blueprint=blueprint)
    if app_context.has_error():
        return return_resultset_jsonified_or_exception(
            app_context.get_error_resultset()
        )
    if additional_callable:
        additional_callable(app_context)

    # In endpoints handlers, the value must be taken first from os.environ,
    # then from Config(). e.g. 
    #   ai_technology = os.environ.get('AI_TECHNOLOGY', settings.AI_TECHNOLOGY)
    # This is because the value is set in the database
    # when Config() was already initialized in the import statement.
    # In the rest of the code called from the enpoint handler,
    # it's ok to take params from Config() directly because
    # app_context_and_set_env() has been called before in the
    # endpoint handler, then the os.environ variable is already set.
    # The other way to do it is to assing `settings = Config()` 
    # after the app_context_and_set_env(request)
    settings = Config(app_context)
    billing = BillingUtilities(app_context)

    def get_model_engine():
        model_engine = \
            f'\n | AI_TECHNOLOGY: {settings.AI_TECHNOLOGY}'

        if settings.AI_TECHNOLOGY == "langchain":
            model_engine += \
                "\n | LANGCHAIN_DEFAULT_MODEL: " + \
                f"{settings.LANGCHAIN_DEFAULT_MODEL}" + \
                get_sub_model_name()

        elif settings.AI_TECHNOLOGY == "openai":
            model_engine += \
                "\n | OPENAI_MODEL: " + \
                f"{billing.get_openai_chat_model()}"

        return model_engine

    def get_sub_model_name():
        sub_model_name = ""
        if settings.LANGCHAIN_DEFAULT_MODEL == 'clarifai':
            sub_model_name = f" ({settings.AI_CLARIFAI_DEFAULT_CHAT_MODEL})"
        elif settings.LANGCHAIN_DEFAULT_MODEL in ['chat_openai', 'openai']:
            sub_model_name += \
                "\n | OPENAI_MODEL: " + \
                f"{billing.get_openai_chat_model()}"
        return sub_model_name

    if DEBUG:
        log_debug('AICBEP-1) AI_CHATBOT_ENDPOINT - Init:' +
                  get_model_engine() +
                  f'\nRequest: {request.to_dict()}')

    ai_technology = settings.AI_TECHNOLOGY

    if ai_technology == "langchain":
        ai_chatbot_response = run_conversation_langchain(app_context)
    else:
        ai_chatbot_response = run_conversation_openai(app_context)

    if DEBUG:
        log_debug('AICBEP-2) AI_CHATBOT_ENDPOINT' +
                  f' - Response: {ai_chatbot_response}')

    if '[SEND_FILE_BACK]' in ai_chatbot_response.get('response'):
        file_to_send = ai_chatbot_response['response'].split('=')[1]
        _ = DEBUG and log_debug(
            'AICBEP-3) AI_CHATBOT_ENDPOINT' +
            f' - Sending file: {file_to_send}')
        if sendfile_callable:
            return sendfile_callable(
                file_to_send=file_to_send,
                background_tasks=background_tasks,
            )
        return send_file(
            file_to_send=file_to_send,
        )

    return return_resultset_jsonified_or_exception(
        ai_chatbot_response
    )


def vision_image_analyzer_endpoint(
    request: AuthorizedRequest,
    blueprint: BlueprintOne,
    other_params: Optional[dict] = None,
    additional_callable: Optional[Callable] = None,
    uploaded_file_path: Optional[str] = None,
) -> Response:
    """
    This endpoint receives an image file, saves it to a temporary directory
    with a uuid4 .jpg | .png filename, calls @vision_image_analyzer with
    the file path, and returns the result.

    :param request: The request containing the image file.
    :return: The text with the image analysis.
    """

    # Set environment variables from the database configurations.
    app_context = app_context_and_set_env(request=request, blueprint=blueprint)
    if app_context.has_error():
        return return_resultset_jsonified_or_exception(
            app_context.get_error_resultset()
        )
    if additional_callable:
        additional_callable(app_context)

    settings = Config(app_context)

    def get_model_engine():
        model_engine = \
            f'\n | AI_VISION_TECHNOLOGY: {settings.AI_VISION_TECHNOLOGY}'

        if settings.AI_VISION_TECHNOLOGY == "clarifai":
            model_engine += \
                "\n | AI_CLARIFAI_DEFAULT_VISION_MODEL: " + \
                f"{settings.AI_CLARIFAI_DEFAULT_VISION_MODEL}"

        elif settings.AI_VISION_TECHNOLOGY == "openai":
            model_engine += \
                "\n | OPENAI_VISION_MODEL: " + \
                f"{settings.OPENAI_VISION_MODEL}"

        return model_engine

    if DEBUG:
        log_debug('AIVTT-1) VISION_IMAGE_ANALYZER_ENDPOINT' +
                  get_model_engine() +
                  f'\nRequest: {request.to_dict()}')

    # return return_resultset_jsonified_or_exception({
    #     'error': True,
    #     'error_message': 'TEMPORARY ON PURPOSE ERROR',
    # })

    if other_params is None:
        other_params = {}

    response = get_default_resultset()
    query_params = get_query_params(request)

    if "question" not in query_params or not query_params.get("question"):
        response['error'] = True
        response['error_message'] = "A question about the image should" + \
                                    " be provided"
    if "cid" not in query_params:
        response['error'] = True
        response['error_message'] = "A conversartion id should be provided"
    if "file_name" not in query_params:
        response['error'] = True
        response['error_message'] = "A file name should be provided"

    if response['error']:
        return return_resultset_jsonified_or_exception(response)

    cac_vision.set(app_context)
    cac_clarifai.set(app_context)

    return file_upload_handler(
        app_context=app_context,
        p={
            "uploaded_file_path": uploaded_file_path,
            "extension": get_file_extension(query_params["file_name"]),
            "handler_function": vision_image_analyzer,
            "unique_param_name": "params",
            "file_path_param_name": "image_path",
            "other_params": {
                "params": {
                    "question": query_params["question"],
                    "other": {
                        "file_name": query_params["file_name"],
                        "cid": query_params["cid"],
                    }
                }
            }
        },
    )


def transcribe_audio_endpoint(
    request: AuthorizedRequest,
    blueprint: BlueprintOne,
    other_params: Optional[dict] = None,
    additional_callable: Optional[Callable] = None,
    uploaded_file_path: Optional[str] = None,
) -> Response:
    """
    This endpoint receives an audio file, saves it to a temporary directory
    with a uuid4 .mp3 filename, calls @audio_to_text_transcript with
    the file path, and returns the result.

    :param request: The request containing the audio file.
    :return: The transcription result.
    """

    # Set environment variables from the database configurations.
    app_context = app_context_and_set_env(request=request, blueprint=blueprint)
    if app_context.has_error():
        return return_resultset_jsonified_or_exception(
            app_context.get_error_resultset()
        )
    if additional_callable:
        additional_callable(app_context)

    settings = Config(app_context)

    def get_model_engine():
        model_engine = \
            '\n | AI_AUDIO_TO_TEXT_TECHNOLOGY: ' + \
            f'{settings.AI_AUDIO_TO_TEXT_TECHNOLOGY}'

        if settings.AI_AUDIO_TO_TEXT_TECHNOLOGY == "clarifai":
            model_engine += \
                "\n | AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL: " + \
                f"{settings.AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL}"

        elif settings.AI_AUDIO_TO_TEXT_TECHNOLOGY == "openai":
            model_engine += \
                "\n | OPENAI_VOICE_MODEL: " + \
                f"{settings.OPENAI_VOICE_MODEL}"

        return model_engine

    if DEBUG:
        log_debug('AIVTT-1) TRANSCRIBE_AUDIO_ENDPOINT' +
                  get_model_engine() +
                  f'\nRequest: {request.to_dict()}')

    # return return_resultset_jsonified_or_exception({
    #     'error': True,
    #     'error_message': 'TEMPORARY ON PURPOSE ERROR',
    # })

    if other_params is None:
        other_params = {}

    query_params = get_query_params(request)
    cac_audio.set(app_context)
    cac_clarifai.set(app_context)

    file_extension = query_params.get("extension", "mp3")

    source_lang = query_params.get("source_lang", "auto")
    if source_lang == "get_user_lang":
        source_lang = get_user_lang_code(app_context)
    other_options = query_params.get("other", '')

    return file_upload_handler(
        app_context=app_context,
        p={
            "uploaded_file_path": uploaded_file_path,
            "extension": file_extension,
            "handler_function": audio_to_text_transcript,
            "unique_param_name": "params",
            "file_path_param_name": "sound_filespec",
            # "delete_file_after_processing": False,
            "other_params": {
                "params": {
                    "source_lang": source_lang,
                    "other_options": other_options,
                },
            }
        },
    )
