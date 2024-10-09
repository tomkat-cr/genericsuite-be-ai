"""
ChatGPT functions management
"""
# from typing import Union
import json

from openai.types.chat.chat_completion_message import ChatCompletionMessage

from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import AppContext

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.web_search import (
    cac as cac_web_search,
    web_search,
    web_search_func,
)
from genericsuite_ai.lib.ai_audio_processing import (
    cac as cac_audio,
    audio_processing_text_response,
    audio_processing_text_response_func,
    text_to_audio_response,
    text_to_audio_response_func,
)
from genericsuite_ai.lib.ai_vision import (
    cac as cac_vision,
    vision_image_analyzer_text_response,
    vision_image_analyzer_text_response_func,
)
from genericsuite_ai.lib.ai_image_generator import (
    cac as cac_image_gen,
    image_generator_text_response,
    image_generator_text_response_func,
)
from genericsuite_ai.lib.ai_gpt_fn_conversations import (
    cac as cac_conversations,
    conversation_summary_tool,
    conversation_summary_tool_func,
    # get_current_date_time,
    # get_current_date_time_func,
)
from genericsuite_ai.lib.clarifai import (
    cac as cac_clarifai,
)
from genericsuite_ai.lib.huggingface import (
    cac as cac_huggingface,
)
from genericsuite_ai.lib.web_scraping import (
    cac as cac_web_scraping,
    webpage_analyzer_text_response,
    webpage_analyzer_text_response_func,
)

DEBUG = False


def get_functions_dict(
    app_context: AppContext,
):
    """
    Get the available ChatGPT functions and its callables.

    Returns:
        dict: A dictionary containing the available ChatGPT functions
        and its callable.
    """
    settings = Config(app_context)
    is_lc = settings.AI_TECHNOLOGY == 'langchain'
    if is_lc:
        # Langchain Tools
        result = {
            # "vision_image_analyzer": vision_image_analyzer_text_response,
            # "image_generator": image_generator_text_response,
            # "web_search": web_search,
            # "conversation_summary": conversation_summary_tool,
            # "audio_processing": audio_processing_text_response,
            # "text_to_audio_response": text_to_audio_response,
            # "webpage_analyzer": webpage_analyzer_text_response,
            "vision_image_analyzer_text_response": vision_image_analyzer_text_response,
            "image_generator_text_response": image_generator_text_response,
            "web_search": web_search,
            "conversation_summary_tool": conversation_summary_tool,
            "audio_processing_text_response": audio_processing_text_response,
            "text_to_audio_response": text_to_audio_response,
            "webpage_analyzer_text_response": webpage_analyzer_text_response,
            # "get_current_date_time": get_current_date_time,
        }
    else:
        # GPT Functions
        result = {
            "vision_image_analyzer": vision_image_analyzer_text_response_func,
            "image_generator": image_generator_text_response_func,
            "web_search": web_search_func,
            "conversation_summary": conversation_summary_tool_func,
            "audio_processing": audio_processing_text_response_func,
            "text_to_audio_response": text_to_audio_response_func,
            "webpage_analyzer": webpage_analyzer_text_response_func,
            # "get_current_date_time": get_current_date_time_func,
        }

    additional_callable = app_context.get_other_data('additional_function_dict')
    if additional_callable:
        result.update(additional_callable(app_context))

    if DEBUG:
        log_debug(f"GET_FUNCTIONS_DICT | is_lc: {is_lc} result: {result}")
    return result


def get_function_list(
    app_context: AppContext,
):
    """
    Get the available ChatGPT/Tools functions callable list.

    Returns:
        list: the available ChatGPT functions callables.
    """
    return list(get_functions_dict(app_context).values())


def gpt_func_appcontext_assignment(
    app_context: AppContext,
) -> None:
    """
    Assign the app_context to the ChatGPT functions.

    Args:
        app_context (AppContext): GPT Context
    """
    available_func_context = [
        cac_web_search,
        cac_audio,
        cac_vision,
        cac_image_gen,
        cac_conversations,
        cac_clarifai,
        cac_web_scraping,
        cac_huggingface,
    ]

    additional_func_context = app_context.get_other_data('additional_func_context')
    if additional_func_context:
        available_func_context.extend(additional_func_context(app_context))

    for cac in available_func_context:
        cac.set(app_context)


def run_one_function_from_chatgpt(
    app_context: AppContext,
    response_message: ChatCompletionMessage,
) -> dict:
    """
    Execute a ChatGPT function based on the response message.

    Args:
        app_context (AppContext): GPT Context
        response_message (dict): The message containing the function call
        and its arguments.

    Returns:
        The result of the function execution.
    """
    function_name = response_message.function_call.name
    function_args = json.loads(
        response_message.function_call.arguments
    )
    return run_one_function(
        app_context,
        function_name,
        function_args,
    )


def run_one_function(
    app_context: AppContext,
    function_name: str,
    function_args: dict,
) -> dict:
    """
    Execute a function based on the given function_name
    and function_args.

    Args:
        app_context (AppContext): GPT Context
        function_name (str): function name
        function_args (dict): function args

    Returns:
        The result of the function execution.
    """
    # user_lang = app_context.get_user_data().get('language', 'auto')
    question = app_context.get_other_data("question")["content"]
    available_functions = get_functions_dict(app_context)
    fuction_to_call = available_functions[function_name]
    function_response = None
    if DEBUG:
        log_debug("AI_ROF-1) run_one_function_from_chatgpt" +
                  f" | function_name: {function_name}" +
                  f" | function_args: {function_args}")

    if function_name == "vision_image_analyzer":
        function_response = fuction_to_call(
            params={
                "image_path": function_args.get("image_path"),
                "question": function_args.get("question", question),
                "other": {
                    "cid": app_context.get_other_data("cid"),
                }
            }
        )
    elif function_name == "image_generator":
        function_response = fuction_to_call(
            params={
                "question": function_args.get("question"),
                "other": {
                    "cid": app_context.get_other_data("cid"),
                }
            }
        )
    elif function_name == "web_search":
        function_response = fuction_to_call(
            params={
                "query": function_args.get("query"),
                "num_results": function_args.get("num_results", 20),
            }
        )
    elif function_name == "conversation_summary":
        function_response = fuction_to_call(
            params={
                "who_is_reading": function_args.get("who_is_reading"),
            }
        )
    elif function_name == "audio_processing":
        function_response = fuction_to_call(
            params={
                "sound_filespec": function_args.get("sound_filespec"),
                "source_lang": function_args.get("source_lang"),
            }
        )
    elif function_name == "text_to_audio_response":
        function_response = fuction_to_call(
            params={
                "input_text": function_args.get("input_text"),
                "target_lang": function_args.get("target_lang"),
                "other_options": function_args.get("other_options"),
            }
        )
    elif function_name == "webpage_analyzer":
        function_response = fuction_to_call(
            params={
                "url": function_args.get("url"),
                "question": function_args.get("question", question),
            }
        )
    # elif function_name == "get_current_date_time":
    #     function_response = fuction_to_call()

    additional_callable = app_context.get_other_data('additional_run_one_function')
    if not function_response and additional_callable:
        result = additional_callable(
            app_context,
            function_name,
            function_args,
        )
        function_response = result['function_response']

    result = {
        "function_response": function_response,
        "function_name": function_name,
        "function_args": function_args,
    }
    if DEBUG:
        log_debug('AI_ROF-2) run_one_function_from_chatgpt' +
                  f' | result: {result}')
    return result


def get_function_specs(
    app_context: AppContext,
) -> list:
    """
    Get the ChatGPT function specifications (parameters, documentation).

    Returns:
        list[dict]: A list of dictionaries containing the available
        ChatGPT functions.
    """
    if DEBUG:
        log_debug("AI_GFS-1) get_function_specs")
    result = [{
        "name": "vision_image_analyzer",
        "description": "Process the specified image and answer the" +
        " question about it. Useful when" +
        " you receive an URL with an image and need information" +
        " about its content, if that information is not in the chat,"
        " right before the URL." +
        " To call this tool, use the URL of the image." +
        " You receive that URL in the format: " +
        " \"Image URL: https://...\"",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "description": "The image path. It must be an URL",
                    "type": "string",
                },
                "question": {
                    "description": "a question about the image specified.",
                    "type": "string",
                },
            }
        },
        "required": ["image_path", "question"],
    },
    {
        "name": "image_generator",
        "description": "Process the specified text and answer the" +
        " question about it using an image generator. Useful when" +
        " you receive a question like 'give an image of a pineapple' and" +
        " you cannot find image URLs on the internet using the" +
        " 'web_search' function, or the question explicity say" +
        " 'generate an image of...'." +
        'Your answer must be exactly this function response:' +
        '\nExample of function output:' +
        ' "[Click here to see the image](https://bucketname.s3.amazonaws.com/xxx/img.png)"' +
        '\nExample of your response:' +
        ' "[Click here to see the image](https://bucketname.s3.amazonaws.com/xxx/img.png)"' +
        '\nNOTE: If it is the case, you can translate the text "[Click here to see the image]"',
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "description": "a question to generate the image.",
                    "type": "string",
                },
            }
        },
        "required": ["question"],
    },
    {
        "name": "web_search",
        "description": "Searches the web, useful when you can't" +
        " get enough or updated information from your model." +
        " e.g. when calories cannot be found in the FDA API or internal" +
        " user's ingredients or general ingredients, or you been asked" +
        " explicity to search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "description": "The search query",
                    "type": "string",
                },
                "num_results": {
                    "description": "Maximun number of results. Defaults to 20",
                    "type": "integer",
                }
            }
        },
        "required": ["query"],
    },
    {
        "name": "conversation_summary",
        "description": "Useful when you need to summarize large the user and" +
        " assistant conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "who_is_reading": {
                    "description": "Who reads this summary",
                    "type": "string",
                },
            }
        },
        "required": ["who_is_reading"],
    },
    {
        "name": "audio_processing",
        "description": "Transcribe an audio file with a audio to text" +
        " generator",
        "parameters": {
            "type": "object",
            "properties": {
                "sound_filespec": {
                    "description": "The sound local filepath or URL.",
                    "type": "string",
                },
                "source_lang": {
                    "description": "audio language. Defaults to 'auto'.",
                    "type": "string",
                },
            }
        },
        "required": ["sound_filespec"],
    },
    # google: openai chat completion api how to force the chat model to send back exactly what
    #         the gpt function returns ?
    # How to let GPT do not return any accompanying text?
    # https://community.openai.com/t/how-to-let-gpt-do-not-return-any-accompanying-text/324513/6
    {
        "name": "text_to_audio_response",
        "description": "Useful when you need to generate audio files" +
        " from a given text. Call it when the question begins with one" +
        ' of these text: "/TTS:", "/tts:"' +
        ' "Say it:", "Say it loud:", "Speak it:", "Speak it loud:",' +
        ' "Dimelo:", "Dime esto:", "Di esto en voz alta:",' +
        ' "Di este texto:", "Hablame:", "Habla esto:", "habla este texto:"' +
        ', etc.".' +
        "\n" +
        ' Your answer must be exactly this function response:\n' +
        ' Example of function output:' +
        ' "[SEND_FILE_BACK]=/tmp/openai_tts_xxxcccvvbbbbb.mp3"' +
        "\n" +
        ' Example of your response:' +
        ' "[SEND_FILE_BACK]=/tmp/openai_tts_xxxcccvvbbbbb.mp3"',
        "parameters": {
            "type": "object",
            "properties": {
                "input_text": {
                    "description": "text to speech out. Don't translate it!",
                    "type": "string",
                },
                "target_lang": {
                    "description": "target language. Defaults to user's" +
                    " preferred language.",
                    "type": "string",
                },
                "other_options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker_voice": {
                                "description": "speaker voice. Available " +
                                'options: "male" or "female".' +
                                ' Defaults to None.',
                                "type": "string",
                            },
                        },
                    },
                },
            }
        },
        "required": ["input_text"],
    }, {
        "name": "webpage_analyzer",
        "description": "Useful to answer a question about a" +
        " given webpage URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "description": "webpage URL",
                    "type": "string",
                },
                "question": {
                    "description": "question about the webpage",
                    "type": "string",
                },
            }
        },
        "required": ["url", "question"],
    # }, {
    #     "name": "get_current_date_time",
    #     "description": "Get the current date and UTC time. Useful when the" +
    #     " question refers for a specific date respect of today, e.g." +
    #     " today's date, today's calories consumed, today's meals," +
    #     " yesterday's meals, yesterday's calories",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         }
    #     },
    #     "required": [],
    }]

    additional_callable = app_context.get_other_data('additional_function_specs')
    if additional_callable:
        result.extend(additional_callable(app_context))

    # if DEBUG:
    #     log_debug('AI_GFS-2) get_function_specs' +
    #               f' | section: {section}' +
    #               f' | result: {result}')
    return result
