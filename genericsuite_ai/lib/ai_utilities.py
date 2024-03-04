"""
AI general utilities
"""
from typing import Union, Optional
import json

from genericsuite.util.app_logger import (
    log_debug,
    log_error,
)
from genericsuite.util.app_context import AppContext
from genericsuite.constants.const_tables import get_constant

from genericsuite_ai.config.config import Config

DEBUG = False


def standard_gpt_func_response(
    result: dict,
    action_description: str,
    include_resultset: bool = False,
    additional_msg: Union[str, None] = None,
    add_success_msg: bool = True,
    json_result: bool = True,
) -> str:
    """
    Generates a standard response for ChatGPT from function calls.
    'result' must be a resultset dictionary, with "error", "error_message"
    and "resultset" entries.
    If result["error"] is True, the response message will be
    "ERROR: " + error_message.
    Otherwise the response message will be the action_description +
    " was successful." and if include_resultset is True,
    the json string of result["resultset"] will be appended.

    Args:
        result (dict): The result of the function call as a resultset.
        action_description (str): Description of the action acomplished.
            E.g. "Element X creation" or "Item Y retrieval.
        include_resultset: True to include in the response the
            result["resultset"] element as a json string. Defaults to False.
        additional_msg: message to place before the results.
            Defaults to "The resulting values are".
        add_success_msg (bool): If True, adds "was successful" to the
            response if result have items, otherwise add "has no items".
            Defaults to True.
        json_result (bool): True if the result["resultset"] attribute needs to
            be json.dumped. Defaults to True.

    Returns:
        str: The response message.
    """
    result_have_items = True
    include_resultset_int = include_resultset
    if include_resultset_int:
        # Verify if there are items in the resultset
        # to fine tune the final message with the phrase
        # "has no items"
        if json_result:
            results_qty = len(result["resultset"])
        else:
            try:
                results_qty = len(json.loads(result["resultset"]))
            except Exception as err:
                results_qty = 1
                log_error('AI_SGFR-E010) STANDARD_GPT_FUNC_RESPONSE' +
                          f' | error: {err}')
        result_have_items = results_qty > 0
        _ = DEBUG and log_debug('AI_SGFR-1) STANDARD_GPT_FUNC_RESPONSE' +
            f'\n | type(result["resultset"]): {type(result["resultset"])}' +
            f'\n | result["resultset"]: {result["resultset"]}' +
            f'\n | results_qty: {results_qty}' +
            f'\n | result_have_items: {result_have_items}' +
            '\n')
    if additional_msg is None:
        additional_msg = "The resulting values are "
    if result["error"]:
        response = gpt_func_error(result['error_message'])
    else:
        response = action_description
        if add_success_msg:
            msg_to_add = " " + ("was successful" if result_have_items
                else "has no items") + ". "
            response += msg_to_add
            if not result_have_items:
                include_resultset_int = False
        if include_resultset_int:
            response += additional_msg
            if json_result:
                response += " (in JSON format): " + \
                    json.dumps(result["resultset"])
            else:
                response += result["resultset"]
    # _ = DEBUG and log_debug('AI_SGFR-2) STANDARD_GPT_FUNC_RESPONSE' +
    #                         f' | result: {result}')
    return response


def get_user_lang_code(app_context: AppContext) -> str:
    """
    Get the prefferred language code for the user.
    If the user has not set a preference, the default language code is
    returned.

    Language ISO codes:
    https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
    en = English
    es = Spanish

    Returns:
        str: The preferred language code for the current user.
    """
    settings = Config(app_context)
    user_lag_code = app_context.get_user_data().get('language',
        settings.DEFAULT_LANG)
    if str(user_lag_code).strip() in ['', 'None']:
        user_lag_code = settings.DEFAULT_LANG
    _ = DEBUG and log_debug('AI_GULC-1) get_user_lang_code' +
                            f' | user_lag_code: {user_lag_code}')
    return user_lag_code


def get_response_lang(app_context: AppContext) -> str:
    """
    Get the desired response language description based on the user's
    preference, in plain english.
    """
    lang_desc = get_response_lang_desc(app_context)
    lang_name_trans = {
        'Español': 'Spanish',
    }
    _ = DEBUG and log_debug('AI_GRL-1) get_response_lang' +
        f' | lang_desc: {lang_desc}')
    return lang_name_trans.get(lang_desc, lang_desc)


def get_response_lang_code(app_context: AppContext) -> str:
    """
    Get the desired response language code based on the user's
    preference.
    """
    user_lang_code = get_user_lang_code(app_context)
    _ = DEBUG and log_debug('AI_GRLC-1) get_response_lang_code' +
                            f' | user_lang_code: {user_lang_code}')
    return user_lang_code


def get_response_lang_desc(app_context: AppContext) -> str:
    """
    Get the desired response language description based on the user's
    preference.
    """
    settings = Config(app_context)
    user_lang_code = get_user_lang_code(app_context)
    lang_desc = get_constant("LANGUAGES", user_lang_code,
        settings.DEFAULT_LANG)
    _ = DEBUG and log_debug('AI_GRLD-1) get_response_lang_desc' +
        f' | user_lang_code: {user_lang_code} | lang_desc: {lang_desc}')
    return lang_desc


def standard_msg(content: str, role: str = "user",
    other: Optional[Union[dict, None]] = None) -> dict:
    """
    Create a standard message object.

    Args:
        content (str): The content of the message.
        role (str, optional): The role of the message. Defaults to "user".
        other (dict, optional): additional data to be added to to message.

    Returns:
        dict: The message object with role and content.
    """
    message = {"role": role, "content": content}
    if other:
        message.update(other)
    return message


def report_error(conv_response: dict) -> dict:
    """
    Reports a Chatbot Error, printing the conv_response dict.
    """
    if not conv_response["error"]:
        return conv_response
    if not conv_response.get("response"):
        conv_response["response"] = \
            "Sorry, I couldn't get responses from the AI model [AIRE-E010]."
    log_error(f">>> AI MODEL ERROR | conv_response: {conv_response}")
    return conv_response


def gpt_func_error(error_message: str) -> str:
    """ Returns a standard GPT Function/Langchain Error """
    return f"[FUNC+ERROR] {error_message}"


def get_assistant_you_are(app_context: AppContext) -> str:
    """
    Get the system initial prompt.

    Returns:
        str: The system initial prompt with the assistant name and purpose.
    """
    settings = Config(app_context)
    assistant_name = settings.AI_ASSISTANT_NAME
    you_are_prompt = get_constant("AI_PROMPT_TEMPLATES", "ASSISTANT_YOU_ARE",
                                  "a helpful assistant")
    return you_are_prompt.replace('{assistant_name}', assistant_name)
