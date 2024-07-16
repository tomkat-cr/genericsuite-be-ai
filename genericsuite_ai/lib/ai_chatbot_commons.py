"""
Chatbot run conversation common functions.
"""
from typing import Union
import json

from genericsuite.util.utilities import get_default_resultset, get_request_body
from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import AppContext
from genericsuite.constants.const_tables import get_constant

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.ai_conversations import (
    init_conversation_db,
    update_conversation_db
)
from genericsuite_ai.lib.ai_utilities import (
    report_error,
    get_response_lang,
    get_assistant_you_are,
)
from genericsuite_ai.lib.ai_gpt_fn_conversations import (
    get_current_date_time
)

DEBUG = False


def get_role(v):
    """
    Returns the role from the conversation item and show debug
    """
    if DEBUG:
        log_debug("")
        log_debug(f"v = {v}")
    return v["role"]


def prepare_conversation(prev_conv: list,
                         question: Union[list, None] = None):
    """
    Filter and convert messages in the conversation for GPT chat
    """
    if question:
        if isinstance(question, dict):
            question = [question]
    else:
        question = []
    conversation = prev_conv + question
    conversation = [
        v for v in conversation
        if v.get("role") not in ['ignore']
    ]
    conversation = [
        v for v in conversation
        if v.get("role") not in ['attachment'] or
        (v.get("role") == "attachment" and
         v.get("attachment_url"))
    ]
    conversation = [
        {
            "role":  get_role(v) if v["role"] != "attachment"
            else v.get("fix_role", "user"),
            "content": (
                v["content"] if v["role"] != "attachment"
                else (
                    f'Image URL: {v["attachment_url"]}' +
                    ". The description for this URL is in the" +
                    " previous message. The URL is required when you cannot" +
                    " figure out the image description, so you can send that" +
                    " URL to vision_image_analyzer()"
                ) if v.get("sub_type", "image").startswith("image") else (
                    f'File URL: {v["attachment_url"]}' +
                    ". The description for this URL is in the" +
                    " previous message. The URL is required when you cannot" +
                    " figure out the file description"
                )
            )
        } for v in conversation
    ]
    return conversation


def get_starting_prompt(app_context):
    """ Starting prompt for the assistant """
    bottom_line_prompt = get_constant("AI_PROMPT_TEMPLATES", "BOTTOM_LINE", "")
    # return ("You are a helpful assistant." +
    return (f"You are {get_assistant_you_are(app_context)}." +
            "\n" +
            "When you call a GPT function, and the function" +
            " description includes the phrase 'Your answer must be exactly" +
            " this function response', or the GPT function response" +
            " begins with '[SEND_FILE_BACK]', then your final response" +
            " will be exactly that function response, with no additions" +
            " or changes." +
            # " E.g." +
            # ' Function output example:' +
            # ' "[SEND_FILE_BACK]=/tmp/tts.mp3"' +
            # '. Your response:' +
            # ' "[SEND_FILE_BACK]=/tmp/tts.mp3"',
            "\n" +
            f"For date references, {get_current_date_time({})}." +
            "\n" +
            bottom_line_prompt +
            "\n" +
            "Your final response will be in" +
            f" '{get_response_lang(app_context)}'" +
            " despite of the question's original language." +
            "")


def get_llm_lib_data(app_context):
    """
    Get LLM library data (message code and lib name) for log_debug messages.
    """
    settings = Config(app_context)
    response = {}
    response["name"] = settings.AI_TECHNOLOGY.upper()
    if settings.AI_TECHNOLOGY == "langchain":
        response["code"] = "AIRCLC"
    else:
        response["code"] = "AIRCOA"
    return response


def start_run_conversation(app_context: AppContext,
                           initial_prompt: bool = True):
    """
    Start conversation run.

    Args:
        request: The request object.
        app_context (AppContext): The run_conversation's GPT Context.
        If it's None, creates a new one from the request.
        Defaults to None.

    Returns:
        dict: returs a dict ith these entries:
            "app_context": the GPT context.
            "query_params": the body parametes of the request,
            "messages": a lisr of messages formatted for ChatGPT.
            "conv_response" (dict) a standard response with this structure:
                response (str): = the message answered by the model
                error (bool): = True if there was a error
                error_message (str): = eventual error message
                cid (str): coversation ID.
    """
    lld = get_llm_lib_data(app_context)
    request = app_context.get_request()
    query_params = get_request_body(request)
    conv_response = get_default_resultset()

    # conversation is a json string comes from the frontend
    # must be converted with json to a list of strings
    # before sending to the endpoint
    if not query_params.get('conversation'):
        conv_response["error"] = True
        conv_response["error_message"] = "No conversation provided" + \
                                         f" [AICC-E010] Stack: {lld['code']}"
        return report_error(conv_response)

    if DEBUG:
        log_debug(f"{lld['code']}-1) RUN_CONVERSATION_{lld['name']}" +
                  f'\nquery_params: {query_params}')

    try:
        conversation = json.loads(query_params.get('conversation'))
    except TypeError as err:
        conv_response["error"] = True
        conv_response["error_message"] = \
            f"Invalid conversation format: {str(err)}" + \
            f" [AICC-E011] Stack: {lld['code']}"
        return report_error(conv_response)

    # Extract the user's question from the last element received
    conv_response["question"] = conversation[-1]
    if DEBUG:
        log_debug(f"{lld['code']}-2) RUN_CONVERSATION_{lld['name']}" +
                  f'\nuser_input: {conv_response["question"]}')

    # Store the question for GPT function / LangChain Tools
    app_context.set_other_data("question", conv_response["question"])

    # Get the conversation ID and data
    conv_response = init_conversation_db(
        app_context=app_context,
        conv_response=conv_response,
        conversation_id=query_params.get("cid"),
    )
    if conv_response["error"]:
        return report_error(conv_response)

    # Convert the conversation to a ChaGPT readable format
    conversation = prepare_conversation(
        prev_conv=app_context.get_other_data("conv_data")["messages"],
        # question=conv_response["question"],
    )

    # Step 1: send the conversation and available functions to GPT
    if initial_prompt:
        messages = [
            {"role": "system", "content": get_starting_prompt(app_context)},
        ]
    else:
        messages = []
    messages += conversation
    if DEBUG:
        log_debug(f"{lld['code']}-3) RUN_CONVERSATION_{lld['name']}" +
                  f"\nmessages: {messages}")

    return {
        "app_context": app_context,
        "conv_response": conv_response,
        "query_params": query_params,
        "messages": messages,
        "error": False,
        "error_message": None,
    }


def finish_run_conversation(
    app_context: AppContext,
    conv_response: dict,
    messages: list,
):
    """
    Finish conversation run.

    Args:
        conv_response:
        app_context (AppContext): The run_conversation's GPT Context.
        conv_response (dict): The conversation response.
        messages (list): message list for debuggin purposes.

    Returns:
        dict: returs a dict with these entries:
            question (str): the usre's question.
            response (str): the message answered by the model
            error (bool): True if there was a error
            error_message (str): eventual error message
            cid (str): coversation ID
    """
    settings = Config(app_context)
    lld = get_llm_lib_data(app_context)

    if DEBUG:
        log_debug(f"\n{lld['code']}-4) RUN_CONVERSATION_{lld['name']}" +
                  '\nFinal Response:' +
                  f' {conv_response.get("response")}\n')

    if conv_response["error"] and settings.DEBUG:
        # To include the error message in the conversation
        # (only in dev/qa environments)
        conv_response["response"] = conv_response["error_message"]
    elif not conv_response.get("response"):
        conv_response["response"] = \
            "Sorry, I couldn't get responses from the AI model." + \
            f" [AICC-E040] Stack: {lld['code']}"

    conv_response = update_conversation_db(
        app_context=app_context,
        conv_response=conv_response,
    )

    if DEBUG:
        log_debug(f"\n{lld['code']}-5) RUN_CONVERSATION_{lld['name']}" +
                  f"\nmessages: {messages}" +
                  f'\nconv_response: {conv_response}\n')

    return report_error(conv_response)
