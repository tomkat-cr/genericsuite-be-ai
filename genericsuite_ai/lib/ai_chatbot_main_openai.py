"""
This module contains the implementation of the AI Chatbot API.
"""
from openai import OpenAI

from genericsuite_ai.lib.ai_utilities import (
    report_error,
)
from genericsuite_ai.lib.ai_chatbot_commons import (
    start_run_conversation,
    finish_run_conversation,
)
from genericsuite_ai.lib.ai_gpt_functions import (
    get_function_specs,
    run_one_function_from_chatgpt,
    gpt_func_appcontext_assignment,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities
from genericsuite_ai.config.config import Config
from genericsuite.util.app_context import AppContext
from genericsuite.util.app_logger import log_debug


DEBUG = False


def run_conversation(app_context: AppContext) -> dict:
    """
    OpenAI conversation run.

    Args:
        app_context (AppContext): the application context.

    Returns:
        dict: a standard response with this structure:
            response (str): = the message answered by the model
            error (bool): = True if there was a error
            error_message (str): = eventual error message
            cid (str): coversation ID
    """
    settings = Config(app_context)
    start_response = start_run_conversation(app_context=app_context, initial_prompt=True)
    if start_response["error"]:
        return report_error(start_response)
    conv_response = start_response["conv_response"]
    # query_params = p["query_params"]
    messages = start_response["messages"]

    gpt_function = {}

    # Get the model according to user's billing plan
    billing = BillingUtilities(app_context)
    openai_api_key = billing.get_openai_api_key()
    if billing.is_free_plan() and not openai_api_key:
        conv_response["error"] = True
        conv_response["error_message"] = "You must specify your OPENAI_API_KEY in your" + \
            " profile or upgrade to a paid plan [AIRCOA-E010]"
    chat_model = billing.get_openai_chat_model()

    # Prepare application context for all GPT Functions
    gpt_func_appcontext_assignment(app_context)

    if DEBUG:
        log_debug('AIRCOA-3) CHAT_GPT RUN_CONVERSATION_OPENAI' +
                  f' messages: {messages}')

    try:
        client = OpenAI(
            api_key=openai_api_key,
        )
        response = client.chat.completions.create(
            model=chat_model,
            temperature=float(settings.OPENAI_TEMPERATURE),
            messages=messages,
            functions=get_function_specs(app_context),
            function_call="auto"
        )
        gpt_function["response_message"] = response.choices[0].message
        if DEBUG:
            log_debug('AIRCOA-3.5) CHAT_GPT client.chat.completions.create()' +
                      f' RESPONSE: {response}')
    except Exception as err:
        conv_response["error"] = True
        conv_response["error_message"] = f"{str(err)} [AIRCOA-E020]"
        return report_error(conv_response)

    # Step 2: check if GPT wanted to call a function
    if not gpt_function["response_message"].function_call:
        # No function call, just return the response
        final_response = gpt_function["response_message"]
    else:
        # Step 3: call the function
        # Note: the JSON response may not always be valid;
        # be sure to handle errors

        gpt_function["run_resp"] = run_one_function_from_chatgpt(
            app_context=app_context,
            response_message=gpt_function["response_message"],
        )
        gpt_function["response"] = gpt_function["run_resp"]["function_response"]
        gpt_function["name"] = gpt_function["run_resp"]["function_name"]

        # Step 4:
        # send the info on the function call and function response to GPT

        # Extend conversation with assistant's reply
        messages.append(gpt_function["response_message"])
        # Extend conversation with function response
        messages.append(
            {
                "role": "function",
                "name": gpt_function["name"],
                "content": gpt_function["response"],
            }
        )
        try:
            # get a new response from GPT where
            # it can see the function response
            gpt_function["second_response"] = client.chat.completions.create(
                model=chat_model,
                temperature=float(settings.OPENAI_TEMPERATURE),
                messages=messages
            )
        except Exception as err:
            conv_response["error"] = True
            conv_response["error_message"] = f"{str(err)} [AIRCOA-E030]"
            return report_error(conv_response)

        # Return second_response
        final_response = gpt_function["second_response"].choices[0].message

    if DEBUG:
        log_debug("\nAIRCOA-3.10) CHAT_GPT response (after function" +
                  f" call): {final_response}\n")

    conv_response["response"] = final_response.content
    return finish_run_conversation(
        app_context=app_context,
        conv_response=conv_response,
        messages=messages,
    )
