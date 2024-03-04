"""
GPT Functions / Langchain Tools to handle embeddings
"""
from typing import Any
from datetime import datetime

from langchain.agents import tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ReadOnlySharedMemory

from genericsuite.util.app_context import CommonAppContext
from genericsuite.util.utilities import log_debug
from genericsuite_ai.lib.ai_langchain_tools import (
    get_conversation_buffer,
    interpret_tool_params,
)
from genericsuite_ai.lib.ai_langchain_models import (
    get_model_obj,
    get_model_load_error,
)
from genericsuite_ai.lib.ai_utilities import (
    gpt_func_error,
)

DEBUG = False
cac = CommonAppContext()


@tool
def conversation_summary_tool(params: Any = None) -> Any:
    """
Useful when you need to summarize large Human and Assistant conversations.
Args: params (dict): Tool parameters. Must have: "who_is_reading" (str): Who reads this summary
    """
    return conversation_summary_tool_func(params)


def conversation_summary_tool_func(params: Any = None) -> Any:
    """
    Summarize large Human and Assistant conversations.

    Args:
        params (dict): Tool parameters.
            Must have: "who_is_reading" (str): Who reads this summary

    Returns:
        str: Summary of the conversation or [FUNC+ERROR] {error_message}
    """
    template = """This is a conversation between a human and a bot:

    {chat_history}

    Write a summary of the conversation for {input}:
    """
    params = interpret_tool_params(tool_params=params,
                                   first_param_name="who_is_reading")
    if DEBUG:
        log_debug(f"AI_CNS_1) CONVERSATION_SUMMARY | template: {template}")
    prompt = PromptTemplate(input_variables=["input", "chat_history"],
                            template=template)
    # memory = ConversationBufferMemory(memory_key="chat_history")
    messages = cac.app_context.get_other_data("conv_data")["messages"]
    memory = get_conversation_buffer(messages=messages)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    model = get_model_obj(cac.get())
    if not model:
        error_message = get_model_load_error(
            app_context=cac.app_context,
            error_code="AI_CNS-E010",
        )
        result = gpt_func_error(error_message)
        return result
    summary_chain = LLMChain(
        # llm=OpenAI(),
        llm=model,
        prompt=prompt,
        verbose=True,
        # use the read-only memory to prevent the tool from modifying
        # the memory
        memory=readonlymemory,
    )
    try:
        result = summary_chain.invoke(
            input=params.get("who_is_reading", 'Assistant'))
    except Exception as error:
        result = gpt_func_error(error)
    if DEBUG:
        log_debug(f"AI_CNS_2) CONVERSATION_SUMMARY | result: {result}")
    return result


@tool
def get_current_date_time(params: Any = None) -> str:
    """
Useful when you need to get the current date and UTC time when a question refers for a specific date respect of today, e.g. today's date, today's calories consumed, today's meals, yesterday's meals, yesterday's calories.
    """
    return get_current_date_time_func(params)


def get_current_date_time_func(params: Any = None) -> str:
    """
    Get the current date and UTC time when a question refers for a specific date
    respect of today, e.g. today's date, today's calories consumed, today's
    meals, yesterday's meals, yesterday's calories.
    """
    params = interpret_tool_params(tool_params=params)
    result = "today is" + \
             f" {datetime.utcnow().strftime('%Y-%m-%d')}" + \
             ", and the current UTC time is" + \
             f" {datetime.utcnow().strftime('%H:%M:%S')}"
    return result
