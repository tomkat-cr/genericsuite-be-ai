"""
Langchain tools
"""
from typing import Union, Any
import json

from langchain.schema.messages import (
    HumanMessage, SystemMessage, AIMessage   #  AnyMessage
)
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import is_under_test

# from genericsuite_ai.config.config import Config

DEBUG = False
INPUT_KEY = "stop"


class ExistingChatMessageHistory(BaseChatMessageHistory):
    """
    Creates a chat message history from a existing conversation.
    """
    def __init__(self, messages: list[BaseMessage]):
        self.messages = []
        self.add_messages(messages=messages)

    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store (e.g. HumanMessage,
            AIMessage).
        """
        self.messages.append(message)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of Messages object to the store.

        Args:
            messages: A list[BaseMessage] objects to store.
        """
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """Remove all messages from the store"""
        self.messages = []


def messages_to_langchain_fmt(messages: list, message_format: str = None
                              ) -> list:
    """
    Prepare messages for LangChain, converting messages from ChatGPT to
    Langchain format. Example:
    messages = messages_to_langchain_fmt(messages, "tuple")

    Args:
        messages (list): The messages to prepare in ChatGTP format.
        message_format (str): the message format to return.
            Options: "class", "tuple", "text". Defaults to "class".

    Returns:
        list: The messages converted to the desired message format.
    """
    message_format = message_format or "class"
    if message_format == "class":
        messages = [
            HumanMessage(content=v["content"]) if v["role"] == "user" else
            SystemMessage(content=v["content"]) if v["role"] == "system" else
            AIMessage(content=v["content"]) if v["role"] == "assistant" else
            (
                HumanMessage(content=v["attachment_url"])
                if v.get("fix_role", "user") else
                AIMessage(content=v["attachment_url"])
            ) if v["role"] == 'attachment' else
            None
            # Throws Cannot instantiate typing.Union, <class 'TypeError'> [AIRCLC-E020]
            # AnyMessage(content=v["content"])
            for v in messages
        ]
    else:
        # message_format == "tuple"
        messages = [
            ("human", v["content"]) if v["role"] == "user" else
            ("system", v["content"]) if v["role"] == "system" else
            ("ai", v["content"]) if v["role"] == "assistant" else
            None
            for v in messages
        ]
    return messages


def messages_to_langchain_fmt_text(messages: list) -> str:
    """
    Prepare string chat_history, suitable for LLM's prompt, not Chat Models.

    Args:
        messages (list): The messages to prepare in ChatGTP format.

    Returns:
        str: The messages converted to text.
    """
    # String chat_history, suitable for LLM's prompt, not Chat Models
    messages = "\n".join([
        f'Human: {v["content"]}' if v["role"] == "user" else
        f'System: {v["content"]}' if v["role"] == "system" else
        f'AI: {v["content"]}' if v["role"] == "assistant" else
        None
        for v in messages
    ])
    return messages


def get_conversation_buffer(messages: list) -> ConversationBufferMemory:
    """
    Get the Conversation Buffer Memory filled with the chat history

    Args:
        messages (list): chat message history in ChatGPT format:
        {"role": "user/assistant", "content": "..."}

    Returns:
        ConversationBufferMemory: Buffer for storing conversation memory.
    """
    chat_memory = ExistingChatMessageHistory(
        messages=messages_to_langchain_fmt(messages)
    )
    # https://python.langchain.com/v0.1/docs/modules/memory/
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_memory,
    )
    # memory = load_conversarion_messages_to_buffer(memory, messages)
    return memory


def load_conversarion_messages_to_buffer(
    memory: ConversationBufferMemory,
    messages: list,
) -> ConversationBufferMemory:
    """
    Load the conversation messages to the buffer
    TODO: probably it's not needed...
    """
    inputs = ""
    outputs = ""
    for message in messages:
        if message["role"] == "system":
            continue
        if message["role"] == "user":
            if inputs:
                # Two or more user inputs in a roll...
                # this could happen if there's was any error receiving the
                # assistant response with no error reporting in the chat.
                inputs += f'. {message["content"]}'
            else:
                inputs = message["content"]
        else:
            if outputs:
                # Two or more assistant responses in a roll...
                # This is a inconsistencia... should not happen....
                outputs += f'. {message["content"]}'
            else:
                outputs = message["content"]
        if inputs and outputs:
            memory.save_context(
                inputs={INPUT_KEY: inputs},
                outputs={INPUT_KEY: outputs},
            )
    if inputs and not outputs:
        memory.save_context(
            inputs={INPUT_KEY: inputs},
            outputs={INPUT_KEY: ""},
        )
    return memory


def interpret_tool_params(
    tool_params: Any,
    schema: Any = None,
    first_param_name: str = None) -> dict:
    """
    Interpret tool params from LangChain. It's suppose to be a dict
    and some times comes as a string.

    Args:
        tool_params (Any): The tool params to interpret.
        schema (Any): the schema to validate the tool params with.
            Defaults to None.
        first_param_name (str): first param name, for those cases when the
            Agent pass only a string (no JSON) and it'll be assigned as
            the 1st required parameter

    Returns:
        dict: The interpreted tool params.
    """

    self_debug = DEBUG or is_under_test()
    # self_debug = True
    _ = self_debug and \
        log_debug("INTERPRET_TOOL_PARAMS" +
            f"\n | tool_params: {tool_params}" + 
            f"\n | tool_params type: {type(tool_params)}"
            f"\n | schema: {schema}" +
            f"\n | first_param_name: {first_param_name}")
    if not tool_params:
        return tool_params
    if isinstance(tool_params, dict) and 'params' in tool_params:
        tool_params = tool_params['params'].copy()
    if isinstance(tool_params, str):
        # if tool_params has a "\n" and a additional line with anything
        # like "```" or "Observation", apart the last } and \n,
        # that extra info must be removed...
        for _ in range(3):
            if "\n" in tool_params:
                # Remove last line if it exists after a newline
                tool_params = tool_params.rsplit("\n", 1)[0]
        if tool_params.endswith("```"):
            tool_params = tool_params[:-3]  # Remove the trailing characters
        if tool_params.endswith("Observation"):
            tool_params = tool_params[:-11]  # Remove the trailing characters
        try:
            tool_params = json.loads(tool_params)
        except json.decoder.JSONDecodeError:
            if not first_param_name:
                first_param_name = "single_param"
            tool_params = {first_param_name: tool_params}
    if schema:
        tool_params = schema(**tool_params)
    return tool_params
