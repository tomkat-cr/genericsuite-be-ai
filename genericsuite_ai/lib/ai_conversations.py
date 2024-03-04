"""
Conversations handling
"""
from typing import Union

from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import get_default_resultset
from genericsuite.util.generic_db_middleware import (
    add_item_to_db,
    modify_item_in_db,
    get_item_from_db,
)
from genericsuite.util.app_context import AppContext
from genericsuite_ai.lib.ai_utilities import (
    standard_msg,
)

DEBUG = False


def load_conversation(
    app_context: AppContext,
    conversation_id: str,
) -> dict:
    """
    Add a new conversation to the database.

    Args:
        app_context: GPT Context
        id (str): ID of the conversation to be retrieved.

    Returns:
        dict: the conversation data.
    """
    return get_item_from_db(
        app_context=app_context,
        json_file='ai_chatbot_conversations',
        entry_name="_id",
        entry_value=conversation_id
    )


def create_conversation(
    app_context: AppContext,
    new_item_data: dict,
) -> dict:
    """
    Add a new conversation to the database.

    Args:
        app_context: GPT Context
        new_item_data (dict): Data for the new conversation.

    Returns:
        dict: Operation result. e.g. ['resultset']['_id'].
        See: add_item_to_db()
    """
    return add_item_to_db(
        app_context=app_context,
        json_file='ai_chatbot_conversations',
        data=new_item_data,
    )


def update_conversation(
    app_context: AppContext,
    item_data: dict,
) -> dict:
    """
    Update an existing conversation in the database.

    Args:
        app_context: GPT Context
        item_data (dict): Data for the conversation to be updated.

    Returns:
        dict: Operation result. See: modify_item_in_db()
    """
    return modify_item_in_db(
        app_context=app_context,
        json_file='ai_chatbot_conversations',
        data=item_data,
    )


def get_content_from_msg(message: Union[list, dict]) -> str:
    """
    Get the content from a message object.

    Args:
        message (Union[list, dict]): The message object.

    Returns:
        str: The content of the message.
    """
    if isinstance(message, dict) and "content" in message:
        return message["content"]
    text = [v for v in message if v.get("type") == "text"]
    if text:
        return text[0]["text"]
    return ""


def interpret_msg(message: Union[list, dict]) -> list:
    """
    Interpret the message object.
    If the message is a dict and have the content element, returns it.
    If the message is a list and the message.type is text, returns its value.
    If the message is a list and the message.type is file_name,
    returns the attachment data (file name, URL).

    Args:
        message (Union[list, dict]): The message object.

    Returns:
        list: A list with the message object(s)
        in a ChatGPT-like format {role:..., content:...}, and
        attachments format if any.
    """

    result = []
    if isinstance(message, dict) and "content" in message:
        result.append(standard_msg(
            message["content"], message.get("role", "user"))
        )
    elif isinstance(message, list):
        result.extend([
            standard_msg(v["text"], v.get("role", "user"))
            if v.get("type") == "text"
            else standard_msg(
                content=f'```File Attachment: {v["file_name"]}```',
                role="attachment",
                other={
                    "fix_role": v["role"],
                    "sub_type": v.get("sub_type", "file"),
                    "attachment_url": v["attachment_url"],
                    "final_filename": v["final_filename"],
                },
            )
            for v in message
            if v.get("type") in ["text", "file_name"]
        ])

    if DEBUG:
        log_debug("")
        log_debug(">> interpret_msg():")
        log_debug(f"message: {message}")
        log_debug(f"result: {result}")
    return result


def is_conversation_creation(conversation_id: str) -> bool:
    """
    Returns True if it's a conversation creation, if the
    conversation_id is null
    """
    is_creation = not conversation_id or conversation_id == 'null'
    return is_creation


def init_conversation_db(
    app_context: AppContext,
    conv_response: dict,
    conversation_id: str,
) -> dict:
    """
    Handle the conversation initialization.
    If conversation_id is Null, create a new conversation.
    If conversation_id is not Null, load the existing conversation.

    Args:
        app_context (AppContext): GPT Context
        conv_response (dict): The response object containing the conversation
        details.
        conversation_id (str): The conversation ID.

    Returns:
        conv_response: The updated response object.
    """
    if DEBUG:
        log_debug(
            'AI_IC_DB) init_conversation_db() ' +
            f' | conversation_id: {conversation_id}'
        )
    is_creation = is_conversation_creation(conversation_id)

    if is_creation:
        title = get_content_from_msg(conv_response["question"])
        title = title.replace("\n", " ").strip()
        title = title if title != "" else "New Conversation"
        conversation_data = {
            #  The title will be the 1st question
            "title": title,
            "user_id": app_context.get_user_id(),
            "messages": [],
        }
    else:
        # Get existing conversation from Db
        db_response = load_conversation(
            app_context=app_context,
            conversation_id=conversation_id,
        )
        if db_response['error']:
            conv_response["error"] = True
            conv_response["error_message"] = db_response["error_message"]
            return conv_response
        if DEBUG:
            log_debug('AI_IC_DB) init_conversation_db() ' +
                      f' | existing conversation | db_response: {db_response}')
        conversation_data = dict(db_response['resultset'])

    # The initial question is added to the messages list.
    conversation_data['messages'] += \
        interpret_msg(conv_response["question"])

    if is_creation:
        conv_response["cid"] = None
        db_response = create_conversation(
            app_context=app_context,
            new_item_data=conversation_data
        )
        if not db_response["error"]:
            conv_response["cid"] = db_response['resultset']['_id']
    else:
        conv_response["cid"] = conversation_id
        db_response = update_conversation(
            app_context=app_context,
            item_data=conversation_data
        )

    conversation_data['id'] = conv_response["cid"]

    # Set the conversation Id for the GPT functions
    app_context.set_other_data("cid", conv_response["cid"])
    app_context.set_other_data("conv_data", conversation_data)

    if db_response["error"]:
        conv_response["error"] = True
        conv_response["error_message"] = db_response["error_message"]

    if DEBUG:
        log_debug("")
        log_debug(
            'AI_IC_DB) init_conversation_db() | Conversation ' +
            ("Creation" if is_creation else "Update") +
            f' | conversation_id: {conversation_id}'
        )
        log_debug(
            f' | db_response: {db_response}'
        )
        log_debug(
            f' | conversation_data: {conversation_data}'
        )
        log_debug(
            f' | conv_response: {conv_response}'
        )
        log_debug("")
    return conv_response


def update_conversation_db(
    app_context: AppContext,
    conv_response: dict,
) -> dict:
    """
    Update the conversation with the Chatbot response.

    Args:
        app_context (AppContext): GPT Context
        conv_response (dict): The response object containing the Chatbot
        final answer.

    Returns:
        conv_response: The updated response object.
    """
    if DEBUG:
        log_debug(
            'AI_UC_DB) update_conversation_db() ' +
            f' | conv_response received: {conv_response}'
        )

    conversation_data = app_context.get_other_data("conv_data")

    if "cid" not in conv_response and "id" in conversation_data:
        conv_response["cid"] = conversation_data['id']

    # if conv_response["error"]:
    #     return conv_response

    # Record the error message as the assistant response...
    if conv_response["error"]:
        if "response" not in conv_response:
            conv_response["response"] = ""
        conv_response["response"] += ". " if conv_response["response"] \
            else "" + conv_response["error_message"]

    if DEBUG:
        log_debug(
            'AI_UC_DB) update_conversation_db() ' +
            f' | conversation_data in AppContext: {conversation_data}'
        )

    # Add additional messages after the final answer
    if "add_msg_after" in conv_response:
        conversation_data['messages'] += \
            interpret_msg(conv_response["add_msg_after"])
        del conv_response["add_msg_after"]

    # Add the final answer
    if isinstance(conv_response["response"], (dict, list)):
        conversation_data['messages'] += \
            interpret_msg(conv_response["response"])
    else:
        conversation_data['messages'] += [
            {"role": 'assistant', "content": conv_response["response"]}
        ]

    # Add additional messages before the final answer
    if "add_msg_before" in conv_response:
        conversation_data['messages'] += \
            interpret_msg(conv_response["add_msg_before"])
        del conv_response["add_msg_before"]

    # Update the conversation in the Db
    db_response = update_conversation(
        app_context=app_context,
        item_data=conversation_data
    )
    if db_response["error"]:
        conv_response["error"] = True
        conv_response["error_message"] = db_response["error_message"]

    # Update the conversation in the cached GPT Context
    app_context.set_other_data("conv_data", conversation_data)

    if DEBUG:
        log_debug("")
        log_debug(
            'AI_UC_DB) update_conversation_db()' +
            f' | db_response: {db_response}' +
            f' | conv_response: {conv_response}'
        )
        log_debug("")

    return conv_response


def start_conversation(
    app_context: AppContext,
    question: Union[str, dict],
    other: dict,
) -> dict:
    """
    Initialice the conversation object in GPT functions
    (e.g. vision, image generator) if conversation ID is
    specified.
    """
    conv_response = get_default_resultset()
    if "cid" not in other:
        return conv_response

    if isinstance(question, str):
        conv_response["question"] = standard_msg(question)
    else:
        conv_response["question"] = question

    # Get the conversation ID and data, and put the conversation in
    # conv_response["conv_data"]
    conv_response = init_conversation_db(
        app_context=app_context,
        conv_response=conv_response,
        conversation_id=other.get("cid"),
    )
    return conv_response
