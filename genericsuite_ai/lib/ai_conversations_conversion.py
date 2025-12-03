"""
Conversations conversion (unmasked / masked)
"""
from typing import Optional, Union, Any
import os
import json

from genericsuite.util.utilities import (
    get_default_resultset,
    get_query_params,
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import AppContext
from genericsuite.config.config import Config

from genericsuite.util.storage import prepare_asset_url
from genericsuite.util.storage import (
    get_storage_base_url,
)

from genericsuite.util.generic_db_middleware import (
    fetch_all_from_db,
    get_item_from_db,
    modify_item_in_db,
)
from genericsuite_ai.lib.ai_storage import get_chatbot_attachments_bucket_name

DEBUG = os.environ.get("AI_ATTACHMENTS_CONVERSION_DEBUG", "0") == "1"

ATTACHMENTS_CONVERSION_ENABLE = os.environ.get(
    "AI_ATTACHMENTS_CONVERSION_ENABLE", "1") == "1"


class UnmaskedUrlConversion:
    """
    Conversation messages handling
    """

    def __init__(self, bucket_name: str, user_id: str, hostname: str):
        self.bucket_name = bucket_name
        self.user_id = user_id
        self.hostname = hostname
        self.public_url = (
            f"{get_storage_base_url(self.bucket_name)}" + f"/{self.user_id}/"
        )
        self.changed = False

        self.debug = DEBUG
        _ = self.debug and log_debug(
            f">> UNMASKED_URL_CONVERSION | bucket_name: {bucket_name}"
            + f"\n| user_id: {user_id}"
            + f"\n| hostname: {hostname}"
            + f"\n| public_url: {self.public_url}"
        )

    def detect_unmasked_url(self, message):
        """
        Detect if the message contains an unmasked URL.
        """
        return self.public_url in message

    def unmasked_url_filename(self, message):
        """
        Get the unmasked URL filename.
        """
        # Get the unmasked URL, until the end of line, space, comma, or other
        # punctuation
        url = message[message.find(self.public_url) +
                      len(self.public_url):].split()[0]
        _ = self.debug and log_debug(
            "\n>-->> UNMASKED_URL_FILENAME" +
            f"\n | message: {message}\n"
            + f"\n | url: {url}\n"
        )
        # Remove any other punctuation
        url = url.split(",")[0]
        url = url.split(";")[0]
        url = url.split(":")[0]
        # Remove the trailing . but not the dot extension
        url = url.rstrip(".")
        # Remove surrounding parentheses
        url = url.lstrip("(")
        url = url.rstrip(")")
        # Remove " and '
        url = url.replace('"', "")
        url = url.replace("'", "")
        _ = self.debug and log_debug(
            f"\n>-->> UNMASKED_URL_FILENAME | Final url: {url}\n")
        return url

    def get_unmasked_url(self, message):
        """
        Get the masked URL from the message.
        """
        unmasked_url = f"{self.public_url}{self.unmasked_url_filename(message)}"
        return unmasked_url

    def get_masked_url(self, message):
        """
        Get the masked URL from the message.
        """
        if self.detect_unmasked_url(message):
            url = self.get_unmasked_url(message)
            _ = self.debug and log_debug(
                f"\n>-->> GET_MASKED_URL | url: {url}\n")
            masked_url = prepare_asset_url(url)
            _ = self.debug and log_debug(
                f"\n>-->> GET_MASKED_URL | masked_url: {masked_url}\n")
            return masked_url
        return None

    def process_one_message(self, message: dict):
        """
        Process one message.
        """
        if "content" in message:
            masked_url = self.get_masked_url(message["content"])
            if masked_url:
                self.changed = True
                unmasked_url = self.get_unmasked_url(message["content"])
                message["content"] = message["content"].replace(
                    unmasked_url, masked_url
                )
                _ = self.debug and log_debug(
                    "\n>-->> [content]"
                    + f"\n| unmasked_url: {unmasked_url}"
                    + f"\n| masked_url: {masked_url}\n"
                )

        if "attachment_url" in message:
            masked_url = self.get_masked_url(message["attachment_url"])
            if masked_url:
                self.changed = True
                unmasked_url = self.get_unmasked_url(message["attachment_url"])
                message["attachment_url"] = masked_url
                _ = self.debug and log_debug(
                    "\n>-->> [attachment_url]"
                    + f"\n| unmasked_url: {unmasked_url}"
                    + f"\n| masked_url: {masked_url}\n"
                )
        return message


def clean_value(value):
    """
    Clean the value.
    """
    if value is None:
        return ""
    value = str(value)
    value = value.lstrip("['")
    value = value.rstrip("']")
    return value


def mask_one_conversation(conversation: dict):
    if not ATTACHMENTS_CONVERSION_ENABLE:
        return conversation
    bucket_name = get_chatbot_attachments_bucket_name()
    hostname = Config().APP_HOST_NAME
    uuc = UnmaskedUrlConversion(
        bucket_name, conversation["user_id"], hostname)
    conversation["messages"] = [
        uuc.process_one_message(message) for message in conversation["messages"]
    ]
    return conversation


def mask_conversation(app_context: AppContext, conversation_id: str):
    """
    Mask a conversation.
    """
    conversation_rs = get_item_from_db(
        app_context=app_context,
        json_file='ai_chatbot_conversations_complete',
        entry_name="_id",
        entry_value=conversation_id
    )
    conversation = conversation_rs["resultset"]
    conversation["user_id"] = clean_value(conversation["user_id"])
    return mask_one_conversation(conversation)


def mask_all_conversations(app_context: AppContext):
    """
    Mask all conversations in the database.
    """
    query_params = get_query_params(app_context.get_request())
    _ = DEBUG and log_debug(
        f">> MASK_ALL_CONVERSATIONS | query_params: {query_params}")
    save = query_params.get("save", "0") == "1"
    bucket_name = query_params.get(
        "bucket_name", get_chatbot_attachments_bucket_name(app_context)
    )
    hostname = query_params.get("hostname")
    # current_user_id = app_context.get_user_id()
    conversations_rs = fetch_all_from_db(
        app_context=app_context,
        json_file="ai_chatbot_conversations_complete",
    )
    conversations = json.loads(conversations_rs["resultset"])
    for conversation in conversations:
        conversation["user_id"] = clean_value(conversation["user_id"])
        uuc = UnmaskedUrlConversion(
            bucket_name, conversation["user_id"], hostname)
        conversation["messages"] = [
            uuc.process_one_message(message) for message in conversation["messages"]
        ]
        if save and uuc.changed:
            conversation["id"] = conv_id
            del conversation["_id"]

            _ = DEBUG and log_debug(
                f">> MASK_ALL_CONVERSATIONS | Conversation ID {conv_id}:"
                + f"\n\n{prev_conv}\n"
            )

            _ = DEBUG and log_debug(
                ">> MASK_ALL_CONVERSATIONS | NEW conversation:\n" +
                f"\n{conversation}"
            )

            save_result = modify_item_in_db(
                app_context=app_context,
                json_file="ai_chatbot_conversations_complete",
                data=conversation,
            )
            _ = DEBUG and log_debug(
                ">> MASK_ALL_CONVERSATIONS | Conversation ID"
                + f" {conv_id}"
                + f"\n| SAVE_RESULT:\n{save_result}"
            )
            if save_result["error"]:
                return save_result

    final_result = get_default_resultset()
    final_result["resultset"] = {
        "success": True,
        "database": os.environ.get("APP_DB_NAME"),
        "option_save": "Items converted" if save else "Only preview",
    }
    return final_result


def ai_conversation_masking(
    app_context_or_blueprint: Any,
    action_data: Optional[Union[dict, None]]
) -> dict:
    """
    GenericDbHelper specific function to mask one conversation.

    Args:
        app_context_or_blueprint (AppContext): the application context object
        action_data (dict, optional): the action data. Defaults to None.
            If it's not None, it must have the following keys (attributes):
            "action": "list", "read", "create", "update" or "delete"
            "resultset": resultset for data to be stored, delete or
                retrieved with the keys: resultset, error, error_message.
            "cnf_db": the table configuration. E.g. tablename is
                cnf_db['tablename']
    """
    # app_context = get_app_context(app_context_or_blueprint)
    action_data = action_data or {}
    tablename = action_data.get("cnf_db", {}).get("table_name")
    # Only if the action is read
    if action_data.get("action") not in ["read"]:
        return action_data['resultset']
    # Verify if any error
    if action_data['resultset']['error']:
        return action_data['resultset']
    if not tablename:
        return action_data['resultset']
    # Mask the conversation
    action_data['resultset']['resultset'] = \
        mask_one_conversation(
            json.loads(action_data['resultset']['resultset']))
    return action_data['resultset']
