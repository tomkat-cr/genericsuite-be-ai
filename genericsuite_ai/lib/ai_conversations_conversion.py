"""
Conversations conversion (unmasked / masked)
"""
import os
import json

from genericsuite.util.utilities import (
    get_default_resultset,
    get_query_params,
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import AppContext
from genericsuite.util.aws import (
    get_storage_masked_url,
    s3_base_url,
)
from genericsuite.util.generic_db_middleware import (
    fetch_all_from_db,
    modify_item_in_db,
)
from genericsuite_ai.config.config import Config


class UnmaskedUrlConversion:
    """
    Conversation messages handling
    """
    def __init__(self, bucket_name: str, user_id: str, hostname: str):
        self.bucket_name = bucket_name
        self.user_id = user_id
        self.hostname = hostname
        self.public_url = f"{s3_base_url(self.bucket_name)}/{self.user_id}/"
        self.changed = False

    def detect_unmasked_url(self, message):
        """
        Detect if the message contains an unmasked URL.
        """
        return self.public_url in message

    def unmasked_url_filename(self, message):
        """
        Get the unmasked URL filename.
        """
        # Get the unmasked URL, until the end of line, space, comma, or other punctuation
        url = message[message.find(self.public_url) + len(self.public_url):].split()[0]
        # Remove any other punctuation
        url = url.split(",")[0]
        url = url.split(";")[0]
        url = url.split(":")[0]
        # Remove the trailing . but not the dot extension
        url = url.rstrip(".")
        # Remove " and '
        url = url.replace('"', '')
        url = url.replace("'", '')
        log_debug(f"\n>-->> UNMASKED_URL_FILENAME | url: {url}\n")
        return url

    def get_unmasked_url(self, message):
        """
        Get the masked URL from the message.
        """
        unmasked_url = f'{self.public_url}{self.unmasked_url_filename(message)}'
        return unmasked_url

    def get_masked_url(self, message):
        """
        Get the masked URL from the message.
        """
        if self.detect_unmasked_url(message):
            masked_url = get_storage_masked_url(
                self.bucket_name,
                f'{self.user_id}/{self.unmasked_url_filename(message)}',
                self.hostname,
            )
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
                message["content"] = message["content"].replace(unmasked_url, masked_url)
                log_debug("\n>-->> [content]" +
                    f"\n| unmasked_url: {unmasked_url}" +
                    f"\n| masked_url: {masked_url}\n")

        if "attachment_url" in message:
            masked_url = self.get_masked_url(message["attachment_url"])
            if masked_url:
                self.changed = True
                unmasked_url = self.get_unmasked_url(message["attachment_url"])
                message["attachment_url"] = masked_url
                log_debug("\n>-->> [attachment_url]" +
                    f"\n| unmasked_url: {unmasked_url}" +
                    f"\n| masked_url: {masked_url}\n")
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


def mask_all_conversations(app_context: AppContext):
    """
    Mask all conversations in the database.
    """
    query_params = get_query_params(app_context.get_request())
    log_debug(f">> MASK_ALL_CONVERSATIONS | query_params: {query_params}")
    save = query_params.get('save', "0") == "1"
    bucket_name = query_params.get('bucket_name',
        Config(app_context).AWS_S3_CHATBOT_ATTACHMENTS_BUCKET)
    hostname = query_params.get('hostname')
    # current_user_id = app_context.get_user_id()
    conversations_rs = fetch_all_from_db(
        app_context=app_context,
        json_file='ai_chatbot_conversations_complete',
    )
    # log_debug(f">> MASK_ALL_CONVERSATIONS | conversation_rs: {conversations_rs}")
    conversations = json.loads(conversations_rs['resultset'])
    # log_debug(f">> MASK_ALL_CONVERSATIONS | conversations: {conversations_rs}")
    for conversation in conversations:
        prev_conv = dict(**conversation)
        conv_id = str(conversation['_id']['$oid'])
        conversation['user_id'] = clean_value(conversation['user_id'])
        uuc = UnmaskedUrlConversion(bucket_name, conversation['user_id'], hostname)
        conversation['messages'] = [
            uuc.process_one_message(message)
            for message in conversation['messages']
        ]
        if save and uuc.changed:
            conversation['id'] = conv_id
            del conversation['_id']

            log_debug(f">> MASK_ALL_CONVERSATIONS | Conversation ID {conv_id}:\n\n{prev_conv}\n")

            log_debug('>> MASK_ALL_CONVERSATIONS | NEW conversation:\n' +
                f"\n{conversation}")

            save_result = modify_item_in_db(
                app_context=app_context,
                json_file='ai_chatbot_conversations_complete',
                data=conversation,
            )
            log_debug(">> MASK_ALL_CONVERSATIONS | Conversation ID" +
                f" {conv_id}" +
                f"\n| SAVE_RESULT:\n{save_result}")
            if save_result['error']:
                return save_result

    final_result = get_default_resultset()
    final_result['resultset'] = {
        'success': True,
        'database': os.environ.get('APP_DB_NAME'),
        'option_save': "Items converted" if save else "Only preview",
    }
    return final_result
