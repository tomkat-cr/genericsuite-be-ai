"""
HugginFace platform utilities
"""
from typing import Any
import os
import requests
import uuid

from genericsuite.util.app_context import CommonAppContext
from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import (
    get_default_resultset,
    error_resultset,
    get_mime_type,
)
from genericsuite.util.aws import upload_nodup_file_to_s3

from genericsuite_ai.config.config import Config

DEBUG = False
cac = CommonAppContext()


def hf_query(repo_id: str, payload: dict) -> Any:
    """
    Perform a HuggingFace query

    Args:
        api_url (str): HuggingFace API URL
        payload (dict): HuggingFace payload

    Returns:
        Any: HuggingFace response
    """
    # https://huggingface.co/docs/api-inference/detailed_parameters
    settings = Config(cac.get())
    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"
    }
    api_url = f'{settings.HUGGINGFACE_ENDPOINT_URL}/{repo_id}'
    return requests.post(api_url, headers=headers, json=payload)


def huggingface_img_gen(question: str, image_extension: str = 'jpg') -> dict:
    """
    HuggingFace image generation
    """
    settings = Config(cac.get())
    ig_response = get_default_resultset()

    if not question:
        return error_resultset(
            error_message='No question supplied',
            message_code='HFIG-E010',
        )

    _ = DEBUG and log_debug(
        '1) huggingface_img_gen' +
        f'\n| question: {question}' +
        f'\n| api_url: {settings.HUGGINGFACE_DEFAULT_IMG_GEN_MODEL}')

    image_bytes = hf_query(
        repo_id=settings.HUGGINGFACE_DEFAULT_IMG_GEN_MODEL,
        payload={
            "inputs": question,
        }
    ).content

    # Generate a unique filename
    image_filename = f'hf_img_{uuid.uuid4()}.{image_extension}'
    image_path = f'{settings.TEMP_DIR}/{image_filename}'

    # Create the temporary local file
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    # Store the image bytes in AWS
    upload_result = upload_nodup_file_to_s3(
        file_path=image_path,
        original_filename=image_filename,
        bucket_name=settings.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET,
        sub_dir=cac.app_context.get_user_id(),
    )

    if upload_result['error']:
        return error_resultset(
            error_message=upload_result['error_message'],
            message_code="HFIG-E030",
        )

    # Add the S3 URL to the response
    upload_result['file_name'] = image_filename
    upload_result['file_type'] = get_mime_type(image_filename)
    upload_result['file_size'] = os.path.getsize(image_path)
    ig_response['resultset'] = {'uploaded_file': upload_result}

    if DEBUG:
        log_debug('2) huggingface_img_gen | ig_response:')
        print(ig_response)

    return ig_response
