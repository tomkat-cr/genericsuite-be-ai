"""
HugginFace platform utilities
"""
from typing import Any, List, Optional, Iterator
import os
import requests
import uuid
import json

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from genericsuite.util.app_context import CommonAppContext, AppContext
from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import (
    get_default_resultset,
    error_resultset,
    get_mime_type,
)
from genericsuite.util.aws import upload_nodup_file_to_s3

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.ai_langchain_models_abstract import CustomLLM as LLM

DEBUG = os.environ.get("AI_HUGGINGFACE_DEBUG", "0") == "1"

HF_TEXT_API_METHOD = os.environ.get("AI_HUGGINGFACE_TEXT_API_METHOD", "openai")
cac = CommonAppContext()


def hf_text_to_image_query(repo_id: str, payload: dict) -> Any:
    """
    Perform a HuggingFace text to image query

    Args:
        repo_id (str): HuggingFace model repository ID
        payload (dict): HuggingFace payload

    Returns:
        Any: HuggingFace response
    """
    # https://huggingface.co/docs/inference-providers/tasks/text-to-image
    settings = Config(cac.get())
    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }
    api_url = f'{settings.HUGGINGFACE_TEXT_TO_IMAGE_ENDPOINT}/{repo_id}'
    return requests.post(api_url, headers=headers, json=payload)


def hf_text_to_text_query_openai(repo_id: str, api_key: str, payload: dict,
                                 stream: bool = False) -> ChatCompletion:
    """
    Perform a HuggingFace text to text query using the OpenAI API

    Args:
        repo_id (str): HuggingFace model repository ID
        api_key (str): HuggingFace API key
        payload (dict): HuggingFace payload

    Returns:
        ChatCompletion: OpenAI response
    """
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model=repo_id,
        messages=payload['messages'],
        stream=stream,
    )
    return completion


def hf_text_to_text_query_request(repo_id: str, api_key: str, payload: dict,
                                  stream: bool = False) -> requests.Response:
    """
    Perform a HuggingFace text to text query request

    Args:
        repo_id (str): HuggingFace model repository ID
        api_key (str): HuggingFace API key
        payload (dict): HuggingFace payload
        stream (bool): Whether to stream the response

    Returns:
        requests.Response: HuggingFace response
    """
    # https://huggingface.co/docs/inference-providers/tasks/chat-completion
    # https://huggingface.co/docs/api-inference/detailed_parameters
    settings = Config(cac.get())
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    api_url = settings.HUGGINGFACE_TEXT_TO_TEXT_ENDPOINT
    payload['model'] = repo_id
    return requests.post(api_url, headers=headers, json=payload, stream=stream)


def hf_text_to_text_query(repo_id: str, api_key: str, payload: dict,
                          stream: bool = False) -> requests.Response:
    """
    Perform a HuggingFace text to text query request
    using the HuggingFace API or the OpenAI API
    depending on the AI_HUGGINGFACE_TEXT_API_METHOD environment variable

    Args:
        repo_id (str): HuggingFace model repository ID
        api_key (str): HuggingFace API key
        payload (dict): HuggingFace payload
        stream (bool): Whether to stream the response

    Returns:
        requests.Response or ChatCompletion: HuggingFace response or
        OpenAI response
    """
    if HF_TEXT_API_METHOD == "openai":
        return hf_text_to_text_query_openai(repo_id, api_key, payload, stream)
    else:
        return hf_text_to_text_query_request(repo_id, api_key, payload, stream)


def hf_text_to_text_stream(repo_id: str, api_key: str, payload: dict
                           ) -> Iterator[dict]:
    """
    Stream the HuggingFace text to text query

    Args:
        repo_id (str): HuggingFace model repository ID
        api_key (str): HuggingFace API key
        payload (dict): HuggingFace payload

    Returns:
        Iterator[dict]: HuggingFace response as a dictionary, one dictionary
            per line
    """
    response = hf_text_to_text_query(
        repo_id=repo_id,
        api_key=api_key,
        payload=payload,
        stream=True,
    )
    for line in response.iter_lines():
        if not line.startswith(b"data:"):
            continue
        if line.strip() == b"data: [DONE]":
            return
        yield json.loads(line.decode("utf-8").lstrip("data:").rstrip("/n"))


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

    image_bytes = hf_text_to_image_query(
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


def huggingface_chat(
    prompt: str,
    repo_id: str = None,
    api_key: str = None,
) -> dict:
    """
    HuggingFace chat

    Args:
        prompt (str): The prompt to generate from.
        repo_id (str): the model to use.
        api_key (str): The API key to use for the HuggingFace API.

    Returns:
        dict: The response from the model as a string in the ['resultset'] key.
    """
    settings = Config(cac.get())
    chat_response = get_default_resultset()

    if not prompt:
        return error_resultset(
            error_message='No prompt supplied',
            message_code='HFIG-E010',
        )

    messages = [
        {
            'role': 'user',
            'content': prompt
        }
    ]

    completion = hf_text_to_text_query(
        repo_id=repo_id or settings.HUGGINGFACE_DEFAULT_CHAT_MODEL,
        api_key=api_key or settings.HUGGINGFACE_API_KEY,
        payload={
            'messages': messages
        }
    )

    chat_response['resultset'] = {
        'content': completion.choices[0].message
        if isinstance(completion, ChatCompletion)
        else completion.choices[0].message.content
    }

    return chat_response


class HuggingFaceChatModel(LLM):
    """
    HuggingFaceChatModel + models class for LangChain.
    """

    # Mandatory parameters
    api_key: str
    app_context: AppContext

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off
                at the first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising
                NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are
                usually passed to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT
            include the prompt.
        """
        chat_response = huggingface_chat(prompt, self.model_name, self.api_key)
        result = chat_response['resultset']['content']
        _ = DEBUG and log_debug(
            f">> HuggingFaceChatModel | _call chat_response: {chat_response}")
        return result

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off
                at the first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are
                usually passed to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        payload = {
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        for chunk in hf_text_to_text_stream(self.model_name, self.api_key,
                                            payload):
            yield GenerationChunk(
                text=chunk['choices'][0]['message']['content'])
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk['choices'][0]['message']['content'])
            if stop and \
                    chunk['choices'][0]['message']['content'].endswith(stop):
                break

    @property
    def _llm_type(self) -> str:
        """
        Get the type of language model used by this chat model. Used for
        logging purposes only.
        """
        return "HuggingFaceChatModel"
