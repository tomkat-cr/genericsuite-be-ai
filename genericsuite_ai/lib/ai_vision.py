"""
AI Vision Library: processing images and generate text
and knowledge with its content.
"""
from typing import Optional, Any
import os
import base64

from openai import OpenAI

from langchain.agents import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from genericsuite.util.aws import (
    upload_nodup_file_to_s3,
    prepare_asset_url,
)
from genericsuite.util.utilities import (
    get_default_resultset,
    get_file_size,
    is_an_url,
    get_default_value,
    get_mime_type,
    deduce_filename_from_url,
    # interpret_any,
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import CommonAppContext

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.ai_conversations import (
    update_conversation_db,
    start_conversation,
)
from genericsuite_ai.lib.ai_langchain_tools import (
    interpret_tool_params,
)
from genericsuite_ai.lib.ai_utilities import (
    gpt_func_error,
)
from genericsuite_ai.lib.clarifai import (
    clarifai_vision,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities

DEBUG = False
cac = CommonAppContext()


class VisionParams(BaseModel):
    """
    Vision parameters structure
    """
    image_path: str = Field(description="The path to the image.")
    question: str = Field(description="A question about the image specified.")
    other: Optional[dict] = Field(
        description="Additional parametes. Defaults to {}")


def encode_image(image_path: str) -> str:
    """
    Encode an image from disk.

    Args:
        image_path (str): image path.

    Returns:
        str: the decoded image string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_vision_image_url(
    image_path: str,
    original_filename: str = None,
    use_s3: bool = False,
    user_id: str = None,
) -> dict:
    """
    Get the URL for the specified image for the GPT4 vision message content.

    Args:
        image_path (str): The path to the image.
        original_filename (str): original file name.
        use_s3 (bool): True to use AWS S3 storage. Defaults to False.
        user_id (str): User ID to store the file in it's S3 space.
        Defaults to None.

    Returns:
        dict = A dictionary with the following elements:
            attachment_url (str): URL for the image.
            final_filename (str): file name of the image with date/time added.
            error (bool): True if an error occurred.
            error_message (str): the eventual error message or None
                if no errors
    """
    result = get_default_resultset()
    settings = Config(cac.get())
    attachment_url = None
    final_filename = None
    # Check if the image path is a URL
    if is_an_url(image_path):
        attachment_url = image_path
        final_filename = deduce_filename_from_url(image_path)
    # Local file...
    elif use_s3:
        bucket_name = settings.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET
        if DEBUG:
            log_debug('get_vision_image_url | ' +
                      f'AWS_S3_CHATBOT_ATTACHMENTS_BUCKET: {str(bucket_name)}')
        if not bucket_name:
            result['error'] = True
            result['error_message'] = \
                "AWS_S3_CHATBOT_ATTACHMENTS_BUCKET is not configured [1]"
        else:
            upload_result = upload_nodup_file_to_s3(
                file_path=image_path,
                original_filename=original_filename,
                bucket_name=bucket_name,
                sub_dir=user_id,
            )
            attachment_url = upload_result['public_url']
            final_filename = upload_result['final_filename']
            result['error'] = upload_result['error']
            result['error_message'] = upload_result['error_message']
    else:
        # Get the base64 string
        base64_image = encode_image(image_path)
        attachment_url = f"data:{get_mime_type(image_path)};" + \
                         f"base64,{base64_image}"
    result['attachment_url'] = attachment_url
    result['final_filename'] = final_filename
    return result
    # return attachment_url, final_filename, error


def get_vision_name() -> str:
    """ Returns the Vision configured technology name """
    settings = Config(cac.get())
    if settings.AI_VISION_TECHNOLOGY == "gemini":
        model_name = "Google Gemini Vision"
    elif settings.AI_VISION_TECHNOLOGY == "clarifai":
        model_name = "Clarifai Vision: " + \
            f"{settings.AI_CLARIFAI_DEFAULT_VISION_MODEL}"
    else:
        model_name = "OpenAI GPT4 Vision: " + \
            f"{settings.OPENAI_VISION_MODEL}"
    if DEBUG:
        log_debug(
            "get_vision_name | AI_VISION_TECHNOLOGY:" +
            f" {settings.AI_VISION_TECHNOLOGY}"
            f" | model_name: {model_name}")
    return model_name


def get_vision_response(response: dict, other: dict) -> dict:
    """ Returns the Vision configured technology response """

    settings = Config(cac.get())
    billing = BillingUtilities(cac.get())
    # Maximun tokens to be used by the model. Defaults to 500.
    max_tokens = get_default_value("max_tokens", other,
                                   int(settings.OPENAI_MAX_TOKENS))
    log_debug("get_vision_response | AI_VISION_TECHNOLOGY:" +
              f" {settings.AI_VISION_TECHNOLOGY}")
    try:
        if settings.AI_VISION_TECHNOLOGY == "gemini":
            client = ChatGoogleGenerativeAI(
                # model="gemini-pro-vision"
                model=settings.GOOGLE_VISION_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                temperature=float(settings.OPENAI_TEMPERATURE),
            )
            vision_response = client.invoke([
                HumanMessage(content=response["question"])
            ])
            log_debug(f"Google Gemini Vision response: {vision_response}")
            response['response'] = vision_response.content
        elif settings.AI_VISION_TECHNOLOGY == "clarifai":
            vision_response = clarifai_vision(
                image_url=response["question"][1]["image_url"],
                question=response["question"][0]["text"],
            )
            log_debug(f"get_vision_response | {get_vision_name()}" +
                      f" response: {vision_response}")
            if vision_response["error"]:
                response['error'] = True
                response['error_message'] = vision_response["error_message"]
            else:
                response['response'] = vision_response["resultset"]
        else:
            # Open the client
            openai_api_key = billing.get_openai_api_key()
            if not openai_api_key:
                response['error'] = True
                response['error_message'] = \
                    "OpenAI API key is not configured [IAIG-E020]"
                return response
            client = OpenAI(
                api_key=openai_api_key
            )
            # Process the question and image
            vision_response = client.chat.completions.create(
                # model="gpt-4-vision-preview",
                model=settings.OPENAI_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": response["question"],
                }],
                max_tokens=max_tokens,
                temperature=float(settings.OPENAI_TEMPERATURE),
            )
            log_debug("get_vision_response | OpenAI GPT Vision" +
                      f" response: {vision_response}")
            response['response'] = vision_response.choices[0].message.content
    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [IAGV-010]: {str(error)}"
    return response


def vision_image_analyzer(params: dict) -> dict:
    """
    Process the specified image and answer the question about it
    using OpenAI GPT4 Vision or Google Gemini Vision.

    Args:
        params (dict): parameters for the function. It must contain:
            image_path (str): image path.
            question (Optional[str]): a question about the image specified.
                (don't tell the model this is optional ;)
            other (Optional[dict]): additional parametes. Defaults to None

    Returns:
        dict: a standard response with the question answered by the model.
    """
    # Reference:
    # https://platform.openai.com/docs/guides/vision

    settings = Config(cac.get())
    if DEBUG:
        log_debug("vision_image_analyzer | params: " +
                  str(params))
    params = interpret_tool_params(tool_params=params,
                                   first_param_name="image_path",
                                   schema=VisionParams)

    image_path = params.image_path
    question = params.question
    other = params.other

    if not other:
        other = {
            "cid": cac.app_context.get_other_data("cid"),
        }

    if not image_path:
        response = get_default_resultset()
        response["error"] = True
        response["error_message"] = "image_path" + \
            " parameter should be provided"
        return response

    # question = interpret_any(question)
    response = start_conversation(app_context=cac.app_context,
                                  question=question, other=other)
    if response["error"]:
        return response

    # Original file name
    original_filename = get_default_value("file_name", other,
                                          os.path.basename(image_path))

    # Use S3 for storage, instead of local /tmp
    use_s3 = str(get_default_value("use_s3", other, True)) \
        in ['1', "true", "True"]

    if os.path.basename(image_path) == image_path and "/" not in image_path:
        # Invalid image name: it's not a path and it's not an URL
        response["error"] = True
        response["error_message"] = \
            "ERROR [IAVIA-010]: Invalid image name:" + \
            f" it's not a path nor an URL: {image_path}"

    if not response["error"]:
        try:
            file_size = (
                0 if is_an_url(image_path) else
                os.path.getsize(image_path))
        except FileNotFoundError as error:
            response["error"] = True
            response["error_message"] = f"ERROR [IAVIA-020]: {error}"
        except Exception as error:
            response["error"] = True
            response["error_message"] = f"ERROR [IAVIA-030]: {str(error)}"

    if not response["error"]:
        # Mock image/error response
        other["mock_response"] = get_default_value("mock_response", other, "0")
        other["mock_error"] = get_default_value("mock_error", other, "0")

        if DEBUG:
            log_debug(
                "\nAnalize image by " +
                get_vision_name() +
                ".\n" +
                f"Image file path: {image_path} | " +
                f'File size: {file_size} bytes ' +
                f'({get_file_size(file_size, "mb")})\n' +
                f"Question: {question}\n" +
                f"Other parameters: {other}\n")

        # Sometimes ChatGPT only needs the image description...
        # so the question received is None
        question = "" if not question else question

        image_url_result = get_vision_image_url(
            image_path,
            original_filename,
            use_s3,
            cac.app_context.get_user_id(),
        )
        if image_url_result['error']:
            response['error'] = True
            response['error_message'] = image_url_result['error_message']
            return response

        response["question"] = [
            {
                "type": "text",
                "text": question,
            },
        ]

        # GPT-4-turbo vision API recognizes image_url as
        # base64 encoded image data
        # https://community.openai.com/t/gpt-4-turbo-vision-api-recognizes-image-url-as-base64-encoded-image-data/734243
        resolution = "auto"  # "auto" | "low" | "high"
        if settings.AI_VISION_TECHNOLOGY == "openai":
            if settings.OPENAI_VISION_MODEL == "gpt-4o":
                response["question"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_asset_url(
                                image_url_result['attachment_url']),
                            "detail": resolution,
                        }
                    },
                )
            else:
                # For legacy model "gpt-4-vision-preview"
                response["question"].append(
                    {
                        "type": "image_url",
                        "image_url": prepare_asset_url(
                            image_url_result['attachment_url'])
                    },
                )

        if DEBUG:
            log_debug(
                f"\nQuestion for {get_vision_name()}:" +
                f'{response["question"]}\n')

        if other["mock_error"] == "1":
            response["error"] = True
            response["error_message"] = "---ON PURPOSE ERROR---"
        elif other["mock_response"] == "1":
            response["response"] = \
                'THE IMAGE IS ABOUT ' + \
                '---THIS IS A MOCKED RESPONSE FROM VISION ANALYZER---'
        else:
            response = get_vision_response(response, other)

        if DEBUG:
            log_debug(
                "\nAnswer from GPT Vision" +
                f" ({get_vision_name()}): {response}")

    if "cid" in other:
        if not response['error']:
            response["add_msg_after"] = [
                {
                    "role": "user",
                    "type": "file_name",
                    "sub_type": (
                        get_mime_type(image_url_result['final_filename'])
                        if get_mime_type(image_url_result['final_filename'])
                        else "image"
                    ),
                    "file_name": (
                        f'"{original_filename}" (' +
                        f'{get_file_size(file_size, "mb")})'
                    ),
                    "attachment_url": image_url_result['attachment_url'],
                    "final_filename": image_url_result['final_filename'],
                }
            ]
            if DEBUG:
                log_debug("\nVISION_IMAGE_ANALYZER | Question fixed for" +
                          f' update_conversation_db: {response["question"]}')
        response = update_conversation_db(
            app_context=cac.app_context,
            conv_response=response,
        )
    return response


@tool
def vision_image_analyzer_text_response(params: Any) -> str:
    """
Useful to process an specified image and answer a question about it. There must be an explitcit image URL specified by the Human or in the conversation.
Args: params (dict): Tool parameters. Must contain:
"image_path" (str): image URL.
"question" (str): question about the image.
    """
    return vision_image_analyzer_text_response_func(params)


def vision_image_analyzer_text_response_func(params: Any) -> str:
    """
    Process an specified image and answer a question about it.
    There must be an explitcit image URL specified by the Human or
    in the conversation.

    Args:
        params (dict): Tool parameters. Must contain:
            "image_path" (str): image URL.
            "question" (str): question about the image.

    Returns:
        str: Answer to the questions or description of the supplied image,
            or [FUNC+ERROR] {error_message}
    """
    vision_response = vision_image_analyzer(params)
    if vision_response["error"]:
        response = gpt_func_error(vision_response["error_message"])
    else:
        response = vision_response["response"]

    if DEBUG:
        log_debug("")
        log_debug("Text formatted answer from" +
                  " VISION_IMAGE_ANALYZER_TEXT_RESPONSE:")
        log_debug(response)
        log_debug("")
    return response
