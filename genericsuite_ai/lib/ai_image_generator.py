"""
AI Image Generation: processing text and generate images
"""
from typing import Any, Optional

from openai import OpenAI
from openai.resources.images import ImagesResponse

from langchain.agents import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from genericsuite.util.aws import save_file_from_url
from genericsuite.util.utilities import (
    get_default_resultset,
    get_default_value,
    get_mime_type,
    get_file_size,
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
    clarifai_img_gen,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities
from genericsuite_ai.lib.huggingface import (
    huggingface_img_gen,
)
from genericsuite_ai.lib.amazon_bedrock import (
    aws_bedrock_img_gen,
)


DEBUG = False
cac = CommonAppContext()


class ImageGeneratorParams(BaseModel):
    """
    Vision parameters structure
    """
    question: str = Field(description="A question about the image specified.")
    other: Optional[dict] = Field(description="Additional parametes." +
                                  " Defaults to {}",
                                  default={})


def get_filename_from_dalle_url(url: str) -> (str, str):
    """
    Get the filename from a image URL generated by OpenAI DALL-E
    e.g.
    url='https://oaidalleapiprodscus.blob.core.windows.net/private/org-OkjXZTFKxFbZY57FJkWoakVB/user-AfDKXv3oVkfmF6c3rWoJmG5C/img-DNdtnngx4f5zoPYSpqGH61xZ.png?st=2023-12-02T22%3A41%3A02Z&se=2023-12-03T00%3A41%3A02Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-12-02T22%3A39%3A05Z&ske=2023-12-03T22%3A39%3A05Z&sks=b&skv=2021-08-06&sig=Cc9lSkeJ5l8P7NDUEq%2Bw049UT5qx8G1IQVojSREVNO8%3D'

    Args:
        url (str): The URL to get the image.

    Returns:
        (str, str): (file_name, file_type) tuple, or None, None
        if those elements cannot be found.
    """
    if DEBUG:
        log_debug("\nGET_FILENAME_FROM_DALLE_URL\n" +
                  f"url: {url}")
    file_name = deduce_filename_from_url(url)
    file_type = get_mime_type(file_name)
    if DEBUG:
        log_debug("\nGET_FILENAME_FROM_DALLE_URL\n" +
                  f"file_name: {file_name}\n" +
                  f"file_type: {file_type}\n")
    return file_name, file_type


def save_image(
    url: str,
    user_id: str,
    original_filename: str = None,
) -> dict:
    """
    Save an image from a URL generated by DALL-E to AWS S3

    Args:
        url (str): The URL to get the image.
        user_id (str): User ID to store the file in it's S3 space.
        Defaults to None.
        original_filename (str): original file name. Defaults to None.

    Returns:
        dict: A dictionary with the following elements:
            public_url (str): URL for the image.
            final_filename (str): file name of the image with date/time added.
            file_size (int): the file size in bytes.
            error (bool): True if there was an error, False otherwise.
            error_message (str): the eventual error message or None if no
                errors
    """
    settings = Config(cac.get())
    result = get_default_resultset()
    result['public_url'] = None
    result['final_filename'] = None
    result['file_size'] = None
    bucket_name = settings.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET
    if not bucket_name:
        result['error'] = True
        result['error_message'] = \
            "AWS_S3_CHATBOT_ATTACHMENTS_BUCKET is not configured [2]"
    else:
        # public_url, final_filename, file_size, error = save_file_from_url(
        save_result = save_file_from_url(
            url=url,
            bucket_name=bucket_name,
            sub_dir=user_id,
            original_filename=original_filename,
        )
        # attachment_url = public_url
        result['error'] = save_result['error']
        result['error_message'] = save_result['error_message']
        result['public_url'] = save_result['public_url']
        result['final_filename'] = save_result['final_filename']
        result['file_size'] = save_result['file_size']
    return result
    # return attachment_url, final_filename, file_size, error


def get_img_gen_name() -> str:
    """ Returns the Image Generator configured technology name """
    settings = Config(cac.get())
    if settings.AI_IMG_GEN_TECHNOLOGY == "gemini":
        model_name = f"Google Gemini: f{settings.GOOGLE_IMG_GEN_MODEL}"
    elif settings.AI_IMG_GEN_TECHNOLOGY == "huggingface":
        model_name = "HuggingFace Image Generator: " + \
            f"{settings.HUGGINGFACE_DEFAULT_IMG_GEN_MODEL}"
    elif settings.AI_IMG_GEN_TECHNOLOGY == "bedrock":
        model_name = "Amazon Bedrok Image Generator: " + \
            f"{settings.AWS_BEDROCK_IMAGE_GEN_MODEL_ID}"
    elif settings.AI_IMG_GEN_TECHNOLOGY == "clarifai":
        model_name = "Clarifai Image Generator: " + \
            f"{settings.AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL}"
    else:
        model_name = f"OpenAI: {settings.OPENAI_IMAGE_GEN_MODEL}"
    log_debug("get_img_gen_name | AI_IMG_GEN_TECHNOLOGY:" +
              f" {settings.AI_IMG_GEN_TECHNOLOGY}"
              f" | model_name: {model_name}")
    return model_name


def get_img_gen_response(response: dict, other: dict) -> dict:
    """
    Returns the configured Image Generator technology response
    """

    settings = Config(cac.get())
    billing = BillingUtilities(cac.get())

    # Maximun tokens to be used by the model. Defaults to 500.
    max_tokens = get_default_value("max_tokens", other,
                                   settings.OPENAI_MAX_TOKENS)

    log_debug("get_img_gen_response | AI_IMG_GEN_TECHNOLOGY:" +
              f" {settings.AI_IMG_GEN_TECHNOLOGY}")
    try:
        if settings.AI_IMG_GEN_TECHNOLOGY == "gemini":
            client = ChatGoogleGenerativeAI(
                # model="gemini-pro-vision"
                model=settings.GOOGLE_IMG_GEN_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                temperature=float(settings.OPENAI_TEMPERATURE),
            )
            ig_response = client.invoke([
                HumanMessage(content=response["question"])
            ])
            log_debug(f"Google Gemini Image Generator response: {ig_response}")
            response['response'] = ig_response.content

        elif settings.AI_IMG_GEN_TECHNOLOGY == "huggingface":
            ig_response = huggingface_img_gen(
                question=response["question"],
            )
            log_debug(f"get_img_gen_response | {get_img_gen_name()}" +
                      f" response: {ig_response}")
            if ig_response["error"]:
                response['error'] = True
                response['error_message'] = ig_response["error_message"]
            else:
                response['response'] = ig_response["resultset"]

        elif settings.AI_IMG_GEN_TECHNOLOGY == "bedrock":
            ig_response = aws_bedrock_img_gen(
                question=response["question"],
            )
            log_debug(f"get_img_gen_response | {get_img_gen_name()}" +
                      f" response: {ig_response}")
            if ig_response["error"]:
                response['error'] = True
                response['error_message'] = ig_response["error_message"]
            else:
                response['response'] = ig_response["resultset"]

        elif settings.AI_IMG_GEN_TECHNOLOGY == "clarifai":
            ig_response = clarifai_img_gen(
                question=response["question"],
            )
            log_debug(f"get_img_gen_response | {get_img_gen_name()}" +
                      f" response: {ig_response}")
            if ig_response["error"]:
                response['error'] = True
                response['error_message'] = ig_response["error_message"]
            else:
                response['response'] = ig_response["resultset"]

        else:
            # Open the client
            openai_api_key = billing.get_openai_api_key()
            if not openai_api_key:
                response['error'] = True
                response['error_message'] = "OpenAI API key is not configured [IAIG-E020]"
                return response
            client = OpenAI(
                api_key=openai_api_key
            )
            # Process the question and image
            ig_response = client.images.generate(
                # model="dall-e-3",
                model=settings.OPENAI_IMAGE_GEN_MODEL,
                prompt=response["question"],
                size=other["size"],
                quality=other["quality"],
                n=1,
            )
            log_debug(f"Dall-E ig_response: {ig_response}")

            # The 'ImagesResponse' object has an attribute 'data' which is a
            # list of 'Image' objects.
            # We should iterate over this list and extract the URL from each
            # 'Image' object if it exists.

            # Check if the 'ig_response' is an instance of 'ImagesResponse'
            # and extract the URLs from the 'data' attribute
            if isinstance(ig_response, ImagesResponse):
                # Assuming each 'Image' object in the 'data' list has a 'url' attribute
                image_urls = [image.url for image in ig_response.data if hasattr(image, 'url')]
                response['response'] = image_urls
            else:
                # Handle other types of responses or raise an error
                response['error'] = True
                response['error_message'] = "ERROR [IAIG-E030] Unexpected " + \
                    "response type received from image generation API."

    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [IAIG-E010]: {str(error)}"
    return response


def image_generator(params: Any) -> dict:
    """
    Process the specified image and answer the question about it
    using OpenAI DALL·E 3, Google Gemini or the configured
    Clarifi text to image processor.

    Args:
        question (Any): a question about the image specified.
        other (dict): additional parametes. Defaults to {}
            Other entries are:
                "cid" (str): conversation ID. Defaults to None.
                "size" (str): image size. Defaults to "1024x1024".
                "quality" (str): image quality. Defaults to "standard".
                "mock_response" (str): mock response. Defaults to "0".
                "mock_error" (str): mock error. Defaults to "0".

            "size" can be:
            "256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"
            If "size" is not provided, it will be set to "1024x1024".

            "quality" can be: "low", "standard", "high"
            If "quality" is not provided, it will be set to "standard".

            "mock_response" can be: "0", "1", "2"
            "mock_error" can be: "0", "1", "2"

            If "mock_response" is not provided, it will be set to "0".

    Returns:
        dict: a standard response with the question answered by the model.
    """
    # Reference:
    # https://platform.openai.com/docs/guides/images

    settings = Config(cac.get())
    params = interpret_tool_params(tool_params=params,
                                   first_param_name="question",
                                   schema=ImageGeneratorParams)

    question = params.question
    other = params.other

    if not other:
        other = {
            "cid": cac.app_context.get_other_data("cid"),
        }

    if not question:
        response = get_default_resultset()
        response["error"] = True
        response["error_message"] = "question parameter" + \
            " should be provided"
        return response

    # question = interpret_any(question)
    response = start_conversation(app_context=cac.app_context,
                                  question=question, other=other)
    if response["error"]:
        return response

    # Image size
    other["size"] = get_default_value("size", other, "1024x1024")

    # other["size"] = "640x640"
    #   {'error': {'code': None, 'message': "'640x640' is not one of
    #   ['256x256', '512x512', '1024x1024','1024x1792', '1792x1024'] - 'size'",
    #   'param': None, 'type': 'invalid_request_error'}}

    # other["size"] = "512x512"
    #   {'error': {'code': 'invalid_size', 'message': 'The size is not
    #   supported by this model.', 'param': None,
    #   'type': 'invalid_request_error'}}

    # Image quality
    other["quality"] = get_default_value("quality", other, "standard")

    # Mock image/error response
    other["mock_response"] = get_default_value("mock_response", other, "0")
    other["mock_error"] = get_default_value("mock_error", other, "0")

    _ = DEBUG and log_debug(
            "\nGS Image Generator\n" +
            f"Image size: {other['size']}\n" +
            f"Image quality: {other['quality']}\n" +
            f"Question: {question}\n" +
            f"Other parameters: {other}\n"
            f"OPENAI_IMAGE_GEN_MODEL: {settings.OPENAI_IMAGE_GEN_MODEL}")

    response["question"] = question

    if other["mock_error"] == "1":
        response["error"] = True
        response["error_message"] = "---ON PURPOSE ERROR---"
    elif other["mock_response"] == "1":
        response["response"] = 'https://images.openai.com/blob/' + \
            '0303dc78-1b1c-4bbe-a24f-cb5f0ac95565/avocado-square.png?' + \
            'trim=0,0,0,0&width=2600'
    else:
        response = get_img_gen_response(response, other)

    _ = DEBUG and log_debug(
        "\nAnswer from Image Generator:\n" +
        f"{response}\n")

    return update_img_and_conversation(response, other)


def update_img_and_conversation(response: dict, other: dict) -> dict:
    """
    Update image(s) URL and conversation.
    """
    public_url_list = []
    if not response['error']:
        if 'uploaded_file' in response["response"]:
            # The file was already uploaded the the App's own storage...
            save_result = response["response"]['uploaded_file']
            public_url_list.append({
                "file_name": save_result['file_name'],
                "file_type": save_result['file_type'],
                "public_url": save_result['public_url'],
                "final_filename": save_result['final_filename'],
                "file_size": save_result['file_size'],
            })
        else:
            # Images generated must be saved because the URLs
            # returned by the model expires.
            if isinstance(response["response"], str):
                response["response"] = [response["response"]]
            for url in response["response"]:
                file_name, file_type = get_filename_from_dalle_url(url)
                save_result = save_image(
                    url=url,
                    user_id=cac.app_context.get_user_id(),
                    original_filename=file_name
                )
                if save_result['error']:
                    response['error'] = True
                    response['error_message'] = \
                        f"ERROR [IAIG-020]: {str(save_result['error_message'])}"
                    break
                public_url_list.append({
                    "file_name": file_name,
                    "file_type": file_type,
                    "public_url": save_result['public_url'],
                    "final_filename": save_result['final_filename'],
                    "file_size": save_result['file_size'],
                })

    if not response['error']:
        response['response'] = public_url_list

    if "cid" in other:
        if not response['error']:
            response["response"] = [
                {
                    "role": "assistant",
                    "type": "file_name",
                    "sub_type": item["file_type"] if item["file_type"] else "image",
                    "file_name": (
                        f'"{item["final_filename"]}" (' +
                        f'{get_file_size(item["file_size"], "mb")})'
                    ),
                    "attachment_url": item["public_url"],
                    "final_filename": item["final_filename"],
                }
            for item in public_url_list]
            _ = DEBUG and log_debug(
                "\nIMAGE_GENERATOR | Answer fixed for" +
                " 'update_conversation_db':\n" +
                f'{response["response"]}\n')
        response = update_conversation_db(
            app_context=cac.app_context,
            conv_response=response,
        )
        # The final response is always the image URL
        if not response['error']:
            response['response'] = public_url_list
    return response


@tool
def image_generator_text_response(params: Any) -> str:
    """
Useful when you need to perform text to image generation. This Tool returns the generated image(s) URL(s) only.
Args: params (dict): Tool parameters. Must contain: "question" (str): a question about the image specified.
    """
    return image_generator_text_response_func(params)


def image_generator_text_response_func(params: Any) -> str:
    """
    Performs text to image generation.
    
    Args:
        params (dict): Tool parameters. Must contain:
            "question" (str): a question about the image specified.

    Returns:
        str: Generated image(s) URL(s) with a title separated by a newline
        character, or [FUNC+ERROR] {error_message}
    """
    model_response = image_generator(params)
    if model_response["error"]:
        response = gpt_func_error(model_response["error_message"])
    else:
        if isinstance(model_response["response"], str):
            model_response["response"] = [model_response["response"]]
        response = "\n".join([
            f"[Click here to see the image]({url})"
            for url in model_response["response"]
        ])

    if DEBUG:
        log_debug("\nText formatted answer from" +
                  " IMAGE_GENERATOR_TEXT_RESPONSE:\n" +
                  f"{response}\n")
    return response
