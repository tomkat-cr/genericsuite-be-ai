"""
AI Endpoints for Chalice
"""
from typing import Optional

# from chalice.app import Response

# from genericsuite.util.blueprint_one import BlueprintOne
from genericsuite.util.framework_abs_layer import Response, BlueprintOne

from genericsuite.util.jwt import (
    request_authentication,
    AuthorizedRequest
)
from genericsuite_ai.lib.ai_chatbot_endpoint import (
    ai_chatbot_endpoint as ai_chatbot_endpoint_model,
    vision_image_analyzer_endpoint as vision_image_analyzer_endpoint_model,
    transcribe_audio_endpoint as transcribe_audio_endpoint_model,
)

DEBUG = False
bp = BlueprintOne(__name__)


@bp.route(
    '/chatbot',
    methods=['POST'],
    authorizor=request_authentication(),
)
def ai_chatbot_endpoint(
    request: AuthorizedRequest,
    other_params: Optional[dict] = None
) -> Response:
    """
    This function is the endpoint for the AI chatbot.
    It takes in a request and other parameters,
    logs the request, retrieves the user data, and runs the
    conversation with the AI chatbot.
    It then returns the AI chatbot's response.

    :param request: The request from the user.
    :param other_params: Other parameters that may be needed.
    :return: The response from the AI chatbot.
    """
    return ai_chatbot_endpoint_model(
        request=request,
        blueprint=bp,
        other_params=other_params)


@bp.route(
    '/image_to_text',
    methods=['POST'],
    authorizor=request_authentication(),
    content_types=['multipart/form-data']
)
def vision_image_analyzer_endpoint(
    request: AuthorizedRequest,
    other_params: Optional[dict] = None
) -> Response:
    """
    This endpoint receives an image file, saves it to a temporary directory
    with a uuid4 .jpg | .png filename, calls @vision_image_analyzer with
    the file path, and returns the result.

    :param request: The request containing the image file.
    :return: The text with the image analysis.
    """
    return vision_image_analyzer_endpoint_model(
        request=request,
        blueprint=bp,
        other_params=other_params)


@bp.route(
    '/voice_to_text',
    methods=['POST'],
    authorizor=request_authentication(),
    content_types=['multipart/form-data']
)
def transcribe_audio_endpoint(
    request: AuthorizedRequest,
    other_params: Optional[dict] = None
) -> Response:
    """
    This endpoint receives an audio file, saves it to a temporary directory
    with a uuid4 .mp3 filename, calls @audio_to_text_transcript with
    the file path, and returns the result.

    :param request: The request containing the audio file.
    :return: The transcription result.
    """
    return transcribe_audio_endpoint_model(
        request=request,
        blueprint=bp,
        other_params=other_params)
