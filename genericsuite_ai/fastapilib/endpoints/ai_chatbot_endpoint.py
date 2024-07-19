"""
AI Endpoints for FastAPI
"""
import os
from typing import Union, Any
# from typing import Annotated

from pydantic import BaseModel

# from fastapi import APIRouter, Query, Form
from fastapi import Depends, File, UploadFile, Body, Request as FaRequest
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic

# from genericsuite.util.framework_abs_layer import Response, BlueprintOne
from genericsuite.fastapilib.framework_abstraction import (
    Response,
    BlueprintOne,
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import (
    send_file_text_text,
)

from genericsuite.fastapilib.util.parse_multipart import download_file_fa
from genericsuite.fastapilib.util.dependencies import (
    get_current_user,
    get_default_fa_request,
)

from genericsuite_ai.lib.ai_chatbot_endpoint import (
    ai_chatbot_endpoint as ai_chatbot_endpoint_model,
    vision_image_analyzer_endpoint as vision_image_analyzer_endpoint_model,
    transcribe_audio_endpoint as transcribe_audio_endpoint_model,
)

# from app.internal.ai_gpt_fn_index import (
#     assign_app_gpt_functions
# )


class VisionImageAnalyzerRequest(BaseModel):
    """
    This class is used to parse the query parameters for the
    vision_image_analyzer endpoint.
    """
    question: str
    file_name: str
    cid: Union[str, None] = None


class TranscribeAudioRequest(BaseModel):
    """
    This class is used to parse the query parameters for the
    transcribe_audio_endpoint endpoint.
    """
    extension: str
    source_lang: str
    other: str


DEBUG = False

# send_file_fa() mode
SEND_FILE_AS_BINARY = False

# Set FastAPI router
# router = APIRouter()
router = BlueprintOne()

# Set up Basic Authentication
security = HTTPBasic()

# https://fastapi.tiangolo.com/tutorial/request-files/#file-parameters-with-uploadfile
# https://fastapi.tiangolo.com/tutorial/request-forms-and-files/


def remove_temp_file(file_path: str) -> None:
    """ Remove the temp file """
    _ = DEBUG and log_debug(f"Removing temp file: {file_path}")
    os.remove(file_path)


def send_file_fa(
    file_to_send: str,
    background_tasks: BackgroundTasks,
) -> Any:
    """
    Send the file back, the FastAPI way
    """
    if SEND_FILE_AS_BINARY:
        return send_binary_file_fa(file_to_send, background_tasks)
    return send_base64_file_fa(file_to_send, background_tasks)


def send_binary_file_fa(
    file_to_send: str,
    background_tasks: BackgroundTasks,
) -> FileResponse:
    """
    Send the file back, the FastAPI way
    """
    background_tasks.add_task(remove_temp_file, file_path=file_to_send)
    _ = DEBUG and log_debug(f"Temp file read | file_to_send: {file_to_send}")
    # Return the file content the standard FastAPI way
    # https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse
    _ = DEBUG and log_debug("Returning file content as FileResponse")
    return FileResponse(file_to_send)


def send_base64_file_fa(
    file_to_send: str,
    background_tasks: BackgroundTasks,
) -> Any:
    """
    Return the file content as GenericSuite way, Base64 encoded.
    This approach worked for audio file and the ai_chatbot in Chalice
    because AWS API Gateway won't allow binary responses.
    """
    _ = DEBUG and log_debug("Returning file content the Genericsuite way")
    return send_file_text_text(file_to_send)


@router.post('/chatbot', tags='chatbot')
async def ai_chatbot_endpoint(
    request: FaRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    conversation: str = Body(None),
    cid: str = Body(None),
) -> Any:
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
    json_body = {
        'conversation': conversation,
        'cid': cid,
    }
    _ = DEBUG and \
        log_debug(f'ai_chatbot_endpoint | json_body: {json_body}')
    gs_request, other_params = get_default_fa_request(
        current_user=current_user, json_body=json_body)
    router.set_current_request(request, gs_request)
    return ai_chatbot_endpoint_model(
        request=gs_request,
        blueprint=router,
        other_params=other_params,
        # additional_callable=assign_app_gpt_functions,
        sendfile_callable=send_file_fa,
        background_tasks=background_tasks,
    )


@router.post('/image_to_text')
async def vision_image_analyzer_endpoint(
    request: FaRequest,
    file: UploadFile = File(...),
    # file: Annotated[UploadFile, File(...)] = None,
    current_user: str = Depends(get_current_user),
    query_params: VisionImageAnalyzerRequest = Depends(),
) -> Response:
    """
    This endpoint receives an image file, saves it to a temporary directory
    with a uuid4 .jpg | .png filename, calls @vision_image_analyzer with
    the file path, and returns the result.

    :param file: The image file uploaded by the user.
    :return: The text with the image analysis.
    """
    _ = DEBUG and \
        log_debug(
            '1) vision_image_analyzer_endpoint | query_params:' +
            f' {request.query_params}')
    gs_request, other_params = get_default_fa_request(
        current_user=current_user, query_params=query_params.dict(),
        preserve_nones=True)
    router.set_current_request(request, gs_request)
    uploaded_file_path = await download_file_fa(file)
    _ = DEBUG and \
        log_debug(
            '2) vision_image_analyzer_endpoint | uploaded_file_path:' +
            f' {uploaded_file_path} | request: {request}')
    return vision_image_analyzer_endpoint_model(
        request=gs_request,
        blueprint=router,
        other_params=other_params,
        # additional_callable=assign_app_gpt_functions,
        uploaded_file_path=uploaded_file_path,
    )


@router.post('/voice_to_text')
async def transcribe_audio_endpoint(
    request: FaRequest,
    file: UploadFile = File(...),
    # file: Annotated[UploadFile, File()] = None,
    current_user: str = Depends(get_current_user),
    query_params: TranscribeAudioRequest = Depends(),
) -> Response:
    """
    This endpoint receives an audio file, saves it to a temporary directory
    with a uuid4 .mp3 filename, calls @audio_to_text_transcript with
    the file path, and returns the result.

    :param request: The request containing the audio file.
    :return: The transcription result.
    """
    _ = DEBUG and \
        log_debug(
            'transcribe_audio_endpoint | query_params:' +
            f' {request.query_params}')
    gs_request, other_params = get_default_fa_request(
        current_user=current_user, query_params=query_params.dict(),
        preserve_nones=True)
    router.set_current_request(request, gs_request)
    uploaded_file_path = await download_file_fa(file)
    return transcribe_audio_endpoint_model(
        request=gs_request,
        blueprint=router,
        other_params=other_params,
        # additional_callable=assign_app_gpt_functions,
        uploaded_file_path=uploaded_file_path,
    )
