"""AI Audio Library"""
from typing import Any, Optional, Dict
import os
from uuid import uuid4

import urllib.request
import urllib.parse

from openai import OpenAI
from openai.types.audio.transcription import Transcription

from langchain.agents import tool
from pydantic import BaseModel, Field

from genericsuite.util.aws import upload_nodup_file_to_s3, remove_from_s3
from genericsuite.util.utilities import (
    get_default_resultset,
    error_resultset,
    is_an_url,
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import CommonAppContext

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.ai_langchain_tools import (
    interpret_tool_params,
)
from genericsuite_ai.lib.ai_utilities import (
    gpt_func_error,
    get_user_lang_code,
)
from genericsuite_ai.lib.clarifai import (
    clarifai_audio_to_text,
    clarifai_text_to_audio,
)
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities


DEBUG = False
cac = CommonAppContext()


class AudioToText(BaseModel):
    """
    Transcript audio parameters structure
    """
    sound_filespec: str = Field(description="Sound file path")
    source_lang: str = Field(description="Audio language. Defaults to None")
    other_options: str = Field(description="Other options for the model")


class TextToAudio(BaseModel):
    """
    Text-to-audio parameters structure
    """
    input_text: str = Field(description="text to speech out")
    target_lang: Optional[str] = Field(
        description="target language. Defaults to user's preferred language")
    other_options: Optional[dict] = Field(
        description="Other options for the model")


# Audio-to-text / Speech-to-text


def process_audio_file(sound_filespec: str, call_to: callable,
                       params: dict) -> Transcription:
    """
    Get the audio file from the given sound_filespec (local path or URL),
    process it calling the given "call_to" callable, passing the "file"
    parameter, the "params" additional parameters, and finally returns
    its response.

    Args:
        sound_filespec (str): The URL or path to the sound file.
        call_to (callable): callable function, like OpenAI whisper
            "client.audio.transcriptions.create"
        params (dict): additional parameters for the "call_to" callable.

    Raises:
        Exception: If the sound_filespec is not a valid URL or path.

    Returns:
        Transcription: The transcription of the audio file.
    """
    if is_an_url(sound_filespec):
        headers = {}
        request = urllib.request.Request(sound_filespec, headers=headers)
        with urllib.request.urlopen(request) as audio_file:
            return call_to(
                file=audio_file,
                **params,
            )
    else:
        with open(sound_filespec, "rb") as audio_file:
            return call_to(
                file=audio_file,
                **params,
            )


def process_audio_url(sound_filespec: str, call_to: callable,
                      params: dict, url_par_name: str,
                      rm_after_proc: bool = True) -> Transcription:
    """
    Get the audio file from the given sound_filespec (local path or URL),
    if it's a local file, uploads it to AWS S3, process it calling the given
    "call_to" callable, passing the "url_par_name" parameter, the "params"
    additional parameters, and finally returns its response.

    Args:
        sound_filespec (str): The URL or path to the sound file.
        call_to (callable): callable function, like OpenAI whisper
            "client.audio.transcriptions.create"
        params (dict): additional parameters for the "call_to" callable.
        url_par_name (str): The name of the parameter to pass the URL.
        rm_after_proc (bool): Whether to remove the file after processing.
            Defaults to True.

    Raises:
        Exception: If the sound_filespec is not a valid URL or path.

    Returns:
        Transcription: The transcription of the audio file.
    """
    settings = Config(cac.get())
    resultset = get_default_resultset()
    user_id = cac.app_context.get_user_id()
    if is_an_url(sound_filespec):
        params[url_par_name] = sound_filespec
        resultset = call_to(**params)
    else:
        bucket_name = settings.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET
        if DEBUG:
            log_debug('process_audio_url | ' +
                      f'AWS_S3_CHATBOT_ATTACHMENTS_BUCKET: {str(bucket_name)}')
        if not bucket_name:
            resultset["error"] = True
            resultset["error_message"] = \
                "AWS_S3_CHATBOT_ATTACHMENTS_BUCKET is not configured [1]"
        else:
            upload_result = upload_nodup_file_to_s3(
                file_path=sound_filespec,
                original_filename=os.path.basename(sound_filespec),
                bucket_name=bucket_name,
                sub_dir=user_id,
            )
            if upload_result['error']:
                resultset["error"] = True
                resultset["error_message"] = upload_result["error_message"]
            else:
                params[url_par_name] = upload_result['public_url']
                resultset = call_to(**params)
                if rm_after_proc:
                    remove_result = remove_from_s3(
                        bucket_name=bucket_name,
                        key=f"{user_id}/{upload_result['final_filename']}",
                    )
                    if remove_result['error']:
                        resultset["error"] = True
                        resultset["error_message"] = \
                            remove_result["error_message"]
    return resultset


def get_att_name() -> str:
    """ Returns the Audio-to-text configured technology name """
    settings = Config(cac.get())
    if settings.AI_AUDIO_TO_TEXT_TECHNOLOGY == "clarifai":
        model_name = "Clarifai Audio-to-text: " + \
            f"{settings.AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL}"
    else:
        model_name = "OpenAI Whisper"
    log_debug("get_att_name | AI_AUDIO_TO_TEXT_TECHNOLOGY:" +
              f" {settings.AI_AUDIO_TO_TEXT_TECHNOLOGY}"
              f" | model_name: {model_name}")
    return model_name


def get_att_response(
    sound_filespec: str,
    source_lang: str,
    other_options: str,
    response: dict,
) -> dict:
    """
    Returns the audio-to-text configured technology response

    Args:
        sound_filespec (str): sound file path
        source_lang (str): source language
        other_options (str): other options
        response (dict): response dict

    Returns:
        dict: standard resultset
    """
    settings = Config(cac.get())
    billing = BillingUtilities(cac.get())
    log_debug("get_att_response | AI_AUDIO_TO_TEXT_TECHNOLOGY:" +
              f" {settings.AI_AUDIO_TO_TEXT_TECHNOLOGY}")
    try:
        if settings.AI_AUDIO_TO_TEXT_TECHNOLOGY == "clarifai":
            audio_proc_par: Dict[str, Any] = {}
            transcript = process_audio_url(
                sound_filespec=sound_filespec,
                call_to=clarifai_audio_to_text,
                params=audio_proc_par,
                url_par_name="audio_url",
                rm_after_proc=True,
            )
            log_debug(f"get_att_response | {get_att_name()}" +
                      f" response: {transcript}")
            if transcript["error"]:
                response['error'] = True
                response['error_message'] = transcript["error_message"]
            else:
                response['response'] = transcript["resultset"]
        else:
            openai_api_key = billing.get_openai_api_key()
            if not openai_api_key:
                response['error'] = True
                response['error_message'] = \
                    "OpenAI API key is not configured [AI-ATTT-020]"
                return response
            client = OpenAI(
                api_key=openai_api_key
            )
            audio_proc_par = {
                "model": settings.OPENAI_VOICE_MODEL,   # whisper-1
            }
            if source_lang != "auto":
                audio_proc_par["language"] = source_lang
            if "no_mp3_native_support" in other_options:
                audio_proc_par["prompt"] = \
                    "the audio file comes from an iOS/Apple device." + \
                    " Probably it has codec AAC, which sometimes" + \
                    " result in confused transcriptions. Proceed accordingly"
            if DEBUG:
                log_debug(f"audio_proc_par: {audio_proc_par}")
            transcript = process_audio_file(
                sound_filespec=sound_filespec,
                call_to=client.audio.transcriptions.create,
                params=audio_proc_par
            )
            log_debug("get_att_response | OpenAI GPT Audio-to-text" +
                      f" response: {transcript}")
            response['response'] = transcript.text
    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [AI-ATTT-010]: {str(error)}"
    return response


def audio_to_text_transcript(params: Any) -> dict:
    """
    Transcribe an audio file with OpenAI Whisper or Clarifai speech models.
        (Previously named: transcript_whisper)

    Args:
        params (dict): parameters for the function. It must contain:
            sound_filespec (str): sound file path.
            source_lang (str): audio language. Defaults to None, meaning
                the model have to detect the audio language.
            other_options (str): other options for the model.
                Options:
                "no_mp3_native_support" = the audio file comes from a
                    iOS Apple device. Probably it has codec AAC, which
                    sometimes result in confused transcriptions. So the model
                    is warned and instructed to proceed accordingly.
                Defaults to ''.

    Returns:
        dict: a standard response with the question answered by the model.
    """
    # Reference:
    # https://platform.openai.com/docs/guides/speech-to-text/transcriptions
    params = interpret_tool_params(tool_params=params,
                                   first_param_name="sound_filespec",
                                   schema=AudioToText)

    sound_filespec = params.sound_filespec
    source_lang = params.source_lang
    other_options = params.other_options

    if not other_options:
        other_options = ''

    response = get_default_resultset()
    if not sound_filespec:
        response["error"] = True
        response["error_message"] = "sound_filespec parameter" + \
            " should be provided"
        return response
    if not source_lang:
        source_lang = "auto"
    if DEBUG:
        log_debug("audio_to_text_transcript | Transcribe an audio file" +
                  " with OpenAI Whisper.")
        log_debug(f"Audio file: {sound_filespec}")
        log_debug(f"Source language: {source_lang}")

    response = get_att_response(
        sound_filespec=sound_filespec,
        source_lang=source_lang,
        other_options=other_options,
        response=response,
    )
    if DEBUG:
        log_debug("")
        log_debug(f"audio_to_text_transcript | response: {response}")
    return response


@tool
def audio_processing_text_response(params: Any) -> str:
    """
Useful when you need to transcribe audio files with an audio to text generator.
Args: params (dict): Tool parameters. It must have: "sound_filespec" (str): sound file path.
    """
    return audio_processing_text_response_func(params)


def audio_processing_text_response_func(params: Any) -> str:
    """
    Transcribe audio files with an audio to text generator.

    Args:
        params (dict): Tool parameters. It must have:
            "sound_filespec" (str): sound file path.

    Returns:
        str: Audio to text transcription or [FUNC+ERROR] {error_message}
    """
    model_response = audio_to_text_transcript(params)
    if model_response["error"]:
        response = gpt_func_error(model_response["error_message"])
    else:
        response = model_response["response"]

    if DEBUG:
        log_debug("Text formatted answer from" +
                  " AUDIO_PROCESSING_TEXT_RESPONSE:")
        log_debug(response)
    return response


# Text-to-audio / text-to-speech


def get_tta_name() -> str:
    """ Returns the Text-to-audio configured technology name """
    settings = Config(cac.get())
    if settings.AI_TEXT_TO_AUDIO_TECHNOLOGY == "clarifai":
        model_name = "Clarifai text-to-audio: " + \
            f"{settings.AI_CLARIFAI_DEFAULT_TEXT_TO_AUDIO_MODEL}"
    elif settings.AI_TEXT_TO_AUDIO_TECHNOLOGY == "openai":
        model_name = "OpenAI text-to-speech: " + \
            f"{settings.OPENAI_TEXT_TO_AUDIO_MODEL}"
    else:
        model_name = "N/A"
    log_debug("get_tta_name | AI_TEXT_TO_AUDIO_TECHNOLOGY:" +
              f" {settings.AI_TEXT_TO_AUDIO_TECHNOLOGY}"
              f" | model_name: {model_name}")
    return model_name


def openai_text_to_audio(
    text_source: dict,
    target_lang: str = None,
    other_options: dict = None,
) -> dict:
    """
    Create speech and voices using the OpenAI Text to Speech model.

    Args:
        text_source (dict): Select embedding text source. Options are:
            "raw_text" = raw text
        target_lang (str): target language. Defaults None.
        other_options (dict): other options. Defaults to None.

    Returns:
        dict: dict with the audio file path in the "resultset" attribute.
            If it's an error, sets "error" and "error_message".
    """

    _ = DEBUG and log_debug(
        'OAI_TTA_1) OPENAI_TEXT_TO_AUDIO' +
        f' | text_source: {text_source}' +
        f' | target_lang: {target_lang}' +
        f' | other_options: {other_options}')

    settings = Config(cac.get())
    billing = BillingUtilities(cac.get())
    response = get_default_resultset()

    try:
        input_text = text_source.get("raw_text")
        if not input_text:
            return error_resultset("No text supplied", "OAI_TTA_E010")
        model = settings.OPENAI_TEXT_TO_AUDIO_MODEL
        voice = settings.OPENAI_TEXT_TO_AUDIO_VOICE
        speech_file_path = f'/tmp/openai_tts_{uuid4().hex}.mp3'
        openai_api_key = billing.get_openai_api_key()
        if not openai_api_key:
            return error_resultset(
                "OpenAI API key is not configured",
                "OAI_TTA_E020")
        client = OpenAI(
            api_key=openai_api_key
        )
    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [OAI_TTA_E015]: {str(error)}"

    try:
        tts_response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=input_text
        )
    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [OAI_TTA_E030]: {str(error)}"

    try:
        tts_response.stream_to_file(speech_file_path)
    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [OAI_TTA_E040]: {str(error)}"

    response["resultset"] = speech_file_path
    return response


def get_tta_response(
    input_text: str,
    target_lang: str,
    other_options: dict,
    response: dict,
) -> dict:
    """
    Returns the Text-to-audio configured technology response

    Args:
        input_text (str): text to speech out.
        target_lang (str): target language. Defaults to user's
            preferred language.
        other_options (dict): other options.
        response (dict): response dict

    Returns:
        dict: standard resultset
    """
    settings = Config(cac.get())
    log_debug("get_tta_response | AI_TEXT_TO_AUDIO_TECHNOLOGY:" +
              f" {settings.AI_TEXT_TO_AUDIO_TECHNOLOGY}" +
              f"\n | input_text: {input_text}" +
              f"\n | target_lang: {target_lang}" +
              f"\n | other_options: {other_options}")
    try:
        if other_options.get("mock_response"):
            response['response'] = other_options.get("mock_response")
        elif settings.AI_TEXT_TO_AUDIO_TECHNOLOGY == "clarifai":
            audio_file = clarifai_text_to_audio(
                text_source={"raw_text": input_text},
                target_lang=target_lang,
                other_options=other_options,
            )
            log_debug(f"get_tta_response (1) | {get_tta_name()}" +
                      f" response: {audio_file}")
            # audio_file={'error': False, 'error_message': None,
            # 'totalPages': None,
            # 'resultset': '/tmp/0f051624e0074b4aaecf2402911c2c86.wav'}
            if audio_file["error"]:
                response['error'] = True
                response['error_message'] = audio_file['error_message']
            else:
                response['response'] = audio_file["resultset"]
        elif settings.AI_TEXT_TO_AUDIO_TECHNOLOGY == "openai":
            # https://platform.openai.com/docs/guides/text-to-speech
            audio_file = openai_text_to_audio(
                text_source={"raw_text": input_text},
                target_lang=target_lang,
                other_options=other_options,
            )
            log_debug(f"get_tta_response (2) | {get_tta_name()}" +
                      f" response: {audio_file}")
            if audio_file["error"]:
                response['error'] = True
                response['error_message'] = audio_file['error_message']
            else:
                response['response'] = audio_file["resultset"]
            _ = DEBUG and log_debug('OAI_TTA_7) OPENAI_TEXT_TO_AUDIO')
        else:
            response['error'] = True
            response['error_message'] = \
                "ERROR [AI-TTAG-020]: Invalid audio-to-text technology:" + \
                f" {settings.AI_TEXT_TO_AUDIO_TECHNOLOGY}"
    except Exception as error:
        response['error'] = True
        response['error_message'] = f"ERROR [AI-TTAG-010]: {str(error)}"
    return response


def text_to_audio_generator(params: Any) -> dict:
    """
    Generate an audio file from a given text.

    Args:
        params (dict): parameters for the function. It must contain:
            "input_text" (str): text to speech out.
            "target_lang" (Optional[str]): target language.
                Defaults to user's preferred language.
            "other_options" (Optional[dict]): other options. Available options:
                "speaker_voice": "male" or "female". Defaults to female.
                "mock_response": local file path with a mocked audio file.
                    e.g. '/tmp/9acd5e877ac44980b3604069e9b0d8df.wav'

    Returns:
        dict: a standard response with the model response.
    """
    # Reference:
    # https://clarifai.com/eleven-labs/audio-generation/models/speech-synthesis
    params = interpret_tool_params(tool_params=params,
                                   first_param_name="input_text",
                                   schema=TextToAudio)

    input_text = params.input_text
    target_lang = params.target_lang
    other_options = params.other_options

    if not other_options:
        other_options = {}
        mock_file_example = None
        # mock_file_example = '/tmp/xxx.wav'
        # mock_file_example = "/tmp/openai_tts_xxxxx.mp3"
        if mock_file_example and os.path.isfile(mock_file_example):
            other_options["mock_response"] = mock_file_example

    if not target_lang:
        target_lang = get_user_lang_code(cac.get())

    response = get_default_resultset()
    if not input_text:
        response["error"] = True
        response["error_message"] = "input_text parameter" + \
            " should be provided"
        return response
    if DEBUG:
        log_debug("1) TTAG | text_to_audio_generator" +
                  " | Generate an audio file from text." +
                  f"\ninput_text: {input_text}" +
                  f"\ntarget_lang: {target_lang}" +
                  f"\nother_options: {other_options}")
    response = get_tta_response(
        input_text=input_text,
        target_lang=target_lang,
        other_options=other_options,
        response=response,
    )
    if DEBUG:
        log_debug(f"2) TTAG | text_to_audio_generator | response: {response}")
    return response


# @tool("text_to_audio_response", return_direct=True, args_schema=TextToAudio)
@tool("text_to_audio_response", return_direct=True)
def text_to_audio_response(params: Any) -> str:
    """
Useful when you need to generate audio files from a given text. Call this tool when the Human question includes one of these text:
"/TTS:", "/tts:", "Say it:", "Say it loud:", "Speak it:", "Speak it loud:", "Dimelo:", "Dime esto:", "Di esto en voz alta:", "Di este texto:", "Hablame:", "Habla esto:", "habla este texto:", etc.
Return exactly what this Tool returns, with no added comments. E.g. [SEND_FILE_BACK]=/tmp/openai_tts_`uuid4`.mp3
Args: params (dict): Tool parameters. It must have: "input_text" (str): text to speech out. Don't translate it!
    """
    return text_to_audio_response_func(params)


def text_to_audio_response_func(params: Any) -> str:
    """
    Generate an audio file from a given text (OpenAI GPT Function).

    Args:
        params (dict): Tool parameters. It must have:
            "input_text" (str): text to speech out. Don't translate it!
            "target_lang" (Optional[str]): target language.
                Defaults to user's preferred language.
            "other_options" (Optional[dict]): other options. Available options:
                "speaker_voice": "male" or "female". Defaults to female.
                "mock_response": local file path with a mocked audio file.
                    e.g. '/tmp/9acd5e877ac44980b3604069e9b0d8df.wav'

    Returns:
        str: A local file path to send back to the user.
            e.g. [SEND_FILE_BACK]=/tmp/9acd5e877ac44980b3604069e9b0d8df.wav
            or [FUNC+ERROR] {error_message}
    """
    model_response = text_to_audio_generator(params)
    if model_response["error"]:
        response = gpt_func_error(model_response["error_message"])
    else:
        response = f'[SEND_FILE_BACK]={model_response["response"]}'

    if DEBUG:
        log_debug("Answer from TEXT_TO_AUDIO_GENERATOR:")
        log_debug(response)

    return response
