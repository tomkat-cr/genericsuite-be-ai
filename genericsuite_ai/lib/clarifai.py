"""
Clarifai platform utilities
"""
import json
import os
from uuid import uuid4

from clarifai.client.model import Model
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from genericsuite_ai.config.config import Config

from genericsuite.util.app_context import CommonAppContext
from genericsuite.util.app_logger import log_debug, log_error
from genericsuite.util.aws import upload_nodup_file_to_s3
from genericsuite.util.generic_db_middleware import (
    fetch_all_from_db,
)
from genericsuite.util.utilities import get_default_resultset

DEBUG = False
cac = CommonAppContext()


def guess_model_manufacturer(model_data: dict) -> str:
    """
    Guess model manufacturer from model name.
    """
    if DEBUG:
        log_debug('CGMM-1) Clarifai - guess_model_manufacturer' +
                  f"| model_data: {model_data}")
    model_name = model_data.get("model_name", "")
    user_id = model_data.get("user_id", "")
    if "goole" in user_id:
        result = "Google"
    elif "microsoft" in user_id:
        result = "Microsoft"
    elif "facebook" in user_id or "meta" in user_id:
        result = "Meta"
    elif "openai" in user_id:
        result = "OpenAI"
    else:
        if model_name.startswith("general-"):
            result = "Clarifai"
        else:
            result = None
    if DEBUG:
        log_debug('CGMM-2) Clarifai - guess_model_manufacturer |' +
                  f' result: {result}')
    return result


def get_model_config(model_name: str, include_all: bool = True) -> dict:
    """
    Get Clarifai model configuration from database.
    """
    if DEBUG:
        log_debug('CGMC-1) Clarifai - get_model_config' +
                  f"| model_name: {model_name}")
    result = {}
    resultset = get_model_config_raw(model_name)
    if resultset["error"]:
        result["error"] = True
        result["error_message"] = resultset["error_message"]
    else:
        resultset["resultset"] = resultset["resultset"][0]
        if include_all:
            result = resultset["resultset"]
            if not result.get("manufacturer"):
                result["manufacturer"] = guess_model_manufacturer(result)
        else:
            result["user_id"] = resultset["resultset"]["user_id"]
            result["app_id"] = resultset["resultset"]["app_id"]
            result["model_id"] = resultset["resultset"]["model_id"]
            result["model_version_id"] = \
                resultset["resultset"]["model_version_id"]
    if DEBUG:
        log_debug('CGMC-2) Clarifai - get_model_config |' +
                  f' result: {result}')
    return result


def get_model_config_raw(model_name: str) -> dict:
    """
    Get Clarifai model configuration from database (raw version).
    """
    if DEBUG:
        log_debug('CGMC-1) Clarifai - get_model_config_raw' +
                  f"| model_name: {model_name}")
    resultset = fetch_all_from_db(
        app_context=cac.app_context,
        json_file='clarifai_models',
        like_query_params={
            "model_name": model_name,
            "active": "1",
        },
        combinator="$and",
    )
    if not resultset["error"]:
        resultset["resultset"] = json.loads(resultset["resultset"])
    if not resultset["resultset"]:
        resultset = get_default_resultset()
        resultset["error"] = True
        resultset["error_message"] = \
            f"Clarifai model '{model_name}' not found [CGMC-E1]"
    if DEBUG:
        log_debug('CGMC-2) Clarifai - get_model_config_raw |' +
                  f' resultset: {resultset}')
    return resultset


# Clarifai Image Recognition

def get_vision_model_type(model_name: str) -> str:
    """
    Get clarifai vision model type.
    """
    model_type = None
    if model_name in ["openai-gpt-4-vision"]:
        model_type = "completion"
    else:
        model_type = "predict"
    if DEBUG:
        log_debug('GVMT-1) Clarifai - get_vision_model_type' +
                  f' | model_name: {model_name}' +
                  f' | model_type: {model_type}')
    return model_type


def clarifai_vision(
    image_url: str,
    question: str,
    model_name: str = None
) -> dict:
    """
    Image recognition using Clarifai's platform.
    """
    settings = Config(cac.get())
    if not model_name:
        model_name: str = settings.AI_CLARIFAI_DEFAULT_VISION_MODEL
    if DEBUG:
        log_debug('CF-V-1) Clarifai - clarifai_vision' +
                  f'\n | image_url: {image_url}' +
                  f'\n | question: {question}' +
                  f'\n | model_name: {model_name}' +
                  '\n')
    model_config = get_model_config(model_name)
    if model_config.get("error"):
        return model_config
    model_response = clarifai_vision_raw(
        image_url=image_url,
        question=question,
        model_name=model_name,
        model_config=model_config
    )
    resultset = get_default_resultset()
    resultset["resultset"] = model_response
    if DEBUG:
        log_debug('CF-V-2) Clarifai - clarifai_vision |' +
                  f' resultset: {resultset}')
    return resultset


def clarifai_vision_raw(
    image_url: str,
    question: str,
    model_name: str,
    model_config: dict,
):
    """
    Image recognition using the given image url and model config.
    """
    settings = Config(cac.get())
    model_type = get_vision_model_type(model_name)

    if DEBUG:
        log_debug('CF-VRAW-1) Clarifai - clarifai_vision_raw' +
                  f'\n | image_url: {image_url}' +
                  f'\n | question: {question}' +
                  f'\n | model_name: {model_name}' +
                  f'\n | model_type: {model_type}' +
                  f'\n | model_config: {model_config}' +
                  '\n')

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + settings.CLARIFAI_PAT),)

    userDataObject = resources_pb2.UserAppIDSet(
        user_id=model_config["user_id"],
        app_id=model_config["app_id"],
    )

    inputs = []
    if model_type == "completion":
        inputs.append(
            resources_pb2.Input(
                data=resources_pb2.Data(
                    text=resources_pb2.Text(
                        raw="{question} {image_url}"
                        # url=TEXT_FILE_URL
                        # raw=file_bytes
                    )
                )
            )
        )
    else:
        inputs.append(
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url=image_url
                    )
                )
            )
        )

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            # The userDataObject is created in the overview and is
            # required when using a PAT
            user_app_id=userDataObject,
            model_id=model_config["model_id"],
            # This is optional. Defaults to the latest model version
            version_id=model_config["model_version_id"],
            inputs=inputs
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        log_error(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " +
                        post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    if model_type == "completion":
        result = output.data.text.raw
    else:
        if hasattr(output.data, 'regions'):
            elements = []
            for region in output.data.regions:
                # Accessing and rounding the bounding box values
                top_row = round(region.region_info.bounding_box.top_row, 3)
                left_col = round(region.region_info.bounding_box.left_col, 3)
                bottom_row = \
                    round(region.region_info.bounding_box.bottom_row, 3)
                right_col = round(region.region_info.bounding_box.right_col, 3)
                for concept in region.data.concepts:
                    # Accessing and rounding the concept value
                    name = concept.name
                    value = round(concept.value, 4)
                    elements.append(f"{name}: {value} BBox: {top_row}," +
                                    f" {left_col}, {bottom_row}, {right_col}")
            result = \
                "The image is composed by these regions: " + \
                "\n".join(elements)
        elif hasattr(output.data, 'concepts'):
            result = \
                "The image is composed by these elements: " + \
                "\n".join(
                    [f"{concept.name}: {concept.value:.2f}"
                     for concept in output.data.concepts]
                )
        else:
            raise AttributeError("No 'concepts' or 'regions' attributes in" +
                                 " the output data.")

    if DEBUG:
        if model_type == "completion":
            log_debug('CF-VRAW-2) Clarifai - clarifai_vision_raw' +
                      " | Completion:\n" + result)
        else:
            log_debug(
                'CF-VRAW-3) Clarifai - clarifai_vision_raw' +
                " | Predicted concepts:\n" + result)
        # Uncomment this line to print the full Response JSON
        log_debug('CF-VRAW-4) Clarifai - clarifai_vision_raw' +
                  ' | Full response')
        log_debug(output)
    return result


# Clarifai Image Generation


def clarifai_img_gen(
    question: str,
    model_name: str = None
) -> dict:
    """
    Text to image generation using Clarifai's platform.
    """
    settings = Config(cac.get())
    if not model_name:
        model_name: str = settings.AI_CLARIFAI_DEFAULT_IMG_GEN_MODEL
    if DEBUG:
        log_debug('CF-IG-1) Clarifai - clarifai_img_gen' +
                  f'\n | question: {question}' +
                  f'\n | model_name: {model_name}' +
                  '\n')
    model_config = get_model_config(model_name)
    if model_config.get("error"):
        return model_config
    model_response = clarifai_img_gen_raw(
        question=question,
        model_name=model_name,
        model_config=model_config
    )
    resultset = get_default_resultset()
    resultset["resultset"] = model_response
    if DEBUG:
        log_debug('CF-IG-2) Clarifai - clarifai_img_gen |' +
                  f' resultset: {resultset}')
    return resultset


def clarifai_img_gen_raw(
    question: str,
    model_name: str,
    model_config: dict,
):
    """
    Image recognition using the given image url and model config.
    """
    settings = Config(cac.get())

    if DEBUG:
        log_debug('CF-IGRAW-1) Clarifai - clarifai_img_gen_raw' +
                  f'\n | question: {question}' +
                  f'\n | model_name: {model_name}' +
                  f'\n | model_config: {model_config}' +
                  '\n')

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + settings.CLARIFAI_PAT),)

    userDataObject = resources_pb2.UserAppIDSet(
        user_id=model_config["user_id"],
        app_id=model_config["app_id"],
    )

    # To use a local text file, uncomment the following lines
    # with open(TEXT_FILE_LOCATION, "rb") as f:
    #    file_bytes = f.read()

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            # The userDataObject is created in the overview and is required
            # when using a PAT
            user_app_id=userDataObject,
            model_id=model_config["model_id"],
            # This is optional. Defaults to the latest model version
            version_id=model_config["model_version_id"],
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=question
                            # url=TEXT_FILE_URL
                            # raw=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        log_error(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " +
                        post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0].data.image.base64

    bucket_name = settings.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET
    sub_dir = cac.app_context.get_user_id()
    original_filename = "gen-image.jpg"

    image_filename = f"/tmp/{original_filename}"
    with open(image_filename, 'wb') as f:
        f.write(output)

    upload_result = upload_nodup_file_to_s3(
        file_path=image_filename,
        original_filename=original_filename,
        bucket_name=bucket_name,
        sub_dir=sub_dir,
    )
    if upload_result['error']:
        raise Exception("AWS S3 upload error:: " + upload_result['error'])

    result = upload_result['public_url']
    os.remove(image_filename)  # Clean up the temporary file

    if DEBUG:
        log_debug('CF-IGRAW-2) Clarifai - clarifai_img_gen_raw' +
                  " | result: " + result)
        # Uncomment this line to print the full Response JSON
        # log_debug('CF-IGRAW-4) Clarifai - clarifai_img_gen_raw' +
        #           ' | Full response')
        # log_debug(post_model_outputs_response)
    return result


# Clarifai text embeddings


def clarifai_embeddings(
    embed_source: dict,
    model_name: str = None
) -> dict:
    """
    Embeddings using Clarifai's platform.

    Args:
        embed_source (dict): embedding text source. Options are:
            "raw_text" = raw text
            "text_file_url" = text file url
            "text_file_location" = text local file location
        model_name (str): model name stored in the "clarifai_models" table.
            Defaults to None.

    Returns:
        dict: resultset.
    """
    settings = Config(cac.get())
    if not model_name:
        model_name: str = settings.AI_CLARIFAI_DEFAULT_TEXT_EMBEDDING_MODEL
    if DEBUG:
        log_debug('CF-EM-1) Clarifai - clarifai_embeddings' +
                  f'\n | embed_source: {embed_source}' +
                  f'\n | model_name: {model_name}' +
                  '\n')
    model_config = get_model_config(model_name)
    if model_config.get("error"):
        return model_config
    model_response = clarifai_embeddings_raw(
        embed_source=embed_source,
        model_name=model_name,
        model_config=model_config
    )
    resultset = get_default_resultset()
    if "[ERROR-" in model_response:
        resultset["error"] = model_response
    else:
        resultset["resultset"] = model_response
    if DEBUG:
        log_debug('CF-EM-2) Clarifai - clarifai_embeddings |' +
                  f' resultset: {resultset}')
    return resultset


def clarifai_embeddings_raw(
    embed_source: dict,
    model_name: str,
    model_config: dict,
):
    """
    Create embeddings using the given text source and model config.

    Args:
        embed_source (dict): Select embedding text source. Options are:
            "raw_text" = raw text
            "text_file_url" = text file url
            "text_file_location" = text local file location
        model_name (str): model name stored in the "clarifai_models" table.
            Defaults to None
        model_config (dict): model configuration as {"user_id", "app_id",
            "model_id", "model_version_id"}

    Returns:
        Any: model response
    """
    settings = Config(cac.get())

    if DEBUG:
        log_debug('CF-EMRAW-1) Clarifai - clarifai_embeddings_raw' +
                  f'\n | embed_source: {embed_source}' +
                  f'\n | model_name: {model_name}' +
                  f'\n | model_config: {model_config}' +
                  '\n')

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + settings.CLARIFAI_PAT),)

    userDataObject = resources_pb2.UserAppIDSet(
        user_id=model_config["user_id"],
        app_id=model_config["app_id"],
    )

    input_params = {}
    if "raw_text" in embed_source:
        input_params["raw"] = embed_source["raw_text"]
    elif "text_file_url" in embed_source:
        input_params["url"] = embed_source["text_file_url"]
    elif "text_file_location" in embed_source:
        with open(embed_source["text_file_location"], "rb") as f:
            input_params["raw"] = f.read()  # file_bytes
    else:
        return "[ERROR-CF-EMRAW-010] Invalid input source: " + \
            str(embed_source)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            # The userDataObject is created in the overview and is required
            # when using a PAT
            user_app_id=userDataObject,
            model_id=model_config["model_id"],
            # This is optional. Defaults to the latest model version
            version_id=model_config["model_version_id"],
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            **input_params
                            # raw=RAW_TEXT
                            # url=TEXT_FILE_URL
                            # raw=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        log_error(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " +
                        post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    result = output.data.concepts

    if DEBUG:
        log_debug('CF-EMRAW-2) Clarifai - clarifai_embeddings_raw' +
                  " | result: " + result)
        log_debug('CF-EMRAW-3) Clarifai - clarifai_embeddings_raw' +
                  " | Predicted concepts: " +
                  "\n".join([
                        (f"{concept.name} {concept.value}.2f")
                        for concept in output.data.concepts
                    ])
                  )

        # Uncomment this line to print the full Response JSON
        # log_debug('CF-EMRAW-4) Clarifai - clarifai_embeddings_raw' +
        #           ' | Full response')
        # log_debug(post_model_outputs_response)
    return result


# Clarifai audio-to-text


def clarifai_audio_to_text(
    audio_url: str,
    model_name: str = None
) -> dict:
    """
    Audio transcription model for converting speech audio to text
    using Clarifai's platform.

    Args:
        audio_url (str): audio URL source
        model_name (str): model name stored in the "clarifai_models" table.
            Defaults to None.

    Returns:
        dict: resultset.
    """
    settings = Config(cac.get())
    if not model_name:
        model_name: str = settings.AI_CLARIFAI_DEFAULT_AUDIO_TO_TEXT_MODEL
    if DEBUG:
        log_debug('CF-ATT-1) Clarifai - clarifai_audio_to_text' +
                  f'\n | audio_url: {audio_url}' +
                  f'\n | model_name: {model_name}' +
                  '\n')
    model_config = get_model_config(model_name)
    if model_config.get("error"):
        return model_config
    model_response = clarifai_audio_to_text_raw(
        audio_url=audio_url,
        model_name=model_name,
        model_config=model_config
    )
    resultset = get_default_resultset()
    if "[ERROR-" in model_response:
        resultset["error"] = model_response
    else:
        resultset["resultset"] = model_response
    if DEBUG:
        log_debug('CF-ATT-2) Clarifai - clarifai_audio_to_text |' +
                  f' resultset: {resultset}')
    return resultset


def clarifai_audio_to_text_raw(
    audio_url: str,
    model_name: str,
    model_config: dict,
):
    """
    Audio transcription model for converting speech audio to text
    """
    settings = Config(cac.get())

    if DEBUG:
        log_debug('CF-ATTRAW-1) Clarifai - clarifai_audio_to_text_raw' +
                  f'\n | audio_url: {audio_url}' +
                  f'\n | model_name: {model_name}' +
                  f'\n | model_config: {model_config}' +
                  '\n')

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + settings.CLARIFAI_PAT),)

    userDataObject = resources_pb2.UserAppIDSet(
        user_id=model_config["user_id"],
        app_id=model_config["app_id"],
    )

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            # The userDataObject is created in the overview and is required
            # when using a PAT
            user_app_id=userDataObject,
            model_id=model_config["model_id"],
            # This is optional. Defaults to the latest model version
            version_id=model_config["model_version_id"],
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        audio=resources_pb2.Audio(
                            url=audio_url
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        log_error(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " +
                        post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    result = output.data.concepts

    if DEBUG:
        log_debug('CF-ATTRAW-2) Clarifai - clarifai_audio_to_text_raw' +
                  " | result: " + result)
        log_debug('CF-ATTRAW-3) Clarifai - clarifai_audio_to_text_raw' +
                  " | Predicted concepts: " +
                  "\n".join([
                        (f"{concept.name} {concept.value}.2f")
                        for concept in output.data.concepts
                    ])
                  )

        # Uncomment this line to print the full Response JSON
        # log_debug('CF-ATTRAW-4) Clarifai - clarifai_audio_to_text_raw' +
        #           ' | Full response')
        # log_debug(post_model_outputs_response)
    return result


# Clarifai text-to-audio


def clarifai_text_to_audio(
    text_source: dict,
    target_lang: str = None,
    other_options: dict = None,
    model_name: str = None,
    sdk_type: str = None,
) -> dict:
    """
    Create realistic speech and voices using the robust Text to Speech
    and Voice Cloning model using Clarifai's platform.

    Args:
        text_source (dict): Select embedding text source. Options are:
            "raw_text" = raw text
            "text_file_url" = text file url
            "text_file_location" = text local file location
        target_lang (str): target language. Defaults None.
        other_options (dict): other options. Defaults to None.
        model_name (str): model name stored in the "clarifai_models" table.
            Defaults to None.
        sdk_type (str): SDK type. Options are:
            "python_sdk" = Python SDK (requires Eleven Labs API Key)
            "clarifai_grpc" = Clarifai GRPC

    Returns:
        dict: dict with the audio file path in the "resultset" attribute.
            If it's an error, sets "error" and "error_message".
    """
    settings = Config(cac.get())

    if not model_name:
        model_name: str = settings.AI_CLARIFAI_DEFAULT_TEXT_TO_AUDIO_MODEL
    if not sdk_type:
        sdk_type: str = settings.AI_CLARIFAI_AUDIO_TO_TEXT_SDK_TYPE
    if not other_options:
        other_options = {}

    if DEBUG:
        log_debug('CF-TTA-1) Clarifai - clarifai_text_to_audio' +
                  f'\n | text_source: {text_source}' +
                  f'\n | model_name: {model_name}' +
                  '\n')
    model_config = get_model_config(model_name=model_name, include_all=True)
    if model_config.get("error"):
        return model_config

    if sdk_type == "python_sdk":
        model_response = clarifai_text_to_audio_python_sdk(
            text_source=text_source,
            target_lang=target_lang,
            other_options=other_options,
            model_config=model_config
        )
    else:
        model_response = clarifai_text_to_audio_raw(
            text_source=text_source,
            target_lang=target_lang,
            other_options=other_options,
            model_name=model_name,
            model_config=model_config
        )
    resultset = get_default_resultset()
    if "[ERROR-" in model_response:
        resultset["error"] = model_response
    else:
        resultset["resultset"] = model_response
    if DEBUG:
        log_debug('CF-TTA-2) Clarifai - clarifai_text_to_audio |' +
                  f' resultset: {resultset}')
    return resultset


def clarifai_text_to_audio_raw(
    text_source: dict,
    target_lang: str,
    other_options: dict,
    model_name: str,
    model_config: dict,
):
    """
    Audio model for converting text to speech audio using Clarifai GRPC SDK

    Args:
        text_source (dict): Select embedding text source. Options are:
            "raw_text" = raw text
            "text_file_url" = text file url
            "text_file_location" = text local file location
        model_name (str): model name stored in the "clarifai_models" table.
            Defaults to None
        model_config (dict): model configuration as {"user_id", "app_id",
            "model_id", "model_version_id"}

    Returns:
        str: model response as audio file path
    """
    settings = Config(cac.get())

    if DEBUG:
        log_debug('CF-TTARAW-1) Clarifai - clarifai_text_to_audio_raw (GRPC)' +
                  f'\n | model_name: {model_name}' +
                  f'\n | target_lang: {target_lang}' +
                  f'\n | other_options: {other_options}' +
                  f'\n | model_config: {model_config}' +
                  f'\n | text_source: {text_source}' +
                  '\n')

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + settings.CLARIFAI_PAT),)

    userDataObject = resources_pb2.UserAppIDSet(
        user_id=model_config["user_id"],
        app_id=model_config["app_id"],
    )

    input_params = {}
    if "raw_text" in text_source:
        input_params["raw"] = text_source["raw_text"]
    elif "text_file_url" in text_source:
        input_params["url"] = text_source["text_file_url"]
    elif "text_file_location" in text_source:
        with open(text_source["text_file_location"], "rb") as f:
            input_params["raw"] = f.read()  # file_bytes
    else:
        return "[ERROR-CF-TTARAW-010] Invalid input source: " + \
            str(text_source)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            # The userDataObject is created in the overview and is required
            # when using a PAT
            user_app_id=userDataObject,
            model_id=model_config["model_id"],
            # This is optional. Defaults to the latest model version
            version_id=model_config["model_version_id"],
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            **input_params
                            # raw=RAW_TEXT
                            # url=TEXT_FILE_URL
                            # raw=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        log_error(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " +
                        post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    output_base64 = output.data.audio.base64
    tmp_audio_filename = f'/tmp/{uuid4().hex}.wav'
    with open(tmp_audio_filename, 'wb') as f:
        f.write(output_base64)
    result = tmp_audio_filename

    if DEBUG:
        log_debug('CF-TTARAW-2) Clarifai - clarifai_text_to_audio_raw' +
                  " | result: " + result)
        log_debug('CF-TTARAW-3) Clarifai - clarifai_text_to_audio_raw' +
                  " | Predicted concepts: " +
                  "\n".join([
                        (f"{concept.name} {concept.value}.2f")
                        for concept in output.data.concepts
                    ])
                  )

        # Uncomment this line to print the full Response JSON
        # log_debug('CF-TTARAW-4) Clarifai - clarifai_text_to_audio_raw' +
        #           ' | Full response')
        # log_debug(post_model_outputs_response)
    return result


def clarifai_text_to_audio_python_sdk(
    text_source: str,
    target_lang: str,
    other_options: dict,
    model_config: dict,
):
    """
    Audio model for converting text to speech audio using Clarifai Python SDK

    Args:
        text_source (dict): Select embedding text source. Options are:
            "raw_text" = raw text
            "text_file_url" = text file url
            "text_file_location" = text local file location
        target_lang (str): target language
        other_options (dict): other options:
            "speaker_voice": "male" or "female". Defaults to female.

        model_config (dict): model configuration as {"user_id", "app_id",
            "model_id", "model_version_id"}

    Returns:
        str: model response as audio file path
    """
    settings = Config(cac.get())

    if DEBUG:
        log_debug('CF-TTAPS-1) Clarifai - clarifai_text_to_audio_python_sdk' +
                  f'\n | model_config: {model_config}' +
                  f'\n | target_lang: {target_lang}' +
                  f'\n | other_options: {other_options}' +
                  f'\n | text_source: {text_source}' +
                  '\n')

    prev_pat_env_value = os.environ.get("CLARIFAI_PAT")
    os.environ["CLARIFAI_PAT"] = settings.CLARIFAI_PAT

    # input = "I love your product very much"
    if "raw_text" in text_source:
        input_text = text_source["raw_text"]
    elif "text_file_url" in text_source:
        input_text = text_source["text_file_url"]
    elif "text_file_location" in text_source:
        with open(text_source["text_file_location"], "rb") as f:
            input_text = f.read()  # file_bytes
    else:
        return "[ERROR-CF-TTAPS-010] Invalid input source: " + str(text_source)

    api_key = settings.ELEVENLABS_API_KEY

    voice_id = settings.ELEVENLABS_VOICE_ID_FEMALE
    if other_options.get("speaker_voice") == "male":
        voice_id = settings.ELEVENLABS_VOICE_ID_MALE

    inference_params = {
        # Check https://api.elevenlabs.io/v1/voices to list all the available
        # voices
        "voice-id": voice_id,
        "model_id": settings.ELEVENLABS_MODEL_ID,
        "stability": float(settings.ELEVENLABS_STABILITY),
        "similarity_boost": float(settings.ELEVENLABS_SIMILARITY_BOOST),
        "style": int(settings.ELEVENLABS_STYLE),
        "use_speaker_boost": (settings.ELEVENLABS_USE_SPEAKER_BOOST == '1'),
        "api_key": api_key,
    }
    inference_params.update(other_options)

    model_url = model_config["model_url"]

    # Model Predict
    model_prediction = Model(model_url).predict_by_bytes(
        input_text.encode(),
        input_type="text",
        inference_params=inference_params
    )

    output_base64 = model_prediction.outputs[0].data.audio.base64

    tmp_audio_filename = f'/tmp/{uuid4().hex}.wav'
    with open(tmp_audio_filename, 'wb') as f:
        f.write(output_base64)
    result = tmp_audio_filename

    if prev_pat_env_value:
        os.environ["CLARIFAI_PAT"] = prev_pat_env_value

    if DEBUG:
        log_debug('CF-TTAPS-2) Clarifai - clarifai_text_to_audio_python_sdk' +
                  " | result: " + result)

    return result
