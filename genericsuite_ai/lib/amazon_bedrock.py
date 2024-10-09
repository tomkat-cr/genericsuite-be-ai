import base64
import boto3
import json
import os

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


def aws_bedrock_img_gen(question: str) -> dict:
    """
    Amazon Bedrock image generation
    """
    settings = Config(cac.get())
    ig_response = get_default_resultset()

    if not question:
        return error_resultset(
            error_message='No question supplied',
            message_code='AWSBED-IG-E010',
        )

    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)

    # Set the model ID, e.g., Titan Image Generator G1.
    model_id = settings.AWS_BEDROCK_IMAGE_GEN_MODEL_ID

    # Format the request payload using the model's native structure.
    native_request = {
        "text_prompts": [
            {
                "text": question,
                "weight": 1,
            }
        ],
        "cfg_scale": 10,
        "steps": 50,
        "seed": 0,
        "width": 1024,
        "height": 1024,
        "samples": 1
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract the image data.
    base64_image_data = model_response["artifacts"][0]["base64"]

    # Save the generated image to a local folder.
    i, output_dir = 1, settings.TEMP_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    while os.path.exists(os.path.join(output_dir, f"image_{i}.png")):
        i += 1

    image_data = base64.b64decode(base64_image_data)
    image_filename = f"image_{i}.png"
    image_path = os.path.join(output_dir, image_filename)
    with open(image_path, "wb") as file:
        file.write(image_data)

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
            message_code="AWSBED-IG-E030",
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
