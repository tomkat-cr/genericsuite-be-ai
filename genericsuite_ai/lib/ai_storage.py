from genericsuite.util.app_context import AppContext
from genericsuite.util.cloud_provider_abstractor import get_cloud_provider
from genericsuite.util.app_logger import log_error

from genericsuite_ai.config.config import Config


def get_chatbot_attachments_bucket_name(app_context: AppContext = None) -> str:
    """
    Returns the chatbot attachments bucket name.
    """
    settings = Config(app_context)
    cloud_provider = get_cloud_provider()
    bucket_name = None
    if cloud_provider == "AWS":
        bucket_name = settings.AWS_S3_CHATBOT_ATTACHMENTS_BUCKET
    elif cloud_provider == "AZURE":
        bucket_name = settings.get_env('AZURE_CHATBOT_ATTACHMENTS_BUCKET')
    elif cloud_provider == "GCP":
        bucket_name = settings.get_env('GCP_CHATBOT_ATTACHMENTS_BUCKET')
    if not bucket_name:
        log_error("Chatbot attachments bucket name is not configured" +
                  f" for {cloud_provider} [AI-STO-GCABN-E010]")
    return bucket_name
