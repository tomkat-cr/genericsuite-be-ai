"""
GCP (Google Cloud Platform) utilities
"""
import os
import json

from google.oauth2 import service_account

from genericsuite.util.app_logger import log_debug


DEBUG = True


def get_service_account_credentials(creds_file_path: str):
    """
    Get Services account credentials for accessing Google Cloud APIs.
    """
    _ = DEBUG and log_debug(
        ">> GCP / GET_SERVICE_ACCOUNT_CREDENTIALS"
        f"\n | creds_file_path: {creds_file_path}")

    credentials = service_account.Credentials \
        .from_service_account_file(creds_file_path)
    return credentials


def get_gcp_vertexai_credentials(creds_file_path: str):
    if not creds_file_path:
        return None

    if not os.path.exists(creds_file_path):
        raise Exception(
            f"The file {creds_file_path}"
            f" does not exist.")

    with open(creds_file_path,
              encoding="utf-8") as credentials_file:
        creds = json.load(credentials_file)

    if creds.get("type") == "service_account":
        # Services account credentials file
        credentials = get_service_account_credentials(creds_file_path)
    else:
        raise Exception(
            f"The file {creds_file_path}"
            f" does not contain valid GCP credentials.")
    return credentials
