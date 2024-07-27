"""
GenericSuite AI main module (create_app) for FastAPI
"""
from typing import Any
from mangum import Mangum

from genericsuite.fastapilib.util.create_app import (
    create_app as super_create_app
)
# from genericsuite.util.app_logger import log_debug

from genericsuite_ai.config.config import Config
from genericsuite_ai.fastapilib.endpoints import ai_chatbot_endpoint

DEBUG = False


def create_app(app_name: str, settings: Config = None) -> Any:
    """ Create the FastAPI App """

    if settings is None:
        settings = Config()

    fa_app = super_create_app(
        app_name=app_name,
        settings=settings)

    # Register AI endpoints
    fa_app.include_router(ai_chatbot_endpoint.router, prefix='/ai')

    return fa_app


def create_handler(app_object):
    """
    Returns the FastAPI App as a valid AWS Lambda Function handler
    """
    return Mangum(app_object, lifespan="off")
