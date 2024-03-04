"""
GenericSuite AI main module (create_app) for Chalice
"""
from typing import Any

from genericsuite.chalicelib.util.create_app import (
    create_app as super_create_app)

from genericsuite_ai.config.config import Config
from genericsuite_ai.chalicelib.endpoints import ai_chatbot_endpoint

DEBUG = False


def create_app(app_name: str, settings = None) -> Any:
    """ Create the Chalice App """

    if settings is None:
        settings = Config()

    chalice_app = super_create_app(
        app_name=app_name,
        settings=settings)

    # Register AI endpoints
    chalice_app.register_blueprint(ai_chatbot_endpoint.bp, url_prefix='/ai')

    return chalice_app
