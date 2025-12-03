"""
GenericSuite AI main module (create_app) for Chalice
"""
from typing import Any

from genericsuite.chalicelib.util.create_app import (
    create_app as super_create_app,
    set_init_custom_data,
)

from genericsuite_ai.config.config import Config
from genericsuite_ai.chalicelib.endpoints import ai_chatbot_endpoint
from genericsuite_ai.lib.ai_conversations_conversion \
    import ai_conversation_masking

DEBUG = False


def create_app(app_name: str, settings=None) -> Any:
    """ Create the Chalice App """

    if settings is None:
        settings = Config()

    chalice_app = super_create_app(
        app_name=app_name,
        settings=settings)

    # Register GenericDbHelper specific functions
    chalice_app.custom_data = set_init_custom_data({
        "ai_conversation_masking": ai_conversation_masking
    })

    # Register AI endpoints
    chalice_app.register_blueprint(
        ai_chatbot_endpoint.bp, url_prefix=f'/{settings.API_VERSION}/ai')

    return chalice_app
