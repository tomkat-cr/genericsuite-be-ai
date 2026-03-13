"""
GenericSuite AI main module (create_app) for Chalice
"""
from typing import Any

from genericsuite.flasklib.util.create_app import (
    create_app as super_create_app,
    set_init_custom_data,
)

from genericsuite_ai.config.config import Config
from genericsuite_ai.flasklib.endpoints import ai_chatbot_endpoint
from genericsuite_ai.lib.ai_conversations_conversion \
    import ai_conversation_masking

DEBUG = False


def create_app(app_name: str, settings=None) -> Any:
    """ Create the Flask App """

    if settings is None:
        settings = Config()

    flask_app = super_create_app(
        app_name=app_name,
        settings=settings)

    # Register GenericDbHelper specific functions
    flask_app.custom_data = set_init_custom_data({
        "ai_conversation_masking": ai_conversation_masking
    })

    # Register AI endpoints
    flask_app.register_blueprint(
        ai_chatbot_endpoint.bp)

    return flask_app
