"""
GenericSuite AI main module (create_app) for FastAPI
"""
from typing import Any

from genericsuite.fastapilib.util.create_app import (     # noqa: F401
    create_app as super_create_app,
    create_handler,     # noqa: F401
    set_init_custom_data,
)

from genericsuite_ai.config.config import Config
from genericsuite_ai.fastapilib.endpoints import ai_chatbot_endpoint
from genericsuite_ai.lib.ai_conversations_conversion \
    import ai_conversation_masking


def create_app(app_name: str, settings: Config = None) -> Any:
    """ Create the FastAPI App """

    if settings is None:
        settings = Config()

    fa_app = super_create_app(
        app_name=app_name,
        settings=settings)

    # Register GenericDbHelper specific functions
    fa_app.custom_data = set_init_custom_data({
        "ai_conversation_masking": ai_conversation_masking
    })

    # Register AI endpoints
    fa_app.include_router(ai_chatbot_endpoint.router,
                          prefix=f'/{settings.API_VERSION}/ai')

    return fa_app
