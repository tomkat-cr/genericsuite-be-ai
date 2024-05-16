"""
Conversations conversion (unmasked / masked)
"""
from genericsuite.util.framework_abs_layer import Response, BlueprintOne
# from genericsuite.util.blueprint_one import BlueprintOne
from genericsuite.util.utilities import (
    return_resultset_jsonified_or_exception,
)
from genericsuite.util.jwt import (
    request_authentication,
    AuthorizedRequest
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import AppContext

from genericsuite_ai.lib.ai_conversations_conversion import (
    mask_all_conversations
)


bp = BlueprintOne(__name__)


@bp.route(
    '/',
    authorizor=request_authentication(),
    methods=['GET']
)
def ai_conversations_conversion_endpoint(
    request: AuthorizedRequest,
    other_params: dict = None
) -> Response:
    """ Get file from ecrypted URL """
    log_debug('>> AI_CONVERSATIONS_CONVERSION_ENDPOINT')
    app_context = AppContext()
    # app_context.set_context(request)
    app_context.set_context_from_blueprint(blueprint=blueprint, request=request)
    if app_context.has_error():
        log_debug('>> AI_CONVERSATIONS_CONVERSION_ENDPOINT | HAS_ERROR' +
            f' (skip): {app_context.get_error_resultset()}')
        return return_resultset_jsonified_or_exception(
            app_context.get_error_resultset()
        )
    return return_resultset_jsonified_or_exception(
        mask_all_conversations(app_context)
    )

"""

Instructions:
------------

1. Copy the "ai_chatbot_conversations.json" file into "ai_chatbot_conversations_complete.json" in the "config_dbdef/frontend/" folder.

2. Edit the "config_dbdef/frontend/ai_chatbot_conversations_complete.json" file and replace the "messages" definition, changing the "listing" attribute to "true":

    {
        "name": "messages",
        "required": true,
        "label": "Messages",
        "type": "array",
        "listing": true
    }

3. For the Chalice framework, add the following code to the "app.py" file:

# At the begining:
from genericsuite_ai.chalicelib.endpoints import ai_conversations_conversion

# At the end:
app.register_blueprint(ai_conversations_conversion.bp, url_prefix='/ai_conversations_conversion')

4. Login into the "api-[stage].fynapp.com" App and copy the "Authorization: Bearer" token.

5. Make the following call to the "ai_conversations_conversion" endpoint:


curl --location 'https://api-[stage].fynapp.com/ai_conversations_conversion?bucket_name=exampleapp-s3-bucket&save=1&hostname=api-[stage].fynapp.com' \
--header 'Authorization: Bearer [token]'

"""
