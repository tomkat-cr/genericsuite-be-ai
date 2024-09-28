"""
Billing utilities module
"""
from genericsuite.util.app_logger import log_debug
from genericsuite.models.billing.billing_utilities import (
    BillingUtilities as BillingUtilitiesSuperClass
)
from genericsuite.util.app_context import AppContext

from genericsuite_ai.config.config import Config

DEFAULT_PLAN = "free"
DEBUG = False


class BillingUtilities(BillingUtilitiesSuperClass):
    """
    Billing utilities class for AI
    """
    def __init__(self, app_context: AppContext) -> None:
        super().__init__(app_context)
        self.settings = Config(app_context)

    def get_openai_chat_model(self) -> str:
        """
        Get OpenAI chat model based on user's current billing plan
        """
        oai_model = None
        if self.is_premium_plan():
            oai_model = self.settings.OPENAI_MODEL_PREMIUM
        elif self.is_free_plan():
            oai_model = self.app_context.get_user_data().get("openai_model")
        if not oai_model:
            oai_model = self.settings.OPENAI_MODEL
        _ = DEBUG and log_debug(
            'get_openai_chat_model' +
            f' | user_plan: {self.user_plan}' +
            f' | oai_model: {oai_model}')
        return oai_model

    def get_openai_api_key(self) -> str:
        """
        Get OpenAI API from the user profile
        """
        if self.is_free_plan():
            openai_api_key = \
                self.app_context.get_user_data().get("openai_api_key")
        else:
            openai_api_key = self.settings.OPENAI_API_KEY
        # _ = DEBUG and log_debug('get_openai_api_key' +
        #     f' | user_plan: {self.user_plan}' +
        #     f' | openai_api_key: {openai_api_key}')
        return openai_api_key
