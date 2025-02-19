"""
IBM watsonx + models class for LangChain.

Reference:
https://developer.ibm.com/tutorials/integrate-your-watson-assistant-chatbot-with-watsonxai-for-generative-ai/
https://developer.ibm.com/
"""
import os
import requests

from typing import Any, List, Optional
# from typing import Mapping, Dict, Iterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
# from langchain_core.outputs import GenerationChunk

# from langchain_core.language_models.llms import LLM
from genericsuite_ai.lib.ai_langchain_models_abstract import CustomLLM as LLM
from genericsuite.util.app_logger import log_debug


DEBUG = False


class IbmWatsonx(LLM):
    """
    IBM watsonx + models class for LangChain.
    """

    # Mandatory parameters
    api_key: str
    project_id: str
    model_url: str
    identity_token_url: str

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off
                at the first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising
                NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are
                usually passed to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT
            include the prompt.
        """
        llm_response = self._inference_call(prompt, stop, run_manager,
                                            **kwargs)
        result = " ".join([item["generated_text"] for item
                           in llm_response.get("results", [])])
        _ = DEBUG and log_debug(f">> IbmWatsonx | _call result: {result}")
        return result

    @property
    def _llm_type(self) -> str:
        """
        Get the type of language model used by this chat model. Used for
        logging purposes only.
        """
        return "IBM watsonx"

    def _inference_call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the watsonx LLM on the given input.

        Reference:

        Tutorial Video: https://www.youtube.com/watch?v=yq8jNzf9Hxo
                        (minute 42:18)

        Integrate watsonx Assistant with watsonx.ai foundation models
        https://developer.ibm.com/tutorials/integrate-your-watson-assistant-chatbot-with-watsonxai-for-generative-ai/

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off
                at the first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising
                NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are
                usually passed to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT
            include the prompt.
        """

        model_url = self.model_url
        if not model_url:
            model_url = \
                "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?" \
                "version=2023-05-29"

        model_id = self._get_model_name()
        if model_id == "CustomChatModel":
            model_id = kwargs.get(
                "model_name",
                os.environ.get("IBM_WATSONX_MODEL_NAME", model_id))
        project_id = self.project_id
        if not project_id:
            project_id = kwargs.get(
                "project_id",
                os.environ.get("IBM_WATSONX_PROJECT_ID"))
        api_key = self.api_key
        if not api_key:
            api_key = kwargs.get(
                "api_key",
                os.environ.get("IBM_WATSONX_API_KEY"))

        access_token = self._get_access_token(api_key)
        if access_token:
            access_token = access_token.get("access_token")

        parameters = {
            "temperature": 0.7,
            "max_new_tokens": 200,
            "min_new_tokens": 50,
            # "decoding_method": "greedy",
            "decoding_method": "sample",
            # "repetition_penalty": 1
            "repetition_penalty": 2
        }

        if DEBUG:
            log_debug(f"IBM WatsonX model_id: {model_id}")
            log_debug(f"IBM WatsonX project_id: {project_id}")
            log_debug(f"IBM WatsonX api_key: {api_key}")
            log_debug(f"IBM WatsonX access_token: {access_token}")
            log_debug(f"IBM WatsonX model_url: {model_url}")
            log_debug(f"IBM WatsonX kwargs: {kwargs}")
            log_debug(f"IBM WatsonX prompt: {prompt}")

        # "input" attribute example for one-shot or few-shot learning.
        # If no Input/Output combinations in the prompt, the zero-shot
        # learning will be used...
        """
            { "input": \"""Summarize the transcript

        Input: {sample impout here 1}
        Output: {same output here 1}

        Input: {sample impout here 1}
        Output: {same output here 2}

        Input: I need to by a house
        Output:\""" }
        """

        body = {
            "input": f"""Input: {prompt}
Output:""",
            "model_id": model_id,
            "project_id": project_id,
            "parameters": parameters,
            "moderations": {
                "hap": {
                    "input": {
                        "enabled": True,
                        "threshold": 0.5,
                        "mask": {
                            "remove_entity_value": True
                        }
                    },
                    "output": {
                        "enabled": True,
                        "threshold": 0.5,
                        "mask": {
                            "remove_entity_value": True
                        }
                    }
                },
                "pii": {
                    "input": {
                        "enabled": True,
                        "threshold": 0.5,
                        "mask": {
                            "remove_entity_value": True
                        }
                    },
                    "output": {
                        "enabled": True,
                        "threshold": 0.5,
                        "mask": {
                            "remove_entity_value": True
                        }
                    }
                }
            }
        }

        headers = {
            "Accept": "application/json",
            "content-type": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.post(
            model_url,
            headers=headers,
            json=body
        )

        if response.status_code != 200:
            raise Exception("GS-IBM-E010: Non-200 response: " +
                            str(response.text))

        data = response.json()
        _ = DEBUG and log_debug(f">> IBM WatsonX response: {data}")

        """
        Response example:
        {
            'model_id': 'google/flan-t5-xxl',
            'created_at': '2024-11-18T17:14:56.466Z',
            'results': [
                {
                    'generated_text':
                        "System: Assistant is a large language model
                        'google/flan-t5-xxl' trained by 'IBM watsonx'.
                        Assistant is designed to be able to assist with a wide
                        range of tasks, from answering simple questions to
                        providing in-depth explanations and discussions on a
                        wide range of topics. As a language model, Assistant
                        is able to generate human-like text based on the input
                        it receives, allowing it to engage in natural-sounding
                        conversations and provide accurate and informative
                        responses to a wide range of questions. Additionally,
                        Assistant is able to generate its own text based on the
                        input it receives, allowing it to engage in discussions
                        and provide explanations and descriptions on a wide
                        range of topics. Overall, Assistant is a powerful tool
                        that can help with a wide range of tasks and provide
                        valuable insights and information on a wide range of
                        topics. Whether you",
                    'generated_token_count': 200,
                    'input_token_count': 552,
                    'stop_reason': 'max_tokens',
                    'seed': 0
                }
            ],
            'system': {
                'warnings': [
                    {
                        'message': '
                            This model is a Non-IBM Product governed by
                            a third-party license that may impose use
                            restrictions and other obligations. By using this
                            model you agree to its terms as identified in the
                            following URL.',
                        'id': 'disclaimer_warning',
                        'more_info':
                            'https://dataplatform.cloud.ibm.com/docs/content/'
                            'wsj/analyze-data/fm-models.html?context=wx'
                    },
                    {
                        'message': 'Threshold is not supported for PII',
                        'id': 'threshold_parameter_warning'
                    }
                ]
            }
        }
        """

        return data

    def _get_access_token(self, api_key: str):
        """
        Generate an IAM token by using an API key
        Referece:
        https://cloud.ibm.com/docs/account?topic=account-iamtoken_from_apikey#iamtoken_from_apikey
        """
        identity_token_url = self.identity_token_url
        if not identity_token_url:
            identity_token_url = "https://iam.cloud.ibm.com/identity/token"

        headers = {
            "Content-Type": "application/json",
        }

        body = "grant_type=urn:ibm:params:oauth:grant-type:apikey" \
               f"&apikey={api_key}"

        response = requests.post(
            identity_token_url,
            headers=headers,
            json=body
        )

        if response.status_code != 200:
            raise Exception("GS-IBM-E010: Non-200 response: " +
                            str(response.text))

        data = response.json()
        _ = DEBUG and log_debug(f">> IBM Auth response: {data}")

        """
        Response example:
        {
            "access_token": "...",
            "refresh_token": "not_supported",
            "ims_user_id": 12979176,
            "token_type": "Bearer",
            "expires_in": 3600,
            "expiration": 1731951986,
            "scope": "ibm openid"
        }
        """
        return data
