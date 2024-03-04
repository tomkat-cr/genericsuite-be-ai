"""
Implementation of the AI Chatbot API using Langchain.
"""
from typing import Any, Union
import os

from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
    create_react_agent,
    # create_openai_functions_agent,
    # create_openai_tools_agent,
    # create_self_ask_with_search_agent,
)
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from genericsuite.util.app_context import (
    AppContext,
    CommonAppContext,
)
from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import (
    get_standard_base_exception_msg,
    get_default_resultset,
)
from genericsuite.constants.const_tables import get_constant

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.ai_utilities import (
    report_error,
    get_response_lang,
    get_response_lang_code,
    gpt_func_error,
    get_assistant_you_are,
)
from genericsuite_ai.lib.ai_chatbot_commons import (
    start_run_conversation,
    finish_run_conversation,
)
from genericsuite_ai.lib.ai_gpt_functions import (
    get_function_list,
    gpt_func_appcontext_assignment,
)
from genericsuite_ai.lib.ai_langchain_tools import (
    messages_to_langchain_fmt,
)
from genericsuite_ai.lib.ai_langchain_models import (
    get_model_obj,
    get_model_load_error,
    # get_chat_engine_desc,
)
from genericsuite_ai.lib.ai_gpt_fn_conversations import (
    get_current_date_time
)
from genericsuite_ai.lib.translator import translate
from genericsuite_ai.models.billing.billing_utilities import BillingUtilities


DEBUG = False

NON_AI_TRANSLATOR = 'google_translate'
cac = CommonAppContext()


def translate_using(input_text: str, llm: Any) -> dict:
    """
    Translate the input text using the configured translator.
    to the user's preferred language.

    Args:
        input_text (str): texto to be translated.    

    Returns:
        dict: a resultset with an "text" attribute containing
        the translated input text, or "error" and "error_message"
        attributes if there's something wrong.
    """
    resultset = get_default_resultset()
    settings = Config(cac.app_context)
    lang = get_response_lang(cac.app_context)
    lang_code = get_response_lang_code(cac.app_context)
    trans_method = settings.LANGCHAIN_TRANSLATE_USING
    if trans_method == NON_AI_TRANSLATOR:
        # Translate using google translator
        trans_resp = translate(input_text, lang_code)
        if trans_resp["error"]:
            resultset["error"] = True
            resultset["error_message"] = trans_resp["error_message"]
        else:
            resultset['text'] = trans_resp["text"]
    elif trans_method == 'same_model':
        # Translate using same LLM model
        try:
            prompt = PromptTemplate.from_template(
                "Translate the text {text} to {lang}")
            # prompt.format(text=input_text, lang=lang)
            output_parser = StrOutputParser()
            chain = (
                {
                    "text": RunnablePassthrough(),
                    "lang": lambda x: lang
                }
                | prompt
                | llm
                | output_parser
            )
            resultset['text'] = chain.invoke(input_text)
        except Exception as err:
            resultset["error"] = True
            resultset["error_message"] = \
                f"ERROR: AI translation to {lang} cannot be completed" + \
                f" because of the following error: {err}" + \
                " [TA-E010]"
    else:
        resultset["error"] = True
        resultset["error_message"] = \
            f"ERROR: Unknown translation method: {trans_method}" + \
            " [TA-E020]"
    _ = DEBUG and log_debug('TRANSLATE_USING |' +
        f' trans_method: {trans_method}' + 
        f'\n | target "lang": {lang}' +
        f'\n | target "lang_code": {lang_code}' +
        f'\n | input_text: {input_text}' +
        f"\n | resultset: {resultset}"
    )
    return resultset


def needs_answer_translation() -> bool:
    """
    Check if the response needs to be translated
    
    Returns:
        bool: True if the user's prefferred language is not English
    """
    lang = get_response_lang(cac.app_context)
    return lang.lower() != "english"


def get_prompt(prompt_code: str) -> Any:
    """
    Set up the base template
    """
    settings = Config(cac.app_context)
    use_langsmith_hub = settings.LANGCHAIN_USE_LANGSMITH_HUB == "1"
    if use_langsmith_hub:
        try:
            prompt: PromptTemplate = hub.pull(prompt_code)
            base_prompt = prompt.template
            input_variables = prompt.input_variables
        except ConnectionError:
            use_langsmith_hub = False
        except Exception as err:
            raise Exception(f"[AI_GP-E010] Error: {err}") from err
    if not use_langsmith_hub:
        input_variables = ['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools']
        base_prompt = \
"""
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
    # chat_engine_desc = get_chat_engine_desc(cac.app_context)
    # model_name = chat_engine_desc["model"]
    parent_model = cac.app_context.get_other_data("model_type")
    model_name = cac.app_context.get_other_data("model_name")
    model_manufacturer = cac.app_context.get_other_data("model_manufacturer")
    lang = get_response_lang(cac.app_context)
    bottom_line_prompt = get_constant("AI_PROMPT_TEMPLATES", "BOTTOM_LINE", "")
    translate_method = settings.LANGCHAIN_TRANSLATE_USING
    # if chat_engine_desc["parent_model"] in ['gemini'] or \
    #    chat_engine_desc["model"] in ['gemini']:
    if parent_model in ['gemini'] or model_name in ['gemini']:
        # Enforce translation of final answer before send it to the user
        # outside the Agent run for Models that translates the format
        # elements phrases like "Final Answer:" or "Thought:"
        translate_method = NON_AI_TRANSLATOR
    translate_answer_flag = needs_answer_translation() \
        and translate_method == 'initial_prompt'
    prefix = ""
    suffix = ""
    prefix += \
        f"Assistant is named {get_assistant_you_are(cac.app_context)}\n"
    suffix += \
        "\n" + \
        "ADDITIONAL GUIDELINES:\n" + \
        "----------------------\n" + \
        "\n"
    if translate_answer_flag:
        prefix += \
            "\n" + \
            "Think and act entirelly in english, but at the end response" + \
            f" to the Human in {lang}, no matter his/her input language.\n" + \
            "\n"
    suffix += \
        f"For date references, {get_current_date_time({})}.\n" + \
        "\n" + \
        f"If the called Tool result has '{gpt_func_error('')}'," + \
        " stop processing and report the error, otherwise the" + \
        " called Tool response is Ok.\n" + \
        "\n" + \
        "Always use a json structure in the 'Action Input'" + \
        " when calling the Tool," + \
        " enclosing all attributes with curly braces.\n" + \
        bottom_line_prompt + \
        "\n"
    new_prompt = base_prompt
    model_desc = f"Assistant is a large language model '{model_name}'" + \
        f" trained by '{model_manufacturer}'"
    if "Assistant is a large language model trained by OpenAI" in new_prompt:
        new_prompt = new_prompt.replace(
            "Assistant is a large language model trained by OpenAI",
            model_desc
        )
    else:
        prefix += "\n" + model_desc
    new_prompt = prefix + new_prompt.replace("Begin!", suffix + "Begin!")

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=new_prompt,
    )
    # prompt.input_variables = input_variables
    if DEBUG:
        log_debug('>>> GET_PROMPT' +
                  f'\n | prompt_code: {prompt_code}' +
                  f'\n | input_variables: {input_variables}' +
                  f'\n | Prompt object:\n   {prompt}')
    return prompt


def get_agent_executor(
    agent_type: str,
    llm: Any,
    tools: list,
    messages: list,
) -> (AgentExecutor, Union[list, str]):
    """ Get the prompt to use and construct the agent """
    settings = Config(cac.app_context)
    agent_executor = None
    memory = None
    if DEBUG:
        log_debug(f'>>> GET_AGENT | agent_type: {agent_type}')

    if agent_type == "structured_chat_agent":
        # Structured Chat Agent
        # https://python.langchain.com/docs/modules/agents/agent_types/structured_chat
        prompt = get_prompt("hwchase17/structured-chat-agent")
        agent = create_structured_chat_agent(llm, tools, prompt)

    # elif agent_type == "openai_tools_agent":
    #     # https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
    #     # REJECTION REASON: Doesn't accept long Tool descriptions
    #     prompt = get_prompt("hwchase17/openai-tools-agent")
    #     agent = create_openai_tools_agent(llm, tools, prompt)

    # elif agent_type == "openai_functions_agent":
    #     # https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent
    #     # REJECTION REASON: Doesn't accept long Tool descriptions
    #     prompt = get_prompt("hwchase17/openai-functions-agent")
    #     agent = create_openai_functions_agent(llm, tools, prompt)

    # elif agent_type == "self_ask_with_search_agent":
    #     # https://python.langchain.com/docs/modules/agents/agent_types/self_ask_with_search
    #     # REJECTION REASON: ValueError: This agent expects exactly one tool
    #     prompt = get_prompt("hwchase17/self-ask-with-search")
    #     agent = create_self_ask_with_search_agent(llm, tools, prompt)

    elif agent_type in ('react_agent', 'react_chat_agent'):
        # ReAct agent
        # https://python.langchain.com/docs/modules/agents/agent_types/react
        # https://react-lm.github.io/

        if agent_type == "react_agent":
            prompt = get_prompt("hwchase17/react")
        else:
            prompt = get_prompt("hwchase17/react-chat")
            # Notice that chat_history is a string, since this prompt is aimed at LLMs,
            # not chat models.
            memory = messages_to_langchain_fmt(messages, "text")
        agent = create_react_agent(llm, tools, prompt)
    else:
        raise Exception(f"[AI_GAE-E010] Invalid agent_type {agent_type}")
    if not memory:
        # Prepare memory
        memory = messages_to_langchain_fmt(messages)
    if not agent_executor:
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=DEBUG,
            handle_parsing_errors=(
                settings.LANGCHAIN_HANDLE_PARSING_ERR == '1'
            ),
            # early_stopping_method=settings.LANGCHAIN_EARLY_STOPPING_METHOD,
            # max_iterations=int(settings.LANGCHAIN_MAX_ITERATIONS),
        )
    return agent_executor, memory


def run_conversation(app_context: AppContext) -> dict:
    """
    LangChain conversation run.

    Args:
        app_context (AppContext): the application context.

    Returns:
        dict: a standard response with this structure:
            response (str): = the message answered by the model
            error (bool): = True if there was a error
            error_message (str): = eventual error message
            cid (str): coversation ID
    """
    cac.set(app_context)
    settings = Config(cac.app_context)
    start_response = start_run_conversation(
        app_context=cac.app_context,
        initial_prompt=False
    )
    if start_response["error"]:
        return report_error(start_response)
    conv_response = start_response["conv_response"]
    # query_params = start_response["query_params"]
    messages = start_response["messages"]

    translate_method = settings.LANGCHAIN_TRANSLATE_USING

    # Prepare application context for all Tools (GPT Functions)
    gpt_func_appcontext_assignment(cac.app_context)

    # Langsmith
    if settings.LANGCHAIN_API_KEY != '':
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2

    # Get the user's question
    question = conv_response["question"]["content"]

    # Removes the last message, because it's the user's question,
    # which is already in conv_response["question"] and
    # we want the last message to be "The next question is: {topic}"
    messages = messages[:-1]

    # ............................
    # ............................

    # Agent types
    # https://python.langchain.com/docs/modules/agents/agent_types/

    # Call pipe example:
    # https://python.langchain.com/docs/modules/agents/how_to/custom_agent

    # ............................
    # ............................

    # Initialize Tools

    # Defining custom Tools
    # https://python.langchain.com/docs/modules/agents/tools/custom_tools
    tools = get_function_list(cac.app_context)

    # Get the model according to user's billing plan
    billing = BillingUtilities(app_context)
    if billing.is_free_plan():
        if not billing.get_openai_api_key():
            conv_response["error"] = True
            conv_response["error_message"] = "You must specify your OPENAI_API_KEY in your" + \
                " profile or upgrade to a paid plan [AIRCOA-E010]"
            translate_method = NON_AI_TRANSLATOR
        else:
            model_type = 'chat_openai'
    else:
        model_type = settings.LANGCHAIN_DEFAULT_MODEL

    # Choose the LLM that will drive the agent
    if not conv_response["error"]:
        llm = get_model_obj(cac.app_context, model_type)
        if not llm:
            conv_response["error"] = True
            conv_response["error_message"] = get_model_load_error(
                app_context=cac.app_context,
                error_code="AIRCLC-E010"
            )
            translate_method = NON_AI_TRANSLATOR

    if not conv_response["error"]:
        # Get the prompt to use and construct the agent
        agent_executor, memory = get_agent_executor(
            agent_type=settings.LANGCHAIN_AGENT_TYPE,
            llm=llm,
            tools=tools,
            messages=messages,
        )

        # Run Agent
        try:
            exec_result = agent_executor.invoke({
                "input": question,
                "chat_history": memory,
            })
            conv_response["response"] = exec_result.get(
                "output",
                "ERROR - Empty response from Agent [AIRCLC-E030]"
            )
            if DEBUG:
                log_debug('>>> CHAT AGENT | LANGCHAIN_AGENT_TYPE:' +
                          f' {settings.LANGCHAIN_AGENT_TYPE} |' +
                          f' exec_result: {exec_result}')

        except Exception as error:
            conv_response["error"] = True
            conv_response["error_message"] = \
                get_standard_base_exception_msg(error, "AIRCLC-E020")

    if not conv_response["error"] and \
       needs_answer_translation() and \
       translate_method in [NON_AI_TRANSLATOR, 'same_model']:
        # Translate response to the user's preferred language
        trans_response = translate_using(conv_response["response"],
                                          llm=llm)
        if trans_response["error"]:
            conv_response["response"] += \
                "\n\nNOTE: translation to " + \
                f"{get_response_lang(cac.app_context)}" + \
                f" failed: {trans_response['error_message']}"
        else:
            conv_response["response"] = trans_response["text"]

    return finish_run_conversation(
        app_context=cac.app_context,
        conv_response=conv_response,
        messages=messages,
    )
