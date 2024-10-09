"""
Implementation of the AI Chatbot API using Langchain - LCEL or Agents.
"""
from typing import Any, Union, Dict, Tuple
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
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import Output

from genericsuite.util.app_context import (
    AppContext,
    CommonAppContext,
)
from genericsuite.util.app_logger import (
    log_debug,
    log_warning,
)
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
    get_functions_dict,
    gpt_func_appcontext_assignment,
)
from genericsuite_ai.lib.ai_langchain_tools import (
    messages_to_langchain_fmt,
    messages_to_langchain_fmt_text,
    ExistingChatMessageHistory,
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
NON_AGENT_PROMPT = 'mediabros/gs_non_agent_lcel'
START_PHRASE_PROMPT_TOKEN = \
    "Assistant is a large language model trained by OpenAI"
BEGIN_PROMPT_TOKEN = "Begin!"

cac = CommonAppContext()
session_history_store: Dict[str, Any] = {}


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
    _ = DEBUG and log_debug(
        'TRANSLATE_USING |' +
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


def get_self_base_prompt(prompt_code: str) -> str:
    """
    Get the "self" (not from langsmith hub) base template text according
    to prompt_code.
    """
    _ = DEBUG and log_debug(
        f'>>> GET_SELF_BASE_PROMPT | prompt_code: {prompt_code}')
    if prompt_code == NON_AGENT_PROMPT:
        # Non-agent prompt
        base_prompt = """
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Begin!
"""
    else:
        base_prompt = """
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
    return base_prompt


def build_gs_prompt(base_prompt: str) -> str:
    """
    Prepare the base prompt for the agent
    """
    _ = DEBUG and log_debug(
        f'>>> BUILD_GS_PROMPT | base_prompt: {base_prompt}')
    settings = Config(cac.app_context)

    parent_model = cac.app_context.get_other_data("model_type")
    model_name = cac.app_context.get_other_data("model_name")
    model_manufacturer = cac.app_context.get_other_data("model_manufacturer")
    lang = get_response_lang(cac.app_context)
    bottom_line_prompt = get_constant("AI_PROMPT_TEMPLATES", "BOTTOM_LINE", "")

    translate_method = settings.LANGCHAIN_TRANSLATE_USING
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

    _ = DEBUG and log_debug('>>> Before get_current_date_time')
    suffix += \
        f"For date references, {get_current_date_time.invoke({})}.\n" + \
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
    _ = DEBUG and log_debug('>>> After get_current_date_time')

    new_prompt = base_prompt
    model_desc = f"Assistant is a large language model '{model_name}'" + \
        f" trained by '{model_manufacturer}'"

    if START_PHRASE_PROMPT_TOKEN in new_prompt:
        new_prompt = new_prompt.replace(
            START_PHRASE_PROMPT_TOKEN,
            model_desc
        )
    else:
        prefix += "\n" + model_desc

    new_prompt = prefix + new_prompt.replace(
        BEGIN_PROMPT_TOKEN,
        suffix + BEGIN_PROMPT_TOKEN)
    _ = DEBUG and log_debug(f'>>> BUILD_GS_PROMPT | new_prompt: {new_prompt}')
    return new_prompt


def get_agent_prompt(prompt_code: str) -> Any:
    """
    Set up the base template
    """
    _ = DEBUG and log_debug(
        f'>>> 1) GET_AGENT_PROMPT | prompt_code: {prompt_code}')
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
        input_variables = ['agent_scratchpad', 'chat_history', 'input',
                           'tool_names', 'tools']
        base_prompt = get_self_base_prompt(prompt_code)
    new_prompt = build_gs_prompt(base_prompt)
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=new_prompt,
    )
    # prompt.input_variables = input_variables
    if DEBUG:
        log_debug('>>> 2) GET_AGENT_PROMPT' +
                  f'\n | prompt_code: {prompt_code}' +
                  f'\n | input_variables: {input_variables}' +
                  f'\n | Prompt object:\n   {prompt}')
    return prompt


def get_agent_executor(
    agent_type: str,
    llm: Any,
    tools: list,
    messages: list,
) -> Tuple[AgentExecutor, Union[list, str]]:
    """
    Get the prompt to use and construct the agent executor
    """
    settings = Config(cac.app_context)
    agent_executor = None
    memory = None
    if DEBUG:
        log_debug(f'>>> GET_AGENT_EXECUTOR | agent_type: {agent_type}')

    # Agent types
    # https://python.langchain.com/docs/modules/agents/agent_types/

    if agent_type == "structured_chat_agent":
        # Structured Chat Agent
        # https://python.langchain.com/docs/modules/agents/agent_types/structured_chat
        prompt = get_agent_prompt("hwchase17/structured-chat-agent")
        agent = create_structured_chat_agent(llm, tools, prompt)

    # elif agent_type == "openai_tools_agent":
    #     # https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
    #     # REJECTION REASON: Doesn't accept long Tool descriptions
    #     prompt = get_agent_prompt("hwchase17/openai-tools-agent")
    #     agent = create_openai_tools_agent(llm, tools, prompt)

    # elif agent_type == "openai_functions_agent":
    #     # https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent
    #     # REJECTION REASON: Doesn't accept long Tool descriptions
    #     prompt = get_agent_prompt("hwchase17/openai-functions-agent")
    #     agent = create_openai_functions_agent(llm, tools, prompt)

    # elif agent_type == "self_ask_with_search_agent":
    #     # https://python.langchain.com/docs/modules/agents/agent_types/self_ask_with_search
    #     # REJECTION REASON: ValueError: This agent expects exactly one tool
    #     prompt = get_agent_prompt("hwchase17/self-ask-with-search")
    #     agent = create_self_ask_with_search_agent(llm, tools, prompt)

    elif agent_type in ('react_agent', 'react_chat_agent'):
        # ReAct agent
        # https://python.langchain.com/docs/modules/agents/agent_types/react
        # https://react-lm.github.io/

        if agent_type == "react_agent":
            prompt = get_agent_prompt("hwchase17/react")
        else:
            prompt = get_agent_prompt("hwchase17/react-chat")
            # Notice that chat_history is a string, since this prompt is
            # aimed at LLMs, not chat models.
            memory = messages_to_langchain_fmt_text(messages)
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


def filter_messages(messages, k_par: int = 10):
    """
    Filter messages to keep only the last k_par
    """
    settings = Config(cac.app_context)
    qty_msg_to_keep = settings.LANGCHAIN_MAX_CONV_MESSAGES or k_par
    _ = DEBUG and log_debug(f'>>> filter_messages | k: {qty_msg_to_keep}')
    if isinstance(qty_msg_to_keep, str):
        qty_msg_to_keep = int(qty_msg_to_keep)
    if qty_msg_to_keep == -1:
        return messages
    return messages[-qty_msg_to_keep:]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get the session history for the given session_id.
    """
    _ = DEBUG and log_debug(
        f'>>> 1) GET_SESSION_HISTORY | session_id: {session_id}')
    if session_id not in session_history_store:
        # session_history_store[session_id] = ChatMessageHistory()
        # https://python.langchain.com/v0.1/docs/modules/memory/
        messages = cac.app_context.get_other_data("conv_data")["messages"]
        _ = DEBUG and log_debug('>>> 2) GET_SESSION_HISTORY' +
                                f'\n| messages: {messages}')
        session_history_store[session_id] = ExistingChatMessageHistory(
            messages=messages_to_langchain_fmt(filter_messages(messages))
        )
    _ = DEBUG and log_debug(
        '>>> 3) GET_SESSION_HISTORY' +
        f'\n| Result: {session_history_store[session_id]}')
    return session_history_store[session_id]


def get_lcel_chain(
    llm: Any,
    tools: list,
) -> Runnable:
    """
    Get the prompt to use and construct the LCEL chain
    """
    settings = Config(cac.app_context)

    # If system_msg_permitted is False, the LLM model does not
    # support System message nor binding tools or functions.
    system_msg_permitted = \
        cac.app_context.get_other_data("system_msg_permitted")

    _ = DEBUG and log_debug('>>> GET_LCEL_CHAIN')
    _ = DEBUG and log_debug(f'Tools: {tools}')

    # 2024-05-21 | For LCEL:
    # https://python.langchain.com/v0.2/docs/how_to/tool_calling/

    # Check if the LLM supports binding tools or functions
    llm_with_tools = None
    if system_msg_permitted:
        # Models different than o1-mini/o1-preview accept Tools...
        if hasattr(llm, 'bind_tools'):
            llm_with_tools = llm.bind_tools(tools)
        elif hasattr(llm, 'bind_functions'):
            llm_with_tools = llm.bind_functions(functions=tools)
    if not llm_with_tools:
        if settings.AI_ALLOW_INFERENCE_WITH_NO_TOOLS == '0' \
           and system_msg_permitted:
            raise AttributeError("[AI_GLCEL_CH-E005] LLM does not" +
                                 " support binding tools or functions")
        model_type = cac.app_context.get_other_data("model_type")
        log_warning(
            'get_lcel_chain: [AI_GLCEL_CH-E010]' +
            ' LLM does not support binding tools or functions.' +
            f' Model: {model_type}')
        llm_with_tools = llm

    messages = []
    if system_msg_permitted:
        # Models different than o1-mini/o1-preview accept System message...
        new_prompt = build_gs_prompt(get_self_base_prompt(NON_AGENT_PROMPT))
        messages.append(("system", new_prompt,))
    messages.append(MessagesPlaceholder(variable_name="messages"))
    _ = DEBUG and log_debug('Start call to ChatPromptTemplate.from_messages()')
    prompt = ChatPromptTemplate.from_messages(messages)
    _ = DEBUG and log_debug('Start chain = prompt | llm_with_tools')
    # Build a Chatbot
    # https://python.langchain.com/v0.2/docs/tutorials/chatbot/#message-history
    chain = prompt | llm_with_tools
    _ = DEBUG and log_debug('Return chain')
    return chain


def run_lcel_chain(
    agent_executor: RunnableWithMessageHistory,
    question: str,
) -> Output:
    """
    Run the LCEL chain and returns the .invoke() results.
    """
    tools_dict = get_functions_dict(cac.app_context)
    _ = DEBUG and \
        log_debug('>>> 3.0) RUN_ASSISTANT | LCEL | tools_dict:' +
                  f' {tools_dict}')

    session_id = cac.app_context.get_other_data("cid")
    config = RunnableConfig({
        "configurable": {
            "session_id": session_id
        }
    })
    lcel_messages = [HumanMessage(content=question)]
    exec_result = agent_executor.invoke(
        lcel_messages,
        config=config,
    )
    # _ = DEBUG and \
    log_debug(
        '>>> 3.1) RUN_ASSISTANT | LCEL | exec_result:' +
        f' {exec_result}')

    if hasattr(exec_result, 'tool_calls') and exec_result.tool_calls:
        _ = DEBUG and log_debug(
            '>>> 3.3) RUN_ASSISTANT | Calling exec_result.tool_calls...')

        # https://python.langchain.com/v0.2/docs/how_to/tool_calling/#passing-tool-outputs-to-the-model
        lcel_messages.append(exec_result)
        for tool_call in exec_result.tool_calls:
            selected_tool = tools_dict[tool_call["name"].lower()]
            if isinstance(tool_call["args"], dict) \
                    and 'params' not in tool_call["args"]:
                tool_args = {
                    'params': tool_call["args"].copy()
                }
            else:
                tool_args = tool_call["args"].copy()
            _ = DEBUG and \
                log_debug(
                    '>>> 3.3) RUN_ASSISTANT | LCEL' +
                    f' | tool_name: {tool_call["name"].lower()}' +
                    f'\n| tool_call["args"]: {tool_call["args"]}' +
                    f'\n| tool_call["args"] TYPE: {type(tool_call["args"])}' +
                    f'\n| tool_args: {tool_args}' +
                    f'\n| tool_args TYPE: {type(tool_args)}' +
                    f'\n| selected_tool: {selected_tool}')
            tool_output = selected_tool.invoke(tool_args)
            lcel_messages.append(ToolMessage(tool_output,
                                             tool_call_id=tool_call["id"]))
            session_history_store[session_id].add_message(
                ToolMessage(tool_output,
                            tool_call_id=tool_call["id"]))
            _ = DEBUG and \
                log_debug(
                    '>>> 3.4) RUN_ASSISTANT | LCEL' +
                    f'\n| tool_output:\n{tool_output}' +
                    '\n| session_history_store[session_id]:' +
                    f'\n{session_history_store[session_id]}')

        # Call de assistant again with tools results...
        exec_result = agent_executor.invoke(
            lcel_messages,
            config=config,
        )
    return exec_result


def verify_tools(tools, conv_response):
    """
    Verify if the tools are valid, checking the Tool description lenght.
    """
    if conv_response["error"]:
        return conv_response
    bad_tools = [
        (tuple(
            [index, f"Lenght: {len(tools[index].description)}", tools[index]]
        ))
        for index in range(len(tools))
        if len(tools[index].description) > 1024
    ]
    _ = DEBUG and \
        log_debug('>>> VERIFY_TOOLS | LONG DESC tools[list]:' +
                  f' {bad_tools}')
    if len(bad_tools) > 0:
        conv_response["error"] = True
        conv_response["error_message"] = \
            "ERROR: Too long Tool description(s): " + \
            ', '.join([f'{v[2].name} ({v[1]})' for v in bad_tools])
    return conv_response


def run_assistant(
    conv_response: dict,
    llm: Any,
    tools: list,
    messages: list,
    question: str,
) -> dict:
    """
    Run the assistant
    """
    settings = Config(cac.app_context)
    agent_type = settings.LANGCHAIN_AGENT_TYPE
    _ = DEBUG and \
        log_debug('>>> 1) RUN_ASSISTANT | LANGCHAIN_AGENT_TYPE:' +
                  f' {agent_type}')

    conv_response = verify_tools(tools, conv_response)
    if conv_response["error"]:
        return conv_response

    pref_agent_type = cac.app_context.get_other_data("pref_agent_type")
    if agent_type == 'lcel' and pref_agent_type == 'lcel':
        # LCEL implementation (no agent, just a LLM call)

        # Call pipe example (LCEL):
        # https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/#create-the-agent
        agent_executor = RunnableWithMessageHistory(
            get_lcel_chain(llm=llm, tools=tools),
            # Build a Chatbot
            # https://python.langchain.com/v0.2/docs/tutorials/chatbot/#message-history
            get_session_history
        )
    else:
        # If the Model's preferred agent type is not LCEL,
        # use the Model's preference, otherwise use
        # the configured default agent type...
        if pref_agent_type != 'lcel':
            agent_type = pref_agent_type
        # Get the prompt to use and construct the agent
        agent_executor, memory = get_agent_executor(
            agent_type=agent_type,
            llm=llm,
            tools=tools,
            messages=messages,
        )

    # Run Agent
    _ = DEBUG and \
        log_debug('>>> 2) RUN_ASSISTANT | Start chain/agent execution...')
    try:
        if agent_type == 'lcel' and pref_agent_type == 'lcel':
            # LCEL implementation (no agent, just a LLM call)
            exec_result = run_lcel_chain(agent_executor, question)
            if hasattr(exec_result, 'content'):
                conv_response["response"] = exec_result.content
            elif isinstance(exec_result, str):
                conv_response["response"] = exec_result
            else:
                conv_response["response"] = \
                    "ERROR - Empty response from Assistant [AIRCLC-E030]"
        else:
            # Chat Agent implementation
            exec_result = agent_executor.invoke({
                "input": question,
                "chat_history": memory,
            })
            conv_response["response"] = exec_result.get(
                "output",
                "ERROR - Empty response from Agent [AIRCLC-E035]"
            )
        _ = DEBUG and \
            log_debug(
                '>>> 4) RUN_ASSISTANT | LANGCHAIN_AGENT_TYPE:' +
                f' {agent_type} |' +
                f' exec_result: {exec_result}')
    except Exception as error:
        conv_response["error"] = True
        conv_response["error_message"] = \
            get_standard_base_exception_msg(error, "AIRCLC-E020")
    return conv_response


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

    # Initialize Tools

    # Defining custom Tools
    # https://python.langchain.com/docs/modules/agents/tools/custom_tools
    tools = get_function_list(cac.app_context)

    # Get the model according to user's billing plan
    billing = BillingUtilities(app_context)
    if billing.is_free_plan():
        if not billing.get_openai_api_key():
            # No API key found
            conv_response["error"] = True
            conv_response["error_message"] = \
                "You must specify your OPENAI_API_KEY in your" + \
                " profile or upgrade to a paid plan [AIRCOA-E010]"
            # Cancels the translation method
            translate_method = NON_AI_TRANSLATOR
        else:
            model_type = 'chat_openai'
    else:
        model_type = settings.LANGCHAIN_DEFAULT_MODEL

    # Choose the LLM that will drive the agent
    if not conv_response["error"]:
        llm = get_model_obj(cac.app_context, model_type)
        if not llm:
            # No model found
            conv_response["error"] = True
            conv_response["error_message"] = get_model_load_error(
                app_context=cac.app_context,
                error_code="AIRCLC-E010"
            )
            # Cancels the translation method
            translate_method = NON_AI_TRANSLATOR

    if not conv_response["error"]:
        conv_response = run_assistant(
            conv_response=conv_response,
            llm=llm,
            tools=tools,
            messages=messages,
            question=question,
        )

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
