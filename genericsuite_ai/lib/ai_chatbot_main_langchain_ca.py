"""
Implementation of the AI Chatbot API using Langchain - Custom Agent (_ca).
"""
import re
import json
from typing import Union, Callable

from openai import PermissionDeniedError

# from langchain import hub
from langchain.agents import (
    AgentExecutor,
    # create_structured_chat_agent,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish, Document

from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import (
    get_standard_base_exception_msg,
    get_default_resultset,
)
# from genericsuite_ai.config.config import Config

from genericsuite.util.app_context import (
    AppContext,
    CommonAppContext,
)
from genericsuite_ai.lib.ai_utilities import (
    report_error,
    # standard_msg,
    get_response_lang,
    gpt_func_error,
)
from genericsuite_ai.lib.ai_chatbot_commons import (
    start_run_conversation,
    finish_run_conversation,
)
from genericsuite_ai.lib.ai_gpt_functions import (
    # get_function_list,
    get_functions_dict,
    get_function_specs,
    run_one_function,
    gpt_func_appcontext_assignment,
)
from genericsuite_ai.lib.ai_embeddings import (
    get_embeddings_engine,
)
# from genericsuite_ai.lib.ai_langchain_tools import (
#     messages_to_langchain_fmt,
# )
from genericsuite_ai.lib.ai_langchain_models import (
    get_model_obj,
    get_model_load_error,
)
from genericsuite_ai.lib.ai_gpt_fn_conversations import (
    get_conversation_buffer,
    get_current_date_time
)
from genericsuite_ai.lib.vector_index import (
    get_vector_engine,
)

DEBUG = False
PASS_ACTION_NOT_MATCH = True

cac = CommonAppContext()


class CustomPromptTemplate(StringPromptTemplate):
    """
    Set up a prompt template
    [Deprecated]
    """
    # The template to use
    template: str
    # The list of tools available
    tools_getter: Callable
    # The tool's retriever
    # tools_retriever: VectorStoreRetriever

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # tools = self.tools_getter(kwargs["input"], self.tools_retriever)
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    """
    Custom output parser for parsing LLM output
    [Deprecated]
    """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in text:
            return AgentFinish(
                # Return values is generally always a dictionary with a single
                # `output` key.
                # It is not recommended to try anything else at the moment :)
                return_values={"output":
                               text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        # Parse out the action and action input
        regex = \
            r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            if PASS_ACTION_NOT_MATCH:
                return AgentFinish(
                    # If the action is not found, the return value is generally
                    # the final text...
                    # It is not recommended to try anything else at the
                    # moment :)
                    return_values={"output": str(text)},
                    log=text,
                )
            raise ValueError("ERROR ACML-COP-010: " +
                             f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action,
            tool_input=action_input.strip(" ").strip('"'),
            log=text
        )


def run_one_function_langchain(args_raw: dict):
    """
    Execute a function (a.k.a. langachain Tool) based on the given
    args["function_name"] and function args.
    [Deprecated]

    Args:
        args (dict): function args

    Returns:
        The result of the function execution.
    """
    if isinstance(args_raw, str):
        args = json.loads(args_raw)
    else:
        args = args_raw
    if DEBUG:
        log_debug(f'>> RUN_ONE_FUNCTION_LANGCHAIN | args: {args}')

    if "function_name" not in args:
        return ("function_name is required. You only gave me the args: " +
                f" {args}. Please repeat this call with the function_name" +
                " and the args.")

    func_response = run_one_function(
        app_context=cac.app_context,
        function_name=args.get("function_name"),
        function_args=args,
    )
    return func_response["function_response"]


def get_tool_desc(tool_spec: dict) -> str:
    """
    Get the tool description from the tool spec and add additional
    instructions to call the function (tool).
    [Deprecated]

    Args:
        tool_spec (dict): The tool spec.

    Returns:
        str: The tool description.
    """
    self_debug = DEBUG

    param_as_dict = True

    def build_desc(k: str, v: dict, required: list) -> str:
        """ Build one description (with debug) """
        if self_debug:
            log_debug(f'>> build_desc | k: {k} | v: {v}')
        return '"' + k + '"' + \
               ' = ' + v["description"].rstrip('.') + \
               ', type "' + v["type"] + '"' + \
               (' (required)' if k in required
                else ' (optional)') + \
               ';'

    if param_as_dict:
        tool_desc = tool_spec["description"] + \
            "\n\nTo call this function, use these parameters: " + \
            json.dumps(tool_spec.get('parameters', {}).get('properties', {}),
                       indent=2) + \
            "\n\nThe required parameters are: " + \
            json.dumps(tool_spec.get('required', []), indent=2)
        # "Even if there's only one parameter, pass it as a dict" + \
        # " with the key as the parameter name."
    else:
        properties = tool_spec.get('parameters', {}).get('properties', {})
        required = tool_spec.get('required', [])
        param_desc = ""
        if properties:
            param_desc = \
                ". This tool has the parameters: " + \
                " ".join(
                    [
                        build_desc(k, v, required) for k, v in
                        properties.items()
                    ]
                )
        tool_desc = tool_spec["description"].rstrip('.') + param_desc

    if self_debug:
        log_debug('>> get_tool_desc:' +
                  f'\n\ntool_spec: {tool_spec}' +
                  f'\n\nRESULT tool_desc:\n{tool_desc}')
    return tool_desc


def get_tool_list():
    """
    Setup tools for the model.
    [Deprecated]

    Returns:
        list: The tools.
    """
    tool_funcs = get_functions_dict(cac.get())
    tool_specs = get_function_specs(cac.get())
    tools = [Tool(
        name=v["name"],
        func=tool_funcs[v["name"]],
        description=get_tool_desc(v),
        # func=run_one_function_langchain,
        # parameters=v.get("parameters", {}),
        # required=v.get("required", []),
    ) for v in tool_specs]
    return tools


def get_tool_retriever() -> dict:
    """
    Tool Retriever: get the vector store retriever for the
    Langchain Tools (a.k.a. GPT functions)
    [Deprecated]
    """
    self_debug = False

    result = get_default_resultset()

    # Store the tool list in the AppContext
    cac.app_context.set_other_data("all_tools", get_tool_list())

    def build_document(i: int, t: Tool) -> Document:
        """ Build one Document and debug data """
        metadata = {
            "index": i,
            "tool_name": t.name,
        }
        result = Document(page_content=t.description, metadata=metadata)
        if self_debug:
            log_debug('>> GET_TOOL_RETRIEVER -> build_document' +
                      f'\n | i: {i} | t.description: {t.description}' +
                      f'\n | result: {result}')
        return result

    # Create the array of Document() to be uused by the retriever
    docs = [
        build_document(i, t)
        for i, t in enumerate(cac.app_context.get_other_data("all_tools"))
    ]
    # Get the embedding engine
    embeddings = None
    try:
        embeddings = get_embeddings_engine(cac.get())
        if embeddings["error"]:
            result["error"] = True
            result["error_message"] = embeddings["error"]
    except Exception as err:
        result["error"] = True
        result["error_message"] = \
            get_standard_base_exception_msg(err, "GTR-010")
    if not result["error"]:
        # Get the Vector engine and additional Vector engine parameters
        try:
            vector_engine, other_params = get_vector_engine(cac.get())
        except Exception as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GTR-020")
    if not result["error"]:
        # Create a Vectorstore from a list of documents.
        try:
            vector_store = vector_engine.from_documents(
                docs, embeddings["engine"], **other_params
            )
        except PermissionDeniedError as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GTR-030")
        except Exception as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GTR-035")
            # raise e
    if not result["error"]:
        # Store the VectorStoreRetriever initialized from the created
        # VectorStore in the AppContext.
        cac.app_context.set_other_data("retriever",
                                       vector_store.as_retriever())
    return result


def get_tools_from_query(query: str) -> list:
    """
    Get tools that meets a given query.
    [Deprecated]

    Args:
        query (str): The query to be applied in the vector store.
        retriever (VectorStoreRetriever): The Tool Retriever.

    Returns:
        list: The tools.
    """
    debug_local = False
    if debug_local:
        log_debug(f"GTFQ-1) GET_TOOLS_FROM_QUERY | query: {query}")
    all_tools = cac.app_context.get_other_data("all_tools")
    retriever = cac.app_context.get_other_data("retriever")
    docs = retriever.get_relevant_documents(query)
    if debug_local:
        log_debug(f"GTFQ-2) GET_TOOLS_FROM_QUERY | docs: {docs}")
    return [all_tools[int(d.metadata["index"])] for d in docs]


def get_tools_base_template() -> str:
    """
    Set up the base template
    [Deprecated]
    """
    lang = get_response_lang(cac.get())
    template = """You are a helpful assistant.
Answer the following questions as best you can, but speaking as a kind person might speak.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a kind person when giving your final answer. Use lots of "Arg"s

Only the very very last Final Answer should be in """ + lang + """.

For date references, """ + get_current_date_time({}) + """.

If the called tool result has '""" + gpt_func_error("") + """', stop processing and report the error.

{chat_history}

Question: {input}
{agent_scratchpad}"""
    return template.strip()


# def get_tools_prompt(tools_retriever: VectorStoreRetriever,
#                      ) -> CustomPromptTemplate:
def get_tools_prompt() -> CustomPromptTemplate:
    """
    Get tool's prompt
    [Deprecated]

    Args:
        tools_retriever (VectorStoreRetriever): the Vector Store Retriever for
        the tools

    Returns:
        CustomPromptTemplate: The prompt.
    """
    prompt = CustomPromptTemplate(
        template=get_tools_base_template(),
        tools_getter=get_tools_from_query,
        # tools_retriever=tools_retriever,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names`
        # variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is
        # needed
        input_variables=["input", "intermediate_steps", "chat_history"],
    )
    if DEBUG:
        log_debug(f"GET_TOOLS_PROMPT | prompt: {prompt}")
    return prompt


def run_conversation_old(app_context: AppContext) -> dict:
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
    # settings = Config(app_context)
    cac.set(app_context)
    p = start_run_conversation(app_context=cac.app_context,
                               initial_prompt=False)
    if p["error"]:
        return report_error(p)
    conv_response = p["conv_response"]
    # query_params = p["query_params"]
    messages = p["messages"]

    # Prepare application context for all Tools (GPT Functions)
    gpt_func_appcontext_assignment(cac.app_context)

    # Get the user's question
    question = conv_response["question"]["content"]

    # Removes the last message, because it's the user's question,
    # which is already in conv_response["question"] and
    # we want the last message to be "The next question is: {topic}"
    messages = messages[:-1]

    # ............................
    # ............................

    # Custom agent with tool retrieval
    # https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval
    #           and...
    # Custom LLM Agent with tool retrieval and memory
    # https://python.langchain.com/docs/modules/agents/how_to/custom_llm_agent

    # ............................
    # ............................

    # 1) Message history (memory)
    memory = get_conversation_buffer(messages)

    # Add the topic placeholder
    # messages += [standard_msg("The next question is: {topic}")]

    # 2) Tool Retriever:

    # We will use a vector store to create embeddings for each tool
    # description. Then, for an incoming query we can create embeddings
    # for that query and do a similarity search for relevant tools.
    result_holder = get_tool_retriever()
    if result_holder["error"]:
        conv_response["error"] = True
        conv_response["error_message"] = result_holder["error_message"]
    # if not conv_response["error"]:
    #     tools_retriever = result_holder.get("retriever")

    # 3) Prompt template:

    # The prompt template is pretty standard, because we’re not actually
    # changing that much logic in the actual prompt template, but rather
    # we are just changing how retrieval is done.
    if not conv_response["error"]:
        # prompt = get_tools_prompt(tools_retriever)
        prompt = get_tools_prompt()

    # 4) Output parser:

    # The output parser is unchanged from the previous notebook, since
    # we are not changing anything about the output format.
    if not conv_response["error"]:
        output_parser = CustomOutputParser()

    # 5) Set up LLM, stop sequence, and the agent:

    # Set up the LLM (model)
    if not conv_response["error"]:
        model = get_model_obj(cac.get())
        if not model:
            conv_response["error"] = True
            conv_response["error_message"] = get_model_load_error(
                app_context=cac.app_context,
                error_code="AIRCLC_OLD-E010"
            )

    # LLM chain consisting of the LLM and a prompt
    if not conv_response["error"]:
        llm_chain = LLMChain(
            llm=model,
            prompt=prompt,
        )

    # Prepare the agent
    # tools = get_tools("whats the weather?")
    if not conv_response["error"]:
        # tools = get_tools_from_query(question, tools_retriever)
        tools = get_tools_from_query(question)
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

    # Use the Agent
    if not conv_response["error"]:
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
        )

    if not conv_response["error"]:
        try:
            conv_response["response"] = agent_executor.run({
                "input": question,
            })
        except Exception as err:
            conv_response["error"] = True
            conv_response["error_message"] = \
                get_standard_base_exception_msg(err, "AIRCLC_OLD-E020")

    return finish_run_conversation(
        app_context=cac.app_context,
        conv_response=conv_response,
        messages=messages,
    )
