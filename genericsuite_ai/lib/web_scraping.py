"""
Web scrapping module
"""
from typing import Any
import importlib
# import asyncio

from langchain_community.document_loaders.web_base import WebBaseLoader

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.agents import tool
from pydantic import BaseModel, Field

from genericsuite.util.app_context import CommonAppContext
from genericsuite.util.app_logger import log_debug
from genericsuite.util.utilities import get_default_resultset, error_resultset

from genericsuite_ai.lib.vector_index import get_vector_store
from genericsuite_ai.lib.ai_langchain_models import (
    get_model_obj,
    get_model_load_error,
)
from genericsuite_ai.lib.ai_langchain_tools import (
    interpret_tool_params,
)
from genericsuite_ai.lib.ai_utilities import (
    gpt_func_error,
)

DEBUG = False
cac = CommonAppContext()


class WebpageAnalyzerParams(BaseModel):
    """
    Vision parameters structure
    """
    url: str = Field(description="Webpage URL")
    question: str = Field(description="A question about the webpage")


def get_qa_chain(
    url: str
) -> dict:
    """
    Get a QA chain for the given webpage URL.

    Args:
        url (str): The URL of the webpage.

    Returns:
        dict: a standard resultset with the following keys:
            qa_chain (BaseRetrievalQA): A QA chain for the webpage.
            error: True if there is an error.
            error_message: The error message if there is an error.
    """
    result = get_default_resultset()

    try:
        importlib.import_module("bs4")
    except ImportError:
        # except ImportError as exc:
        # raise ImportError("The 'bs4' package is required for this" +
        #     " operation. Please install it using " +
        #     "'pip install beautifulsoup4'.") from exc
        result = error_resultset(
            "The 'bs4' package is required for this operation." +
            " Please install it using 'pip install beautifulsoup4'.",
            "WS_GQC-E010")

    if not result["error"]:
        # Load and split the webpage content
        loader = WebBaseLoader(url)
        data = loader.load()

        _ = DEBUG and \
            log_debug(
                "GET_QA_CHAIN | WebBaseLoader:" +
                f"\n | data: {data}")
        # Embed the webpage content
        get_vectorstore = get_vector_store(
            app_context=cac.app_context,
            documents_list=data)
        # embeddings = OpenAIEmbeddings()
        # vectorstore = Chroma.from_documents(documents=[data],
        #   embedding=embeddings)
        if get_vectorstore["error"]:
            result = error_resultset(
                get_vectorstore["error_message"],
                "WS_GQC-E020")

    if not result["error"]:
        llm = get_model_obj(cac.app_context)
        if not llm:
            result = error_resultset(get_model_load_error(
                app_context=cac.app_context,
                error_code="WS_GQC-E010"))

    result["qa_chain"] = None
    if not result["error"]:
        # Build a QA chain
        vectorstore = get_vectorstore["vector_store"]
        result["qa_chain"] = RetrievalQA.from_chain_type(
            # llm="gpt-3.5-turbo",  # Specify the LLM model you want to use
            llm=llm,  # Specify the LLM model you want to use

            # This should be replaced with the actual chain type you need
            chain_type="stuff",
            # chain_type="refine",
            # ValueError: Got unsupported chain type: retrieval_qa. Should be
            # one of dict_keys(['stuff', 'map_reduce', 'refine', 'map_rerank'])

            retriever=vectorstore.as_retriever(),
        )

    _ = DEBUG and \
        log_debug(
            "GET_QA_CHAIN finished:" +
            f"\n | result: {result}" +
            "\n")
    return result


async def ask_question(qa_chain: Any, question: str) -> str:
    """
    Ask a question about the webpage using the QA chain.

    Args:
        qa_chain (BaseRetrievalQA): The QA chain to use.
        question (str): The question to ask.

    Returns:
        str: The answer to the question.
    """
    response = await qa_chain.run(question)
    return response


def webpage_analyzer(
    params: dict
) -> str:
    """
    Webpage analyzer.

    Args:
        params (dict): function parameters. It must include:
            "url" (str): The URL of the webpage.
            "question" (str): The question to ask. E.g.
                "What is the main topic of the webpage?"

    Returns:
        str: The answer to the question.
    """
    params = interpret_tool_params(tool_params=params,
                                   first_param_name="url",
                                   schema=WebpageAnalyzerParams)

    url = params.url
    question = params.question

    # Run the QA chain asynchronously and return the answer.
    # This is a blocking operation, so we need to run it in an asynchronous
    # context.
    # We use asyncio to run the QA chain in an asynchronous context.
    # We use the ask_question function to run the QA chain asynchronously.
    # We pass the QA chain and the question to the ask_question function.
    # The ask_question function returns the answer to the question.
    # We print the answer to the question and return it.
    # The ask_question function is an asynchronous function, so we need to use
    # the asyncio.get_event_loop() function to get the event loop.
    # We use the run_until_complete() method of the event loop to run the
    # asynchronous function and wait for it to complete.
    # The run_until_complete() method returns the result of the asynchronous
    # function.
    # We print the answer to the question and return it.
    result = get_default_resultset()

    qa_chain_result = get_qa_chain(url)
    if qa_chain_result["error"]:
        result = error_resultset(qa_chain_result["error_message"], "WA_E010")

    if not result["error"]:
        qa_chain = qa_chain_result["qa_chain"]

        _ = DEBUG and \
            log_debug(
                "WEBPAGE_ANALYZER" +
                f"\n | qa_chain: {qa_chain}"
                # f'\n | qa_chain_result["qa_chain"]:
                # {qa_chain_result["qa_chain"]}'
                )

        # loop = asyncio.get_event_loop()
        # result["answer"] = \
        #     loop.run_until_complete(ask_question(qa_chain, question))
        result["answer"] = qa_chain.run(question)

    _ = DEBUG and \
        log_debug(
            "WEBPAGE_ANALYZER" +
            f"\n | url: {url}" +
            f"\n | question: {question}" +
            f"\n | result: {result}" +
            "\n")
    return result


@tool
def webpage_analyzer_text_response(params: Any) -> str:
    """
Useful to answer a question about a given webpage URL.
Args: params (dict): Tool parameters. Must contain:
"url" (str): webpage URL.
"question" (str): question about the webpage.
    """
    return webpage_analyzer_text_response_func(params)


def webpage_analyzer_text_response_func(params: Any) -> str:
    """
    Answer a question about a given webpage URL.

    Args:
        params (dict): Tool parameters. Must contain:
            "url" (str): webpage URL.
            "question" (str): question about the webpage.

    Returns:
        str: Answer to the questions or description of the supplied URL,
            or [FUNC+ERROR] {error_message}
    """
    wp_analyzer_resp = webpage_analyzer(params)
    if wp_analyzer_resp["error"]:
        response = gpt_func_error(wp_analyzer_resp["error_message"])
    else:
        response = wp_analyzer_resp["answer"]

    if DEBUG:
        log_debug("Text formatted answer from" +
                  " WEBPAGE_ANALYZER_TEXT_RESPONSE:" +
                  f"\n | response: {response}" +
                  "\n")
    return response
