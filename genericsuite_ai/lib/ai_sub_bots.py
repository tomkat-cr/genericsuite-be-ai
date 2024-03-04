"""
Sub Bots module, to ask quick questions to the model using vectors
"""
from typing import List
import json

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from genericsuite.util.app_context import AppContext
from genericsuite_ai.lib.vector_index import (
    get_vector_index,
    get_conversation_chain,
    get_vector_retriever,
)
from genericsuite.util.app_logger import log_debug, log_error
from genericsuite.util.utilities import (
    get_default_resultset,
)

DEBUG = False


def ask_ai(
    app_context: AppContext,
    documents_list: list,
    question: str,
    response_type: str = "answer",
) -> dict:
    """
    Ask the AI the question.

    Args:
        documents_list (list): vectored documents list
        question (str): question to be answered.
        response_type (str): response type to be returned.
            Possible values: "answer" or "list".
            Default value: "answer".

    Returns:
        dict: the standard resultset with answer or error
    """
    if DEBUG:
        log_debug('ASK_AI - 1' +
                  f'\n | documents_list: {documents_list}' +
                  f'\n | question: {question}' +
                  f'\n | response_type: {response_type}')
    result = get_default_resultset()
    try:
        if response_type == "answer":
            # Get the vector index
            vector_index_result = get_vector_index(app_context, documents_list)
            if vector_index_result["error"]:
                result["error"] = True
                result["error_message"] = vector_index_result["error_message"]
                log_error(f'ASK_AI-E010 | error: {result["error_message"]}')
                return result
            index: VectorStoreIndexWrapper = vector_index_result["index"]
            # Build the prompt
            template = """
Answer the question based only on the following context:
{context}

Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            # Get the conversation chain
            chain = get_conversation_chain(
                app_context=app_context,
                retriever=index.vectorstore.as_retriever(),
                prompt=prompt,
            )
            # Ask the question to the model
            result["model_response"] = chain.invoke(question)
            _ = DEBUG and log_debug('ASK_AI - 2 | result["model_response"]:' +
                                    f' {result["model_response"]}')
            result["resultset"] = result["model_response"]
        else:
            # Response type == "list"
            retriever = get_vector_retriever(app_context,
                documents_list)
            docs: List[Document] = retriever.get_relevant_documents(question)
            result["resultset"] = json.dumps([d.page_content for d in docs])
    except Exception as error:
        result["error"] = True
        result["error_message"] = str(error)
        log_error(f'ASK_AI-E020 | error: {error}')
    _ = DEBUG and log_debug(f'ASK_AI - 3 | result: {result}')
    return result
