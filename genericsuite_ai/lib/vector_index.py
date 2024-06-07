"""
Conversation Management with LangChain
"""
from typing import Union

# import pinecone
# import weaviate

from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain
)
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Clarifai
# from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores import Weaviate
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import (
    VectorStoreRetriever,
    # VectorStore
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from openai import PermissionDeniedError

from genericsuite.util.app_context import AppContext
from genericsuite.util.utilities import (
    get_standard_base_exception_msg,
    get_default_resultset,
    error_resultset,
    log_debug,
)

from genericsuite_ai.config.config import Config
from genericsuite_ai.lib.ai_langchain_models import (
    get_model_obj,
    get_model_load_error,
)
from genericsuite_ai.lib.ai_embeddings import get_embeddings_engine

DEBUG = False


def get_vector_engine(
    app_context: AppContext,
    engine_type: str = None
) -> (Union[
    MongoDBAtlasVectorSearch,
    Chroma,
    Clarifai,
    # Pinecone,
    Vectara,
    Weaviate,
    FAISS,
], dict):
    """
    Returns the configured vector engine object.

    Args:
        app_context (AppContext): the AppContext object.
        engine_type (str): Vector engine type.

    Returns:
        Union[...]: Vector engine object.
    """
    settings = Config(app_context)
    if not engine_type:
        engine_type = settings.VECTOR_STORE_ENGINE
    if DEBUG:
        log_debug(f"GET_VECTOR_ENGINE | engine_type: {engine_type}")

    other_params = {}
    if engine_type == 'mongo':
        # https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas
        vector_engine = MongoDBAtlasVectorSearch
        other_params = {
            "collection": settings.MONGODB_VS_COLLECTION,
            "index_name": settings.MONGODB_VS_INDEX_NAME,
        }
    elif engine_type == "chroma":
        # https://python.langchain.com/docs/integrations/vectorstores/chroma
        # NOTE: compatible only with python versions up to 3.9
        vector_engine = Chroma
    elif engine_type == "clarifai":
        # https://python.langchain.com/docs/integrations/vectorstores/clarifai
        vector_engine = Clarifai(
            user_id=settings.CLARIFAI_USER_ID,
            app_id=settings.CLARIFAI_APP_ID,
            # number_of_docs=NUMBER_OF_DOCS,
        )
    # elif engine_type == "Pinecone":
        # https://python.langchain.com/docs/integrations/vectorstores/pinecone
        # pinecone.init(
        #     api_key=settings.PINECONE_API_KEY,  # find at app.pinecone.io
        #     environment=settings.PINECONE_ENV,  # next to api key in console
        # )
        # engine = Pinecone
    elif engine_type == "vectara":
        # https://python.langchain.com/docs/integrations/vectorstores/vectara
        vector_engine = Vectara(
            vectara_customer_id=settings.VECTARA_CUSTOMER_ID,
            vectara_corpus_id=settings.VECTARA_CORPUS_ID,
            vectara_api_key=settings.VECTARA_API_KEY
        )
    # elif engine_type == "weaviate":
    #     # https://python.langchain.com/docs/integrations/vectorstores/weaviate
    #     vector_engine = Weaviate
    #     client = weaviate.Client(
    #         url=settings.WEAVIATE_URL,
    #         auth_client_secret=weaviate.AuthApiKey(settings.WEAVIATE_API_KEY),
    #     )
    #     other_params = {
    #         "client": client,
    #         "by_text": False,
    #     }
    else:
        vector_engine = FAISS

    if DEBUG and False:
        log_debug("GET_VECTOR_ENGINE:\n" +
                  f" vector_engine: {vector_engine}\n" +
                  f" other_params: {other_params}\n")
    return vector_engine, other_params


def get_vector_index(
    app_context: AppContext,
    documents_list: list[Document],
) -> dict:
    """
    Create a vector store index from the given vector store.

    Args:
        app_context (AppContext): the AppContext object.
        documents_list (list): List of vector entries.

    Returns:
        dict: a resulset with "index" attribute the created vector store index
            of type (VectorStoreIndexWrapper). If something goes wrong, the
            "error" attr. will be True and "error_message" will have the
            error message.
    """
    if len(documents_list) == 0:
        return error_resultset("No documents to index", "GVI2-E005")
    result = get_default_resultset()
    embeddings = {}
    try:
        # Get Embeddings engine
        _ = DEBUG and log_debug("GVI2_1) GET_VECTOR_INDEX | Get Embeddings engine...")
        embeddings = get_embeddings_engine(app_context)
        if embeddings["error"]:
            result["error"] = True
            result["error_message"] = embeddings["error"]
    except Exception as err:
        result["error"] = True
        result["error_message"] = \
            get_standard_base_exception_msg(err, "GVI2-E010")
    if not result["error"]:
        # Get Vector engine and additional vector engine parameters
        _ = DEBUG and log_debug("GVI2_2) GET_VECTOR_INDEX | get Vector engine...")
        try:
            vector_engine, other_params = get_vector_engine(app_context)
        except Exception as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GVI2-E020")
    if not result["error"]:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents_list)
    if not result["error"]:
        # Create a Vectorstore from a list of splitted documents
        try:
            _ = DEBUG and log_debug("GVI2_3) GET_VECTOR_INDEX | " +
                "get Vectorstore from a list of documents...")
            vectorstore = vector_engine.from_documents(
                documents=texts,
                embedding=embeddings["engine"],
                **other_params
            )
        except PermissionDeniedError as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GVI2-E030")
        except Exception as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GVI2-E035")
            # raise e
    if not result["error"]:
        # Returns the vector store index.
        _ = DEBUG and log_debug("GVI2_4) GET_VECTOR_INDEX | " +
            "create the Vectorstore index...")
        result["index"] = VectorStoreIndexWrapper(vectorstore=vectorstore)
    _ = DEBUG and log_debug("GVI2_4) GET_VECTOR_INDEX | index creation finished...")
    return result


def get_vector_store(
    app_context: AppContext,
    documents_list: list[Document],
) -> dict:
    """
    Create a vector store from the given document_list.

    Args:
        app_context (AppContext): the AppContext object.
        documents_list (list): List of vector entries.

    Returns:
        dict: a resulset with "vector_store" attribute the created vector store
            of type (VectorStore). If something goes wrong, the
            "error" attr. will be True and "error_message" will have the
            error message.
    """
    if len(documents_list) == 0:
        return error_resultset("No documents to index", "GVS-E005")
    result = get_default_resultset()
    embeddings = {}
    try:
        # Get Embeddings engine
        _ = DEBUG and log_debug("GVS_1) GET_VECTOR_STORE | Get Embeddings engine...")
        embeddings = get_embeddings_engine(app_context)
        if embeddings["error"]:
            result["error"] = True
            result["error_message"] = embeddings["error"]
    except Exception as err:
        result["error"] = True
        result["error_message"] = \
            get_standard_base_exception_msg(err, "GVS-E010")
    if not result["error"]:
        # Get Vector engine and additional vector engine parameters
        _ = DEBUG and log_debug("GVS_2) GET_VECTOR_STORE | get Vector engine...")
        try:
            vector_engine, other_params = get_vector_engine(app_context)
        except Exception as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GVS-E020")
    if not result["error"]:
        # Create a Vectorstore from a list of splitted documents
        try:
            _ = DEBUG and log_debug("GVS_3) GET_VECTOR_STORE | " +
                "get Vectorstore from a list of documents...")
            vectorstore = vector_engine.from_documents(
                documents=documents_list,
                embedding=embeddings["engine"],
                **other_params
            )
        except PermissionDeniedError as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GVS-E030")
        except Exception as err:
            result["error"] = True
            result["error_message"] = \
                get_standard_base_exception_msg(err, "GVS-E035")
            # raise e
    if not result["error"]:
        # Returns the vector store index.
        _ = DEBUG and log_debug("GVS_4) GET_VECTOR_STORE | " +
            "create the Vectorstore...")
        result["vector_store"] = vectorstore
    _ = DEBUG and log_debug("GVS_4) GET_VECTOR_STORE creation finished...")
    return result


def get_vector_retriever(
    app_context: AppContext,
    documents_list: list[Document]
) -> VectorStoreRetriever:
    """
    Create a vector store retriever from the given vector store.

    Args:
        app_context (AppContext): the AppContext object.
        documents_list (list): List of vector entries.

    Returns:
        VectorStoreRetriever: The created vector store retriever.
    """
    embeddings = get_embeddings_engine(app_context)
    if embeddings["error"]:
        raise Exception(embeddings["error"])
    vector_engine, other_params = get_vector_engine(app_context)
    vector_store = vector_engine.from_documents(
        documents_list,
        embeddings["engine"],
        **other_params
    )
    retriever = vector_store.as_retriever()
    return retriever


def get_conversation_chain(
    app_context: AppContext,
    retriever: VectorStoreRetriever,
    prompt: ChatPromptTemplate,
) -> ConversationalRetrievalChain:
    """
    Create a conversation chain using the vector store index and GPT model.

    Args:
        app_context (AppContext): the AppContext object.
        retriever (VectorStoreIndexWrapper or VectorStoreRetriever):
            The vector store index or retriever.
        prompt (ChatPromptTemplate): the prompt as ChatPromptTemplate.

    Returns:
        ConversationalRetrievalChain: The created conversation chain.
    """
    _ = DEBUG and log_debug('AI_GCC_1 ) GET_CONVERSATION_CHAIN' +
        f'\n | retriever: {retriever}' +
        f'\n | prompt: {prompt}')

    model = get_model_obj(app_context)
    if not model:
        error_message = get_model_load_error(
            app_context=app_context,
            error_code="AI_GCC-E010",
        )
        raise Exception(error_message)

    _ = DEBUG and log_debug('AI_GCC_2 ) GET_CONVERSATION_CHAIN' +
        f'\n | model: {model}')

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    _ = DEBUG and log_debug('AI_GCC_3 ) GET_CONVERSATION_CHAIN' +
        f'\n | chain: {chain}')

    return chain
