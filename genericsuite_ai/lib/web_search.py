"""
Web Search ability
"""

from typing import Union, Any, Optional
import json
import time
# from itertools import islice

from httpx._exceptions import HTTPError

# Pypi: pip install duckduckgo-search
from duckduckgo_search import DDGS

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper

from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import CommonAppContext
from genericsuite_ai.lib.ai_langchain_tools import (
    interpret_tool_params,
)
from genericsuite_ai.lib.ai_utilities import (
    gpt_func_error,
)
from genericsuite_ai.config.config import Config

DEBUG = False

cac = CommonAppContext()

DUCKDUCKGO_MAX_ATTEMPTS = 3


# from ..registry import ability
# @ability(
#     name="web_search",
#     description="Searches the web",
#     parameters=[
#         {
#             "name": "query",
#             "description": "The search query",
#             "type": "string",
#             "required": True,
#         }
#     ],
#     output_type="list[str]",
# )
# async def web_search(agent, task_id: str, query: str) -> str:


class WebSearch(BaseModel):
    """
    Web search parameters structure
    """
    query: str = Field(description="The search query")
    num_results: Optional[int] = Field(default=20, description="The number of results to return.")


@tool
def web_search(params: Any) -> str:
    """
Usefull when you need to perform a web search to have access to real-time information, and/or answer questions about recent events.
Args: params (dict): Tool parameter. Must contain:
"query" (str): The search query.
"num_results" (int): number of results to return. Defaults to 20.
    """
    return web_search_func(params)


def web_search_func(params: Any) -> str:
    """
    Performs a web search to have access to real-time information, and/or answer
    questions about recent events.

    Args:
        params (dict): Tool parameter. Must contain:
            "query" (str): The search query.
            "num_results" (int): number of results to return. Defaults to 20.

    Returns:
        str: The results of the search or [FUNC+ERROR] {error_message}
    """
    params = interpret_tool_params(
        tool_params=params,
        first_param_name="query",
        schema=WebSearch,
    )
    query = params.query
    num_results = params.num_results

    result = web_search_ddg_lc(query, num_results)
    if not result or "ERROR:" in result:
        result = web_search_google(query, num_results)
    return result


def web_search_google(query: str, num_results: int = 20) -> str:
    """
    Return the results of a Google search with a call to
    the langchain wrapper.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    settings = Config(cac.get())
    if DEBUG:
        log_debug(f"\n\n>> WEB_SEARCH_GOOGLE | query: {query}\n\n")
        log_debug(f"\n\ngoogle_api_key: {settings.GOOGLE_API_KEY}\n\n")
        log_debug(f"\n\ngoogle_cse_id: {settings.GOOGLE_CSE_ID}\n\n")

    # https://python.langchain.com/docs/integrations/tools/google_search#number-of-results
    try:
        search = GoogleSearchAPIWrapper(
            google_api_key=settings.GOOGLE_API_KEY,
            google_cse_id=settings.GOOGLE_CSE_ID,
        )
        results_list = search.results(query, num_results)
        results = safe_google_results(results_list)
    except Exception as error:
        results = gpt_func_error(error)

    if DEBUG:
        log_debug("\n\n>> WEB_SEARCH_GOOGLE | " +
                  f" results: {results}\n\n")

    return results


def web_search_ddg_lc(query: str, num_results: int = 20) -> str:
    """
    Return the results of a DuckDuckGo search with a call to
    the langchain wrapper.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    if DEBUG:
        log_debug(f"\n\n>> WEB_SEARCH_LANGCHAIN_DDG | query: {query}\n\n")

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=num_results)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    try:
        results_list = search.run(query)
        results = safe_google_results(results_list)
    except HTTPError as error:
        log_debug("\n\n>> WEB_SEARCH_LANGCHAIN_DDG" +
                  f" | error: {error}\n\n")
        results = gpt_func_error(error)

    if DEBUG:
        log_debug(f"\n\n>> WEB_SEARCH_LANGCHAIN_DDG | results: {results}\n\n")

    return results


def web_search_ddg(query: str, num_results: int = 20) -> str:
    """
    Return the results of a DuckDuckGo search with a direct call to
    the duckduckgo_search.DDGS function.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    attempts = 0

    if DEBUG:
        log_debug("")
        log_debug(f">> WEB_SEARCH | query: {query}")
        log_debug("")

    try:
        while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
            if not query:
                return json.dumps(search_results)
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=num_results))
            if search_results:
                break
            time.sleep(1)
            attempts += 1
        results = safe_google_results(search_results)
    except Exception as error:
        results = gpt_func_error(error)

    if DEBUG:
        log_debug(f">> WEB_SEARCH | results: {results}")
        log_debug("")

    return results


def safe_google_results(results: Union[str, list]) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps({
            "results": [result.encode("utf-8", "ignore").decode("utf-8")
                        for result in results],
        })
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")

    if DEBUG:
        log_debug(f">> WEB_SEARCH | results: {results}")
        log_debug("")

    return safe_message
