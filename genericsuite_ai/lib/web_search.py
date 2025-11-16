"""
Web Search ability
"""

from typing import Dict, Union, Any, Optional
import json
import time
import os

from ddgs import DDGS
from googleapiclient.discovery import build

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import tool

from pydantic import BaseModel, Field

from genericsuite.util.app_logger import log_debug
from genericsuite.util.app_context import CommonAppContext
from genericsuite_ai.lib.ai_langchain_tools import (
    interpret_tool_params,
)
from genericsuite_ai.lib.ai_utilities import (
    gpt_func_error,
)
from genericsuite_ai.config.config import Config

DEBUG = os.environ.get("AI_WEBSEARCH_DEBUG", "0") == "1"

cac = CommonAppContext()

DUCKDUCKGO_METHOD = os.getenv("WEBSEARCH_DUCKDUCKGO_METHOD", "ddgs")
DUCKDUCKGO_MAX_ATTEMPTS = 3
DUCKDUCKGO_MAX_RESULTS = 5
DUCKDUCKGO_RATE_LIMIT_TOKEN = "202 Ratelimit"
DUCKDUCKGO_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64;" + \
                        " rv:124.0) Gecko/20100101 Firefox/124.0"

GOOGLE_MAX_RESULTS = 100
GOOGLE_MAX_PAGINATED_RESULTS = 10

DEFAULT_MAX_RESULTS = 15


class WebSearch(BaseModel):
    """
    Web search parameters structure
    """
    query: str = Field(description="The search query")
    num_results: Optional[int] = Field(
        default=DEFAULT_MAX_RESULTS,
        description="The number of results to return.")


@tool
def web_search(params: Dict) -> str:
    """
Useful when you need to perform a web search to have access to real-time information, and/or answer questions about recent events.
Args: params (dict): Tool parameter. Must contain:
"query" (str): The search query.
"num_results" (int): number of results to return. Defaults to 30.
    """  # noqa: E501
    return web_search_func(params)


def web_search_func(params: Any) -> str:
    """
    Performs a web search to have access to real-time information,
    and/or answer questions about recent events.

    Args:
        params (dict): Tool parameter. Must contain:
            "query" (str): The search query.
            "num_results" (int): number of results to return.

    Returns:
        str: The results of the search or [FUNC+ERROR] {error_message}
    """
    settings = Config(cac.get())
    params = interpret_tool_params(
        tool_params=params,
        first_param_name="query",
        schema=WebSearch,
    )
    query = params.query
    num_results = params.num_results
    func_error_token = gpt_func_error('').strip()

    if settings.WEBSEARCH_DEFAULT_PROVIDER == "ddg":
        if DUCKDUCKGO_METHOD == "ddgs":
            result = web_search_ddg(query, num_results)
        else:
            result = web_search_ddg_lc(query, num_results)
    elif settings.WEBSEARCH_DEFAULT_PROVIDER == "google":
        result = web_search_google(query, num_results)
    else:
        if DUCKDUCKGO_METHOD == "ddgs":
            result = web_search_ddg(query, num_results)
        else:
            result = web_search_ddg_lc(query, num_results)
        if not result or func_error_token in result or \
                DUCKDUCKGO_RATE_LIMIT_TOKEN in result:
            result = web_search_google(query, num_results)

    return result


def web_search_google(query: str, num_results: int = DEFAULT_MAX_RESULTS
                      ) -> str:
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
        log_debug(
            f">> 1) WEB_SEARCH_GOOGLE | query: {query}" +
            f"\ngoogle_api_key: {settings.GOOGLE_API_KEY}" +
            f"\ngoogle_cse_id: {settings.GOOGLE_CSE_ID}" +
            f"\nnum_results: {num_results}")

    # https://stackoverflow.com/questions/76547862/why-is-my-google-custom-search-api-call-from-python-not-working
    # To fix the error "'Request contains an invalid argument.',
    #   'domain': 'global', 'reason': 'badRequest'"
    max_results = min(int(num_results), GOOGLE_MAX_RESULTS)

    # https://python.langchain.com/docs/integrations/tools/google_search#number-of-results
    try:
        results_list = google_search_paginated(query, settings.GOOGLE_API_KEY,
                                               settings.GOOGLE_CSE_ID,
                                               num_results=max_results)
        results = safe_google_results(results_list)
    except Exception as error:
        results = gpt_func_error(error)

    if DEBUG:
        log_debug(">> 2) WEB_SEARCH_GOOGLE | " +
                  f" results: {results}\n\n")

    return results


def google_search_paginated(search_term: str, api_key: str, cse_id: str,
                            num_results: int = DEFAULT_MAX_RESULTS,
                            **kwargs) -> list:
    service = build("customsearch", "v1", developerKey=api_key)
    start_index = 1
    all_results = []

    max_results = min(
        int(num_results),
        # (100 - 10 + 1) to ensure that start + num does not exceed 100
        GOOGLE_MAX_RESULTS-GOOGLE_MAX_PAGINATED_RESULTS+1
    )

    while True:
        # Makes the API call with the 'start' parameter and 'num=10'
        # The 'num' parameter is set to 10 to comply with the API limit
        res = service.cse().list(q=search_term, cx=cse_id, start=start_index,
                                 num=GOOGLE_MAX_PAGINATED_RESULTS,
                                 **kwargs).execute()

        _ = DEBUG and log_debug(f"google_search_paginated | res: {res}")

        # Adds the results of the current page to the total list
        if 'items' in res:
            all_results.extend(res['items'])

        _ = DEBUG and log_debug(
            f"google_search_paginated | all_results: {all_results}")

        # Verifies if there is a next page.
        # The method.get() is used to avoid errors if 'queries' does not exist.
        queries_info = res.get('queries', {})
        if 'nextPage' not in queries_info:
            break   # No more pages; exit loop

        _ = DEBUG and log_debug(
            f"google_search_paginated | queries_info: {queries_info}")

        # Updates the start index for the next request
        # The startIndex for the next page is found in the first
        # element of the nextPage array
        start_index = queries_info['nextPage'][0]['startIndex']

        _ = DEBUG and log_debug(
            f"google_search_paginated | start_index: {start_index}")

        # Includes a verification to not exceed the total limit of 100 results
        # If the next start_index would lead to a sum of start + num exceeding
        # 100, it stops.
        if start_index > max_results:
            break

    return all_results


def web_search_ddg_lc(query: str, num_results: int = DEFAULT_MAX_RESULTS
                      ) -> str:
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

    max_results = min(int(num_results), DUCKDUCKGO_MAX_RESULTS)
    if int(num_results) > max_results:
        max_results = DUCKDUCKGO_MAX_RESULTS

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    try:
        results_list = search.run(query)
        results = safe_ddg_results(results_list)
    except Exception as error:
        log_debug("\n\n>> WEB_SEARCH_LANGCHAIN_DDG" +
                  f" | error: {error}\n\n")
        results = gpt_func_error(f'{error}')

    if DEBUG:
        log_debug(f"\n\n>> WEB_SEARCH_LANGCHAIN_DDG | results: {results}\n\n")

    return results


def web_search_ddg(query: str, num_results: int = DEFAULT_MAX_RESULTS) -> str:
    """
    Return the results of a DuckDuckGo search with a direct call to
    the duckduckgo_search.DDGS function.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results: list[Any] = []
    attempts = 0
    headers = {
        "User-Agent": DUCKDUCKGO_USER_AGENT
    }

    if DEBUG:
        log_debug("")
        log_debug(f">> WEB_SEARCH | query: {query}")
        log_debug("")

    try:
        while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
            if not query:
                return json.dumps(search_results)
            with DDGS() as ddgs:
                num_results = min(int(num_results), DUCKDUCKGO_MAX_RESULTS)
                search_results = list(ddgs.text(
                    query,
                    max_results=num_results,
                    search_headers=headers
                ))
            if search_results:
                break
            time.sleep(1)
            attempts += 1
        results = safe_google_results(search_results)
    except Exception as error:
        results = gpt_func_error(f'{error}')

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
    _ = DEBUG and \
        log_debug(f">> 1) SAFE_GOOGLE_RESULTS | results: {results}")
    if isinstance(results, list):
        safe_message = json.dumps({
            "results": [{
                'Result': result.get('Result', '').encode(
                    "utf-8", "ignore").decode("utf-8"),
            } if 'Result' in result else {
                'snippet': result.get('snippet', '').encode(
                    "utf-8", "ignore").decode("utf-8"),
                'title': result.get('title', '').encode(
                    "utf-8", "ignore").decode("utf-8"),
                'link': result.get('href', '').encode(
                    "utf-8", "ignore").decode("utf-8"),
            } for result in results],
        })
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    _ = DEBUG and \
        log_debug(f">> 2) SAFE_GOOGLE_RESULTS | results: {results}")
    return safe_message


def safe_ddg_results(results: Union[str, list]) -> str:
    """
        Return the results of a DuckduckGo search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    _ = DEBUG and \
        log_debug(
            ">> 1) SAFE_DDG_RESULTS" +
            f"\n | results TYPE: {type(results)}" +
            f"\n | results: {results}")
    if isinstance(results, list):
        safe_message = json.dumps({
            "results": [{
                'Result': result.Result.encode(
                    "utf-8", "ignore").decode("utf-8"),
            } if hasattr(result, 'Result') else {
                'snippet': result.snippet.encode(
                    "utf-8", "ignore").decode("utf-8"),
                'title': result.title.encode(
                    "utf-8", "ignore").decode("utf-8"),
                'link': result.link.encode(
                    "utf-8", "ignore").decode("utf-8"),
            } for result in results],
        })
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    _ = DEBUG and \
        log_debug(
            ">> 2) SAFE_DDG_RESULTS" +
            f"\n | results TYPE: {type(results)}" +
            f"\n | results: {results}")
    return safe_message
