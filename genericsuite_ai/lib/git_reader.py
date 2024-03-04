"""
Git repositories loader
Pre-requisites:
$ pipenv install GitPython
"""
import os

from git import Repo
from langchain_community.document_loaders.git import GitLoader
from langchain.schema import Document

from genericsuite.util.utilities import get_default_resultset
from genericsuite.util.app_logger import log_debug, log_error


def remove_dir(local_temp_path: str) -> None:
    """
    Removes the "local_temp_path" directory
    """
    if local_temp_path == '/':
        raise Exception("Cannot use / as a temp path")
    if not os.path.exists(local_temp_path):
        return
    if os.path.isfile(local_temp_path):
        os.remove(local_temp_path)
        return
    for root, dirs, files in os.walk(local_temp_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def get_repo(repo_url: str, branch: str = None) -> dict:
    """
    Load `Git` repository files.
    
    Args:
        repo_url (str): repo URL or local repo path.
        branch (str): brach name. If none, get repo.head.reference.
            Defaults to None.

    Returns:
        dict: resultset with a "data" attribute with all Git repository files
            Document object list. If something goes wrong, returns the
            "error" attr. True and "error_message" attr. with the error message.
    """
    response = get_default_resultset()
    repo_name = repo_url.rsplit('/', 1)[1]
    if repo_url[:8] == "https://":
        repo_type = "remote"
        # https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/git
        local_temp_repo_path = f"/tmp/{repo_name}"
        try:
            remove_dir(local_temp_repo_path)
        except Exception as err:
            # if DEBUG:
            #     raise
            response["error"] = True
            response["error_message"] = f"ERROR: {str(err)}"
            return response
        try:
            repo = Repo.clone_from(
                url=repo_url,
                to_path=local_temp_repo_path
            )
        except Exception as err:
            # if DEBUG:
            #     raise
            response["error"] = True
            response["error_message"] = f"ERROR: {str(err)}"
            return response
    else:
        repo_type = "local"
        try:
            local_temp_repo_path = repo_url
            repo = Repo(local_temp_repo_path)
        except Exception as err:
            # if DEBUG:
            #     raise
            response["error"] = True
            response["error_message"] = f"ERROR: {str(err)}"
            return response
    if not branch:
        try:
            branch = repo.head.reference
        except Exception as err:
            # if DEBUG:
            #     raise
            response["error"] = True
            response["error_message"] = f"ERROR: {str(err)}"
            return response
    log_debug(f"Github repository URL: {repo_url}" +
              f"\nBranch (default 'main'): {branch}")
    try:
        loader = GitLoader(
            repo_path=f"{local_temp_repo_path}/",
            branch=branch,
        )
    except Exception as err:
        # if DEBUG:
        #     raise
        response["error"] = True
        response["error_message"] = f"ERROR: {str(err)}"
        return response

    response["data"] = loader.load()

    # Add some context for the repo
    metadata = {
        "source": repo_url,
        "branch": branch,
        "repo_type": repo_type,
        "repo_name": repo_name,
        "repo_path": local_temp_repo_path,
        "comments": ",".join([
                    "this is the context of the:",
                    "repo context",
                    "repository context",
                    "supplied repo"
                    "supplied repository",
                    "loaded repo",
                    "loaded repository",
                    "supplied git repo",
                    "supplied git repository",
                    "loaded git repo",
                    "loaded git repository",
                    "supplied branch",
                    "loaded branch",
            ])
    }
    response["data"].append(
        Document(
            page_content=f"The supplied git repo name is {repo_name}",
            metadata=metadata
        )
    )
    response["data"].append(
        Document(
            page_content=f"{repo_type} git repo content of {repo_url}",
            metadata=metadata
        )
    )

    log_debug(f"Repo: {repo_url}")
    log_debug(f"Branch: {branch}")
    log_debug("Data:")
    log_debug(response["data"])
    return response


def get_repo_data(repo_url: str, branch: str = None) -> list:
    """
    Load `Git` repository files.
    
    Args:
        repo_url (str): repo URL or local repo path.
        branch (str): brach name. If none, get repo.head.reference.
            Defaults to None.

    Returns:
        list: list of Document objects of all Git repository files.
            If something goes wrong, returns [].
    """
    if repo_url == "":
        return []
    repo_response = get_repo(repo_url, branch)
    if repo_response["error"]:
        log_error('GET_REPO_DATA' + 
                  f'\n | repo_url: {repo_url}' +
                  f'\n | branch: {branch}' +
                  f'\n | error_message: {repo_response["error_message"]}')
        return []
    repo_data = repo_response["data"]
    return repo_data
