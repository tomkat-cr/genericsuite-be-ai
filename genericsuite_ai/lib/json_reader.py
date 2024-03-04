"""
JSON indexer
"""
from typing import Union, Dict, Any
import os
import json
from uuid import uuid4

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.schema import Document

from genericsuite.util.app_logger import log_debug


DEBUG = False
REMOVE_JSON_TMP_FILES = True

def index_json_files(
    json_files: list,
    title: str = None,
    json_metadata: dict = None,
) -> list:
    """
    Indexes the content of JSON files and returns a list of Document
    objects.

    Args:
        json_files (list): List of JSON files to be indexed.

    Returns:
        list: List of Document objects containing the indexed content and
        metadata. This list can be combined calling get_vector_index(). E.g.
            index = get_vector_index(
                app_context,
                list1 + list2 + listN
            )
    """
    if not json_metadata:
        json_metadata = {}
    if not title:
        title = "Loaded JSON filename: {json_file}"
    all_pages = []
    for json_file in json_files:
        _ = DEBUG and log_debug(f"json_file: {json_file}")
        json_filename = os.path.basename(json_file)
        with open(json_file, 'r', encoding="utf-8") as file:
            json_content = json.load(file)
            output_file_spec = f"/tmp/{str(uuid4())}_{json_filename}"
            if not json_file.endswith(".json"):
                continue
            with open(output_file_spec, 'wb') as output_file:
                json_dict = {}
                json_dict['content'] = json_content
                # The attribute 'page_content' must be a string,
                # so the json content must be serialized
                # page_content = json.dumps(json_dict['content'])
                # json_dict['page_content'] = page_content
                json_content = bytes(json.dumps(json_dict), 'UTF8')
                _ = DEBUG and log_debug(f"json_content: {json_content}")
                output_file.write(json_content)
        # Load JSON file content
        loader = JSONLoader(
            output_file_spec,
            # jq_schema='.page_content',
            # Get all json content
            jq_schema='.',
            # This build the 'page_content' from the 'content' element as a str
            content_key='content',
            text_content=False,
        )
        json_pages = loader.load()
        if json_filename in json_metadata:
            metadata = json_metadata[json_filename]
        else:
            metadata = dict(json_metadata)
        # Add some context for the JSON file
        json_pages.append(
            Document(
                page_content=title,
                metadata=metadata
            )
        )
        all_pages += json_pages
    return all_pages


def index_dict(
    # dict_list: Union[list, dict],
    json_content: Union[list, dict],
    json_filename: str = None,
    title: str = None,
    json_metadata: dict = None
):
    """
    Indexes a list of dictionaries and returns a list of Document objects.

    Args:
        dict_list (list): List of dictionaries to be indexed.

    Returns:
        list: List of Document objects.
    """

    if not json_filename:
        json_filename = ""
    else:
        json_filename = f"_{json_filename}"
    if not json_metadata:
        json_metadata = {}
    if not title:
        title = "Loaded JSON filename: {json_file}"

    def get_metadata(sample: Dict[str, Any], additional_fields: dict = None):
        metadata = {
            "content_type": title,
        }
        metadata.update(json_metadata)
        return metadata

    _ = DEBUG and log_debug("INDEX_DICT" +
        f"| Creating json_filename {json_filename}"
        f"\n| json_metadata: {json_metadata}\n")

    output_file_spec = f"/tmp/{str(uuid4())}{json_filename}"
    if not output_file_spec.endswith(".json"):
        output_file_spec = f"{output_file_spec}.json"
    with open(output_file_spec, 'wb') as output_file:
        json_dict = {}
        json_dict['content'] = json_content
        # The attribute 'page_content' must be a string,
        # so the json content must be serialized
        # page_content = json.dumps(json_dict['content'])
        # json_dict['page_content'] = page_content
        # json_dict['page_content'] = title
        json_content = bytes(json.dumps(json_dict), 'UTF8')
        # _ = DEBUG and log_debug(f"json_content: {json_content}")
        output_file.write(json_content)
    # Load JSON file content
    loader = JSONLoader(
        file_path=output_file_spec,
        jq_schema='.',
        content_key='content',
        # content_key='page_content',
        text_content=False,
        metadata_func=get_metadata
    )
    json_pages = loader.load()
    # Add some context for the JSON file
    # json_pages.append(
    #     Document(
    #         page_content=title,
    #         metadata=json_metadata
    #     )
    # )
    if REMOVE_JSON_TMP_FILES:
        os.remove(output_file_spec)
    return json_pages


def prepare_metadata(metadata: dict, default_element_name: str = None) -> dict:
    """
    Prepares the metadata for a dict/json ingestion, so the value for each
    key is a dictionary, not a text.

    Args:
        metadata (dict): Metadata to add to the Document object.
        default_element_name (str): default name for the element to be added
        for string type entries. Defaults to None.

    Returns:
        dict: Prepared metadata.
    """
    if not default_element_name:
        default_element_name = "content"
    if not metadata:
        metadata = {}
    return {k: v if isinstance(v, dict)
            else {default_element_name: v}
            for k, v in metadata.items()
            }
