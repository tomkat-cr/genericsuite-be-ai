"""
YouTube video indexer
"""
from typing import Union, List
import os

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document

DEBUG = os.environ.get("AI_YOUTUBE_READER_DEBUG", "0") == "1"


def get_youtube_transcription(
        youtube_url: str,
        languages: Union[list, None] = None) -> List[Document]:
    """
    Get the transcription of a YouTube video.

    Args:
        youtube_url (str): The URL of the YouTube video.
        languages (list): A list of languages to try to extract the
            transcription in.

    Returns:
        str: The transcription of the YouTube video.
    """
    if not languages:
        languages = ['en', 'es']
    # Load the video and extract the transcription.
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=True,
        language=languages
    )
    return loader.load()
