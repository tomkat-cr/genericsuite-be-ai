"""
Translation module
"""
import re
import html
import urllib.request
import urllib.parse

from genericsuite.util.utilities import get_default_resultset
from genericsuite.util.app_logger import log_debug

DEBUG = False


def lang_code_equivalence(lang_code: str) -> str:
    """
    Get the desired response language description based on the user's
    preference, in plain english.
    """
    lang_code_trans = {
        'es': 'es',
    }
    return lang_code_trans.get(lang_code, lang_code)


def translate(
    text: str,
    dest: str = "es",
    src: str = "auto"
) -> dict:
    """
    Translates texts using the Google Translate API.

    Args:
        _text (str): The text to translate.
        dest (str, optional): The language to translate the text to.
        Defaults to "es" (spanish/español).
        src (str, optional): The language of the text to translate.
        Defaults to "auto".

    Returns:
        dict: a resultset with an "text" attribute containing
        the translated input text, or "error" and "error_message"
        attributes if there's something wrong.
    """
    response = get_default_resultset()
    dest_lang = lang_code_equivalence(dest)
    try:
        gta_url = f"http://translate.google.com/m?tl={dest_lang}&" + \
                f"sl={src}&q={urllib.parse.quote(text)}"
        headers = {
            'User-Agent':
            "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET" +
            " CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"
        }
        request = urllib.request.Request(gta_url, headers=headers)
        with urllib.request.urlopen(request) as res_alloc:
            res_alloc = res_alloc.read()
            re_result = re.findall(
                r'(?s)class="(?:t0|result-container)">(.*?)<',
                res_alloc.decode("utf-8")
            )
        response["text"] = (
            "" if len(re_result) == 0
            else html.unescape(re_result[0])
        )
    except Exception as err:
        response['error'] = True
        response['error_message'] = f"ERROR [GLT-010]: {str(err)}"
    if DEBUG:
        log_debug(">> GOOGLE TRANSLATE()" +
                  f"\n | text: {text}"
                  f"\n | dest: {dest} / dest_lang: {dest_lang}"
                  f"\n | src: {src}"
                  f"\n | response: {response}")
    return response


def lang_translate(
    input_text: str,
    target_lang: str = 'en',
    source_lang: str = 'auto',
) -> dict:
    """
    Translates texts to a target language and returns the translation
    in th standard format with error handling. It acts as a middleware.

    Args:
        input_text (str): The text to translate.
        target_lang (str, optional): The language to translate the text to.
        Defaults to "auto".
        source_lang (str, optional): The language of the text to translate.
        Defaults to "auto".

    Returns:
        str: The translated text.
    """
    response = get_default_resultset()
    # try:
    #     log_debug("Init translator...")
    #     translator = Translator()
    # except Exception as err:
    #     response['error'] = True
    #     response['error_message'] = f"ERROR [GLT-010]: {str(err)}"

    if not response['error']:
        # Translate text to target language
        try:
            if DEBUG:
                log_debug(f"lang_translate() to: {target_lang}")
            # output_text = translator.translate(
            output_text = translate(
                text=input_text,
                dest=target_lang,
                src=source_lang
            )
            if output_text["error"]:
                response = dict(output_text)
            else:
                response["output_text"] = output_text["text"]
        except Exception as err:
            response['error'] = True
            response['error_message'] = f"ERROR [GLT-020]: {str(err)}"
        if DEBUG:
            log_debug(
                "lang_translate() | " +
                ("ERROR:" if response["error"] else "Text translated:") +
                f' {response}'
            )
    return response
