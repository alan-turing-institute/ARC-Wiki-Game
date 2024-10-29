from __future__ import annotations

import jieba
import stopwordsiso
from gensim.parsing.preprocessing import (
    preprocess_string,
    stem_text,
    strip_multiple_whitespaces,
    strip_numeric,
    strip_punctuation,
    strip_short,
    strip_tags,
)


def _pre_process_universal(text: str, lang: str) -> list[str]:
    """
    Takes text and a language code as input and applies some functions to process
    the text. These functions are expected to be applied to all languages. The
    stopwords will be removed for the given language code.

    Parameters
    ----------
    text: str
        The text that is to be processed
    lang: str
        Expected to be a two-letter language code.

    Returns
    -------
    list[str]:
        The list of processed words from the text

    Raises
    ------
    ValueError:
        If the language code provided has no stop words.
    """
    if stopwordsiso.has_lang(lang):
        custom_filters = [
            lambda x: x.lower(),
            strip_tags,
            strip_punctuation,
            strip_multiple_whitespaces,
            strip_numeric,
        ]
        processed_text = preprocess_string(text, custom_filters)
        sw = stopwordsiso.stopwords(lang)

        return [word for word in processed_text if word not in sw]
    raise ValueError("No stopwords found for lanugage code " + lang)


def pre_process_with_stemming(text: str, lang: str) -> list[str]:
    """
    Function to process text including stemming (not done for Chinese text)

    Parameters
    ----------
    text: str
        The text that is to be processed
    lang: str
        Expected to be a two-letter language code.

    Returns
    -------
    list[str]:
        The list of processed words from the text
    """
    processed_text = _pre_process_universal(text, lang)
    custom_filters = [lambda x: x.lower(), strip_short, stem_text]
    return preprocess_string(" ".join(processed_text), custom_filters)


def pre_process_chinese(text: str, lang: str) -> list[str]:
    """
    Currently this function is only expected to process simplified Chinese text
    (hence the name of the function). This avoids stemming the text and also
    does not remove 'short' characters.

    Parameters
    ----------
    text: str
        The text that is to be processed
    lang: str
        Expected to be a two-letter language code.

    Returns
    -------
    list[str]:
        The list of processed words from the text
    """
    cut_text = jieba.lcut(text)
    return _pre_process_universal(" ".join(cut_text), lang)
