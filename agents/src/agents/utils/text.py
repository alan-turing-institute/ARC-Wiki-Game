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


def _pre_process_universal(text: str, lang: str):
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


def pre_process_with_stemming(text: str, lang: str):
    if stopwordsiso.has_lang(lang):
        processed_text = _pre_process_universal(text, lang)
        custom_filters = [lambda x: x.lower(), strip_short, stem_text]
        return preprocess_string(" ".join(processed_text), custom_filters)

    raise ValueError("No stopwords found for lanugage code " + lang)


def pre_process_chinese(text: str, lang: str):
    if stopwordsiso.has_lang(lang):
        cut_text = jieba.lcut(text)
        return _pre_process_universal(" ".join(cut_text), lang)
    raise ValueError("No stopwords found for language code " + lang)
