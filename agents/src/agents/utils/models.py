from __future__ import annotations

from typing import Callable

import gensim.downloader as gensim_api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from huggingface_hub import HfApi
from numpy import arange
from numpy.random import choice, seed
from sentence_transformers import SentenceTransformer, SimilarityFunction

from agents.agent_models.tools import AgentToolsBase


def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    Parameters
    ----------
    model_name: str
        The name of the model in the 'sentence-transformers' HuggingFace library.
        If the model is not authored by 'sentence-transformers', then the name
        must be provided in the format 'author-name/model-name'. However, if the
        author is 'sentence-transformers', then just 'model-name' can be provided.

    Returns
    -------
    SentenceTransformer
        The HuggingFace model associated with the given name

    Raises
    ------
    ValueError
        This is thrown if the given model_name argument does not match one in the
        'sentence-transformers' library.
    """
    # Check if the model_name is one that exists in the sentence transformers library
    hf_api = HfApi()
    # Get all sentence-transformer models
    models = hf_api.list_models(library="sentence-transformers")
    model_ids = [model.id for model in list(models)]
    if "sentence-transformers/" + model_name in model_ids:
        model_name = "sentence-transformers/" + model_name
    if model_name in model_ids:
        return SentenceTransformer(
            model_name,
            similarity_fn_name=SimilarityFunction.COSINE,
        )
    raise ValueError("The model name " + model_name + " is not a valid model name.")


def get_tfidf_model(
    agent_tools: AgentToolsBase,
    manual_seed: int,
    max_articles: int,
    preprocess_func: Callable[[str, str], list[str]],
    lang_iso_code: str,
):
    seed(manual_seed)
    num_articles = agent_tools.matidx_ary.max()
    if num_articles < max_articles:
        max_articles = num_articles
    article_ids = choice(arange(1, num_articles + 1), max_articles, replace=False)

    dct = Dictionary()
    corpus = []

    for id in article_ids:
        article_summary = agent_tools.get_article_summary(id)
        if article_summary is not None:
            processed_text = preprocess_func(article_summary, lang_iso_code)
            dct.add_documents([processed_text])
            corpus.append(dct.doc2bow(processed_text))

    vocab = [dct[i] for i in range(len(dct))]
    model = TfidfModel(corpus)
    return model, vocab, dct


def get_tfidf_model_wiki(
    dataset_name: str, preprocess_func: Callable[[str], list[str]]
) -> tuple[TfidfModel, list[str], Dictionary]:
    corpora = list(gensim_api.info()["corpora"])
    if dataset_name in corpora and dataset_name == "wiki-english-20171001":
        dataset = gensim_api.load(dataset_name)
        dct = Dictionary()
        max_lines = 10000
        for id, line in enumerate(dataset):
            texts = line["section_texts"]
            processed_texts = [preprocess_func(text) for text in texts]
            dct.add_documents(processed_texts)
            if id > max_lines:
                break
        corpus = []
        for id, line in enumerate(dataset):
            texts = line["section_texts"]
            processed_texts = [preprocess_func(text) for text in texts]
            all_words = [word for words in processed_texts for word in words]
            corpus.append(dct.doc2bow(all_words))
            if id > max_lines:
                break
        # corpus = [dct.doc2bow(line) for line in dataset]
        vocab = [dct[i] for i in range(len(dct))]
        model = TfidfModel(corpus)
        return model, vocab, dct
    raise ValueError(
        "The dataset name " + dataset_name + " does not exist in the Gensim corpora."
    )
