from __future__ import annotations

from typing import Callable

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from huggingface_hub import HfApi
from numpy import arange
from numpy.random import choice, seed
from sentence_transformers import SentenceTransformer, SimilarityFunction

from agents.agent_models.tools import AgentToolsBase


def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    Returns a SentenceTransformer object given a model name.

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
) -> tuple[TfidfModel, list[str], Dictionary]:
    """
    Builds and returns a TF-IDF model.

    Parameters
    ----------
    agent_tools: AgentToolsBase
        The Agent Tools object to use to create the dataset for the corpus
    manual_seed: int
        The integer value to use to set the first random seed. As the process selects
        articles at random from the dataset, this will ensure that the same articles
        can be selected each time the code is run.
    max_articles: int
        The maximum number of articles that can be selected
    preprocess_func: Callable[[str, str], list[str]]
        A function that takes text and an iso code and returns the processed text
    lang_iso_code: str
        The iso code to use for pre-processing the data (identifies the stop words)

    Returns
    -------
    TfidfModel
        The model built using the corpus that was create in this function
    list[str]
        A list of the vocabulary words used in the model
    Dictionary
        A Gensim dictionary object

    """
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
