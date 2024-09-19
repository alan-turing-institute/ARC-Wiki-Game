from __future__ import annotations

import gensim.downloader as gensim_api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, SimilarityFunction


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


def get_tfidf_model(dataset_name: str) -> tuple[TfidfModel, list[str], Dictionary]:
    corpora = list(gensim_api.info()["corpora"])
    if dataset_name in corpora:
        dataset = gensim_api.load(dataset_name)
        dct = Dictionary(dataset)
        corpus = [dct.doc2bow(line) for line in dataset]
        vocab = [dct[i] for i in range(len(dct))]
        model = TfidfModel(corpus)
        return model, vocab, dct
    raise ValueError(
        "The dataset name " + dataset_name + " does not exist in the Gensim corpora."
    )
