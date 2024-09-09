from __future__ import annotations

from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, SimilarityFunction


def get_model_from_name(model_name: str) -> SentenceTransformer:
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
