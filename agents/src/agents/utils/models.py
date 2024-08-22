from __future__ import annotations

from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, SimilarityFunction


def get_model_from_name(model_name: str) -> SentenceTransformer:
    """
    Parameters
    ----------
    model_name: str
        The name of the model in the 'sentence-transformer' HuggingFace library
        Can either be in the form 'sentence-transformers/model-name' or just
        'model-name' without the prefix.

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
    # Might want to remove the restriction on author - this will restrict the
    # list of valid models to only the original sentence transformer models
    models = hf_api.list_models(
        library="sentence-transformers", author="sentence-transformers"
    )
    model_ids = [model.id for model in list(models)]
    if "sentence-transformers/" + model_name in model_ids or model_name in model_ids:
        return SentenceTransformer(
            "sentence-transformers/" + model_name,
            similarity_fn_name=SimilarityFunction.COSINE,
        )
    raise ValueError("The model name " + model_name + " is not a valid model name.")
