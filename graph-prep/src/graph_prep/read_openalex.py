from __future__ import annotations

import re


def extract_json(
    data: dict, extract_id: re.Pattern, strip_escapes: re.Pattern
) -> str | None:
    """
    This produces a SQL command to insert the data into the database which has been
    extracted from the json data. If the data is not valid, None is returned.

    This function also converts the inverted index of the abstract into a string.
    It also converts the references into a list of integers.

    Args:
        data: dict: The json data to be processed.
        extract_id: re.Pattern: The regex pattern to extract the id from the data.
        strip_escapes: re.Pattern: The regex pattern to remove escape characters.

    Returns:
        str | None: The SQL command to insert the data into the database.
    """
    # Extract id
    id_r = extract_id.search(data["id"])
    if id_r is None:
        return None
    id = int(id_r.group(1))
    # Only want to retain works of type 'article'
    if data["type"] != "article":
        return None
    # Extract title
    title = data["title"]
    if data["title"] is None:
        return None  # Skip works without title
    title = strip_escapes.sub("", data["title"])
    # Extract doi
    doi = strip_escapes.sub("", data["doi"]) if data["doi"] is not None else None
    # Extract field
    if (
        "primary_topic" in data
        and data["primary_topic"] is not None
        and data["primary_topic"]["field"] is not None
    ):
        field = strip_escapes.sub("", data["primary_topic"]["field"]["display_name"])
    else:
        field = None
    if field not in [
        "Medicine",
        "Arts and Humanities",
        "Biochemistry, Genetics and Molecular Biology",
        "Physics and Astronomy",
        "Social Sciences",
    ]:
        return None
    # Extract language
    if data["language"] is not None:
        language = strip_escapes.sub("", data["language"])
    else:
        return None
    # Extract citations
    cites = int(data["summary_stats"]["cited_by_count"])
    # Extract abstract
    if data["abstract_inverted_index"] is not None:
        w_list = [
            (w, p) for w, pl in data["abstract_inverted_index"].items() for p in pl
        ]
        w_list = sorted(w_list, key=lambda x: x[1])
        abstract = " ".join([w[0] for w in w_list])
        abstract = strip_escapes.sub("", abstract)
    else:
        return None
    # Extract referenced works
    refs = []
    for ref in data["referenced_works"]:
        ref_id = extract_id.search(ref)
        if ref_id is not None:
            refs.append(int(ref_id.group(1)))
    # Insert data into database
    return (
        f"    ({id}, '{title}', {cites}, '{doi}', "
        f"'{field}', '{language}', '{abstract}', {refs})"
    )
