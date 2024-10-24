from __future__ import annotations

import re

from graph_prep.read_openalex import extract_json


def test_extract_json():
    data = {
        "id": "https://openalex.org/W2093406744",
        "doi": "https://doi.org/10.1016/j.neuroimage.2009.03.041",
        "title": "The processing and representation of fractions within the brain",
        "publication_year": 2009,
        "publication_date": "2009-08-01",
        "language": "en",
        "type": "article",
        "primary_topic": {"field": {"display_name": "Medicine"}},
        "cited_by_count": 75,
        "summary_stats": {"cited_by_count": 75, "2yr_cited_by_count": 2},
        "referenced_works": [
            "https://openalex.org/W280645764",
            "https://openalex.org/W1965333862",
            "https://openalex.org/W1968525257",
            "https://openalex.org/W1968622089",
            "https://openalex.org/W1970706444",
            "https://openalex.org/W1971139377",
            "https://openalex.org/W1974173422",
            "https://openalex.org/W1975140269",
            "https://openalex.org/W1983405361",
            "https://openalex.org/W2001569942",
            "https://openalex.org/W2004984306",
            "https://openalex.org/W2017345307",
        ],
        "referenced_works_count": 45,
        "abstract_inverted_index": {
            "This": [0],
            "is": [1],
            "a": [2],
            "abstract.": [3, 6],
            "An": [4],
            "excellent": [5],
            "to": [7],
            "be": [8],
            "enjoyed.": [9],
        },
    }

    re_strip_escapes = re.compile(r"[\\']")
    re_extract_id = re.compile(r"W(\d+)$")
    to_insert = extract_json(data, re_extract_id, re_strip_escapes)

    assert isinstance(to_insert, str)
    assert to_insert == (
        "    (2093406744, "
        "'The processing and representation of fractions within the brain', "
        "75, "
        "'https://doi.org/10.1016/j.neuroimage.2009.03.041', "
        "'Medicine', "
        "'en', "
        "'This is a abstract. An excellent abstract. to be enjoyed.', "
        "[280645764, 1965333862, 1968525257, 1968622089, 1970706444, 1971139377, "
        "1974173422, 1975140269, 1983405361, 2001569942, 2004984306, 2017345307])"
    )
