"""
Search utilities for TuneMatch.
"""

from __future__ import annotations

from rapidfuzz import process, fuzz


def build_search_choices(dataframe):
    """
    Build searchable strings.

    Returns
    -------
    list[str]
    """

    return [
        f"{row.track_name} - {row.artists}"
        for row in dataframe.itertuples()
    ]


def fuzzy_search(
    query: str,
    choices: list[str],
    limit: int = 10,
    score_cutoff: int = 50,
):
    """
    Perform fuzzy search.

    Returns
    -------
    list[(choice, score, index)]
    """

    return process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff,
    )