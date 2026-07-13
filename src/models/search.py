"""
Search utilities for TuneMatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz, process, utils


@dataclass(frozen=True)
class SearchChoice:
    """
    One searchable track entry.
    """

    index: int
    label: str
    track_name: str
    artists: str
    search_text: str


def _normalize_text(value: object) -> str:
    """
    Convert values to safe searchable text.
    """

    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def build_search_index(dataframe: pd.DataFrame) -> list[SearchChoice]:
    """
    Build normalized search records for the catalog.
    """

    choices: list[SearchChoice] = []

    for index, row in dataframe.reset_index(drop=True).iterrows():
        track_name = _normalize_text(row.get("track_name"))
        artists = _normalize_text(row.get("artists"))
        genre = _normalize_text(row.get("track_genre"))

        label = f"{track_name} - {artists}" if artists else track_name

        # Include multiple fields so artist search and typo-tolerant matching
        # both benefit from a broader context string.
        search_text = " | ".join(
            part
            for part in (
                track_name,
                artists,
                label,
                genre,
            )
            if part
        )

        choices.append(
            SearchChoice(
                index=index,
                label=label,
                track_name=track_name,
                artists=artists,
                search_text=search_text,
            )
        )

    return choices


def build_search_choices(dataframe: pd.DataFrame) -> list[str]:
    """
    Backward-compatible helper returning display labels.
    """

    return [
        choice.label
        for choice in build_search_index(dataframe)
    ]


def _coerce_search_choice(
    choice: SearchChoice | str,
    position: int,
) -> SearchChoice:
    """
    Accept either the new dataclass format or legacy cached string entries.
    """

    if isinstance(choice, SearchChoice):
        return choice

    label = _normalize_text(choice)
    track_name, separator, artists = label.partition(" - ")
    if not separator:
        track_name = label
        artists = ""

    search_text = " | ".join(
        part
        for part in (
            track_name,
            artists,
            label,
        )
        if part
    )

    return SearchChoice(
        index=position,
        label=label,
        track_name=track_name,
        artists=artists,
        search_text=search_text,
    )


def _extract_search_text(
    value: SearchChoice | str,
) -> str:
    """
    Provide a normalized string for RapidFuzz processors.
    """

    if isinstance(value, SearchChoice):
        return value.search_text

    return _normalize_text(value)


def _extract_search_label(
    value: SearchChoice | str,
) -> str:
    """
    Provide a display label for RapidFuzz processors.
    """

    if isinstance(value, SearchChoice):
        return value.label

    return _normalize_text(value)


def _dedupe_matches(
    matches: Iterable[tuple[SearchChoice, float, int]],
) -> list[dict]:
    """
    Remove duplicate matches by track index while preserving rank order.
    """

    deduped: list[dict] = []
    seen: set[int] = set()

    for choice, score, _ in matches:
        if choice.index in seen:
            continue

        seen.add(choice.index)
        deduped.append(
            {
                "index": choice.index,
                "label": choice.label,
                "track_name": choice.track_name,
                "artists": choice.artists,
                "score": round(float(score), 2),
            }
        )

    return deduped


def intelligent_search(
    query: str,
    search_index: list[SearchChoice | str],
    limit: int = 10,
    score_cutoff: int = 45,
) -> list[dict]:
    """
    Search by partial song title, artist, or typo-tolerant fuzzy matching.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        return []

    normalized_search_index = [
        _coerce_search_choice(choice, position)
        for position, choice in enumerate(search_index)
    ]

    direct_substring_matches: list[tuple[SearchChoice, float, int]] = []
    lowered_query = cleaned_query.casefold()

    for position, choice in enumerate(normalized_search_index):
        searchable = choice.search_text.casefold()
        if lowered_query in searchable:
            direct_substring_matches.append(
                (
                    choice,
                    100.0,
                    position,
                )
            )

    fuzzy_matches = process.extract(
        cleaned_query,
        normalized_search_index,
        processor=lambda value: utils.default_process(
            _extract_search_text(value)
        ),
        scorer=fuzz.WRatio,
        limit=limit * 3,
        score_cutoff=score_cutoff,
    )

    startswith_matches = process.extract(
        cleaned_query,
        normalized_search_index,
        processor=lambda value: utils.default_process(
            _extract_search_label(value)
        ),
        scorer=fuzz.partial_ratio,
        limit=limit * 3,
        score_cutoff=score_cutoff,
    )

    combined_matches = (
        direct_substring_matches
        + fuzzy_matches
        + startswith_matches
    )

    return _dedupe_matches(
        combined_matches
    )[:limit]


def fuzzy_search(
    query: str,
    choices: list[str],
    limit: int = 10,
    score_cutoff: int = 50,
):
    """
    Legacy helper retained for compatibility.
    """

    return process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff,
    )
