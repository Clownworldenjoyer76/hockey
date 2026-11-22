#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from collections.abc import Collection, Mapping, MutableMapping
from pathlib import Path
from typing import Any


TeamIdentity = dict[str, str]


def strip_record(value: str) -> str:
    return re.sub(
        r"\s*\(\d+[-–]\d+[-–]?\d*\)\s*$",
        "",
        str(value),
    ).strip()


def normalize_alias_key(value: str) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value).strip(),
    )
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower().replace("&", " and ")
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_team_alias_key(value: str) -> str:
    return normalize_alias_key(
        strip_record(value)
    )


def parse_nhl_team_map_row(
    row: Mapping[str, Any],
    row_number: int,
    map_path: Path,
    *,
    supported_sources: Collection[str] | None = None,
) -> tuple[str, str, str, str, TeamIdentity] | None:
    if str(row.get("league", "")).strip().lower() != "nhl":
        return None

    source = str(row.get("source", "")).strip().lower()
    alias = str(row.get("alias", "")).strip()
    canonical = str(row.get("canonical_team", "")).strip()
    team_id = str(row.get("nhl_team_id", "")).strip()
    abbrev = str(row.get("nhl_abbrev", "")).strip().upper()

    if not source or not alias or not canonical:
        return None

    if (
        supported_sources is not None
        and source not in supported_sources
    ):
        raise ValueError(
            f"{map_path} row {row_number} has unsupported "
            f"source={source!r}"
        )

    if canonical != "TBD":
        if not team_id or not team_id.isdigit():
            raise ValueError(
                f"{map_path} row {row_number} has invalid "
                f"nhl_team_id={team_id!r}"
            )

        if not re.fullmatch(r"[A-Z]{3}", abbrev):
            raise ValueError(
                f"{map_path} row {row_number} has invalid "
                f"nhl_abbrev={abbrev!r}"
            )

    identity = {
        "canonical_team": canonical,
        "nhl_team_id": team_id,
        "nhl_abbrev": abbrev,
    }

    return (
        source,
        alias,
        team_id,
        abbrev,
        identity,
    )


def register_team_identity(
    by_id: MutableMapping[str, TeamIdentity],
    team_id: str,
    identity: TeamIdentity,
    map_path: Path,
) -> None:
    if not team_id:
        return

    prior = by_id.get(team_id)
    if prior is not None and prior != identity:
        raise ValueError(
            f"{map_path} has conflicting identity for "
            f"nhl_team_id={team_id}: {prior} != {identity}"
        )

    by_id[team_id] = identity


def load_nhl_team_map(
    map_path: Path,
    *,
    supported_sources: Collection[str] | None = None,
    required_sources: Collection[str] = (),
    expected_team_count: int | None = None,
) -> dict[str, Any]:
    """Read team_map_nhl.csv and check it. This is the only loader.

    Returns a dict with:
      by_source: {source: {alias_key: team identity}}
      by_id:     {nhl_team_id: team identity}
      by_abbrev: {nhl_abbrev: team identity}

    It stops with an error if the file or a row is bad, if two rows disagree
    about one team ID, abbreviation or alias, if a source in
    required_sources has no rows, or if expected_team_count is set and the
    number of team IDs is different.
    """
    if not map_path.exists():
        raise FileNotFoundError(
            f"Missing team mapping file: {map_path}"
        )

    by_source: dict[str, dict[str, TeamIdentity]] = defaultdict(dict)
    by_id: dict[str, TeamIdentity] = {}
    by_abbrev: dict[str, TeamIdentity] = {}

    with map_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "league",
            "source",
            "alias",
            "canonical_team",
            "nhl_team_id",
            "nhl_abbrev",
        }
        missing = sorted(required - set(reader.fieldnames or []))

        if missing:
            raise ValueError(
                f"{map_path} missing required columns: {missing}"
            )

        for row_number, row in enumerate(reader, start=2):
            parsed = parse_nhl_team_map_row(
                row,
                row_number,
                map_path,
                supported_sources=supported_sources,
            )
            if parsed is None:
                continue

            source, alias, team_id, abbrev, identity = parsed

            register_team_identity(
                by_id,
                team_id,
                identity,
                map_path,
            )

            if abbrev:
                prior_abbrev = by_abbrev.get(abbrev)
                if (
                    prior_abbrev is not None
                    and prior_abbrev != identity
                ):
                    raise ValueError(
                        f"{map_path} has conflicting identity for "
                        f"nhl_abbrev={abbrev}: "
                        f"{prior_abbrev} != {identity}"
                    )
                by_abbrev[abbrev] = identity

            key = normalize_team_alias_key(alias)
            prior_alias = by_source[source].get(key)

            if (
                prior_alias is not None
                and prior_alias != identity
            ):
                raise ValueError(
                    f"{map_path} has conflicting mapping for "
                    f"source={source} alias={alias!r}: "
                    f"{prior_alias} != {identity}"
                )

            by_source[source][key] = identity

    for source_name in required_sources:
        if not by_source.get(source_name):
            raise ValueError(
                f"No {source_name} mappings loaded from {map_path}"
            )

    if (
        expected_team_count is not None
        and len(by_id) != expected_team_count
    ):
        raise ValueError(
            f"Expected {expected_team_count} stable NHL team IDs in "
            f"{map_path}; found {len(by_id)}"
        )

    return {
        "by_source": dict(by_source),
        "by_id": by_id,
        "by_abbrev": by_abbrev,
    }


def alias_map_for_source(
    team_map: Mapping[str, Any],
    source: str,
    map_path: Path,
) -> dict[str, TeamIdentity]:
    """One alias -> team lookup for a source plus the shared and official rows.

    Used by the scripts that fix team names in files. It stops with an error
    if the same alias points at two different teams, or if nothing is found.
    """
    merged: dict[str, TeamIdentity] = {}

    for candidate_source in (source, "shared", "official_nhl"):
        for key, identity in team_map["by_source"].get(
            candidate_source,
            {},
        ).items():
            prior = merged.get(key)

            if prior is not None and prior != identity:
                raise ValueError(
                    f"{map_path} has conflicting {source} mapping for "
                    f"alias={key!r}: {prior} != {identity}"
                )

            merged[key] = identity

    if not merged:
        raise ValueError(
            f"No NHL mappings loaded for source={source} "
            f"from {map_path}"
        )

    return merged
