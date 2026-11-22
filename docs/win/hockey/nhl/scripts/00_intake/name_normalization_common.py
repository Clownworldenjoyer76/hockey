#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import traceback
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noinspection PyPep8
from team_map_common import (
    alias_map_for_source,
    load_nhl_team_map,
    normalize_alias_key,
)


def make_logger(
    log_file: Path,
    run_name: str,
) -> Callable[[str], None]:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(
            f"=== {run_name} RUN "
            f"{datetime.now(timezone.utc).isoformat()} ===\n"
        )

    def log(message: str) -> None:
        with log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"{datetime.now(timezone.utc).isoformat()} | "
                f"{message}\n"
            )

    return log


def load_team_map(
    map_file: Path,
    source: str,
    log: Callable[[str], None],
) -> dict[str, dict[str, str]]:
    if not map_file.exists():
        raise FileNotFoundError(
            f"team_map_nhl.csv not found: {map_file}"
        )

    mapping = alias_map_for_source(
        load_nhl_team_map(map_file),
        source,
        map_file,
    )

    stable_ids = {
        identity["nhl_team_id"]
        for identity in mapping.values()
        if identity["nhl_team_id"]
    }

    log(
        f"Team map loaded: {len(mapping)} aliases | "
        f"source={source} | "
        f"stable_ids={len(stable_ids)}"
    )

    return mapping


def normalize_files(
    target_files: Iterable[Path],
    team_map: dict[str, dict[str, str]],
    no_map_file: Path,
    log: Callable[[str], None],
) -> None:
    files = list(target_files)
    log(f"Files to process: {len(files)}")

    unmapped: set[str] = set()
    files_processed = 0
    rows_processed = 0
    names_normalized = 0

    # noinspection PyBroadException
    try:
        for csv_file in files:
            # noinspection PyBroadException
            try:
                files_processed += 1
                updated_rows: list[dict[str, str]] = []
                modified = False

                with csv_file.open(
                    newline="",
                    encoding="utf-8-sig",
                ) as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = reader.fieldnames or []

                    for row in reader:
                        rows_processed += 1

                        for column in (
                            "home_team",
                            "away_team",
                        ):
                            team = str(
                                row.get(column, "")
                            ).strip()

                            if not team:
                                continue

                            identity = team_map.get(
                                normalize_alias_key(team)
                            )

                            if identity:
                                canonical = identity[
                                    "canonical_team"
                                ]

                                if (
                                    row.get(column)
                                    != canonical
                                ):
                                    row[column] = canonical
                                    modified = True
                                    names_normalized += 1
                            else:
                                unmapped.add(team)

                        updated_rows.append(row)

                if modified and fieldnames:
                    with csv_file.open(
                        "w",
                        newline="",
                        encoding="utf-8-sig",
                    ) as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=fieldnames,
                        )
                        writer.writeheader()
                        writer.writerows(
                            updated_rows
                        )

                    log(f"UPDATED: {csv_file}")

            except Exception as exc:
                log(
                    f"ERROR processing {csv_file}: {exc}\n"
                    f"{traceback.format_exc()}"
                )

        no_map_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with no_map_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["league", "team"]
            )

            for team in sorted(unmapped):
                writer.writerow(
                    ["nhl", team]
                )

        log("--- SUMMARY ---")
        log(
            f"Files processed: {files_processed}"
        )
        log(
            f"Rows processed: {rows_processed}"
        )
        log(
            f"Names normalized: {names_normalized}"
        )
        log(
            f"Unmapped teams: {len(unmapped)}"
        )
        log(
            f"No-map output: {no_map_file}"
        )
        log("STATUS: SUCCESS")

    except Exception as exc:
        log(
            f"FATAL ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )
        log("STATUS: FAILED")
        raise


def run_name_normalization(
    *,
    run_name: str,
    source: str,
    map_file: Path,
    no_map_file: Path,
    log_file: Path,
    target_files: Iterable[Path],
) -> None:
    log = make_logger(
        log_file,
        run_name,
    )
    team_map = load_team_map(
        map_file,
        source,
        log,
    )
    normalize_files(
        target_files,
        team_map,
        no_map_file,
        log,
    )
