#!/usr/bin/env python3
"""Build NHL 2009-10 through 2025-26 player-game outcome records.

INPUT (READ ONLY):
  G:/hockey/docs/win/hockey/nhl/research/sdv_history/{schedules,player_boxscores}/season_YYYY.parquet
OUTPUT (WRITES ONLY):
  C:/Users/Mat/Downloads/nhl_historical_player_game_targets_2009_2025.csv
  C:/Users/Mat/Downloads/nhl_historical_player_game_targets_2009_2025_summary.csv
  C:/Users/Mat/Downloads/nhl_historical_player_game_targets_2009_2025_manifest.json

- One row per official (game_id, player_id) from SportsDataverse player_boxscores.
- Only regular season (02/R) and playoffs (03/P); three user-approved game exceptions.
- Use TOI strictly for played / did-not-play / unknown classification.
- Zero/unknown TOI: keep identity, blank OUTCOME columns, never treat as a zero.
- The seventh validated prop fields: five skater + two goalie; plus a distinct
  power_play_goals field when observed. No power-play assists or points inferred.
- Does not calculate sportsbook settlement eligibility or pregame features.
- Schedule game_date is the training date; player-boxscore dates are also kept.
- Every date difference is reported; >1 day or multiple source dates flags the output for review.
- No repository modifications, network calls, or package installation.
"""

from __future__ import annotations

BUILDER_VERSION = "2026-11-17-v4"

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

# SportsDataverse Parquet files may contain R Arrow extension metadata.
# Set the storage behavior BEFORE Polars is imported.
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

EXCLUDED = {
    "2009021019": "2009-10: missing complete official game from local SDV history; approved exclusion",
    "2010020868": "2010-11: missing complete official game from local SDV history; approved exclusion",
    "2019030001": "2019-20: official game boxscore uncomparable; approved exclusion",
}
SKATER_FIELDS = ("shots_on_goal", "points", "assists", "goals", "blocked_shots")
GOALIE_FIELDS = ("saves", "goals_against")
OPTIONAL_FIELD = "power_play_goals"
# Additional observed columns in every combined player-boxscore season (not prop targets).
COMMON_ADDITIONAL_STATS = ("pim",)
SKATER_ADDITIONAL_STATS = (
    "plus_minus", "hits", "faceoff_winning_pctg", "shifts", "giveaways", "takeaways",
)
GOALIE_ADDITIONAL_STATS = (
    "even_strength_shots_against", "power_play_shots_against",
    "shorthanded_shots_against", "save_shots_against", "save_pctg",
    "even_strength_goals_against", "power_play_goals_against",
    "shorthanded_goals_against", "shots_against", "decision",
)
ALL_ADDITIONAL_STATS = (*COMMON_ADDITIONAL_STATS, *SKATER_ADDITIONAL_STATS, *GOALIE_ADDITIONAL_STATS)
OUTPUT_FIELDS = (
    "season", "game_id", "game_date", "source_player_game_date", "source_date_offset_days", "source_date_match_status", "game_type",
    "player_id", "player_name", "sweater_number", "team_id", "team", "opponent_team", "home_away",
    "position", "position_group", "toi", "participation_status", "participated",
    "training_target_available", "goalie_starter_raw",
    *SKATER_FIELDS, OPTIONAL_FIELD, *GOALIE_FIELDS, *ALL_ADDITIONAL_STATS,
)
BASE_NAME = "nhl_historical_player_game_targets_2009_2025"
DATE_AUDIT_FIELDS = ("season", "game_id", "schedule_game_date", "source_player_game_date", "offset_days", "affected_player_rows", "status")
REQUIRED_SCHEDULE = ("game_id", "game_date", "game_type", "home_team_abbr", "away_team_abbr")
REQUIRED_BOXSCORES = (
    "game_id", "player_id", "game_date", "team_id", "team_abbrev", "player_name", "sweater_number",
    "home_away", "position", "toi", "starter", *SKATER_FIELDS, OPTIONAL_FIELD, *GOALIE_FIELDS,
    *ALL_ADDITIONAL_STATS,
)


def fail(message: str) -> None:
    raise ValueError(message)


def integer_id(raw: object, label: str, *, digits: int | None = None) -> str:
    if raw is None:
        fail(f"{label}: missing ID")
    s = str(raw).strip()
    if not s.isdigit() or int(s) <= 0 or (digits is not None and len(s) != digits):
        fail(f"{label}: invalid ID {s!r}")
    return s


def iso_date(raw: object, label: str) -> str:
    if raw is None:
        fail(f"{label}: missing date")
    if isinstance(raw, (date, datetime)):
        return raw.strftime("%Y-%m-%d")
    s = str(raw).strip().replace("_", "-")[:10]
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError as exc:
        raise ValueError(f"{label}: invalid date {raw!r}") from exc


def game_type(raw: object, label: str) -> int:
    """Decode the SportsDataverse schedule's game-type values.

    Its historical schedules use strings: PR=preseason, R=regular,
    P=playoffs, A=All-Star. Other sources may use NHL numeric codes.
    The 10-digit official NHL game ID is independently cross-checked.
    """
    if raw is None:
        fail(f"{label}: missing game_type")
    value = str(raw).strip().upper()
    labels = {
        "PR": 1, "PRESEASON": 1,
        "R": 2, "REGULAR": 2, "REGULAR SEASON": 2,
        "P": 3, "PLAYOFF": 3, "PLAYOFFS": 3,
        "A": 4, "ALL-STAR": 4, "ALL STAR": 4,
    }
    if value in labels:
        return labels[value]
    # Numeric forms 1/01, 2/02, 3/03, 4/04.
    if value.isdigit() and int(value) in (1, 2, 3, 4):
        return int(value)
    fail(f"{label}: unsupported schedule game_type {raw!r}")


def official_game_type(gid: str, label: str) -> int:
    n = int(gid[4:6])
    if n not in (1, 2, 3, 4):
        fail(f"{label}: unsupported NHL game type encoded in game_id {gid}")
    return n


def toi_status(raw: object, label: str) -> str:
    """Positive TOI indicates play, zero TOI indicates no verified playing time.

    Numeric TOI, when present, is interpreted only by sign (not unit).
    """
    if raw is None or str(raw).strip() == "":
        return "UNKNOWN_TOI"
    s = str(raw).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts):
            fail(f"{label}: unsupported TOI {s!r}")
        vals = [int(x) for x in parts]
        if vals[-1] >= 60 or (len(vals) == 3 and vals[-2] >= 60):
            fail(f"{label}: impossible TOI {s!r}")
        seconds = vals[0] * 60 + vals[1] if len(vals) == 2 else vals[0] * 3600 + vals[1] * 60 + vals[2]
        return "PLAYED" if seconds > 0 else "ZERO_TOI"
    try:
        value = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"{label}: unsupported TOI {s!r}") from exc
    if not value.is_finite() or value < 0:
        fail(f"{label}: invalid TOI {s!r}")
    return "PLAYED" if value > 0 else "ZERO_TOI"


def integer_stat(raw: object, label: str, *, allow_missing: bool = False) -> int | None:
    if raw is None or str(raw).strip() == "":
        if allow_missing:
            return None
        fail(f"{label}: missing statistic")
    try:
        n = Decimal(str(raw).strip())
    except InvalidOperation as exc:
        raise ValueError(f"{label}: nonnumeric statistic {raw!r}") from exc
    if not n.is_finite() or n < 0 or n != n.to_integral_value():
        fail(f"{label}: invalid count statistic {raw!r}")
    return int(n)


def source_stat(raw: object) -> object | None:
    """Preserve additional observed statistics without converting missing to zero."""
    if raw is None:
        return None
    if isinstance(raw, float) and not math.isfinite(raw):
        return None
    if str(raw).strip() == "":
        return None
    return raw


def parse_schedule(raw_rows: list[dict], season: int) -> dict[str, dict]:
    games: dict[str, dict] = {}
    seen: set[str] = set()
    for ix, row in enumerate(raw_rows, 1):
        label = f"schedule season_{season}.parquet row={ix}"
        gid = integer_id(row["game_id"], label, digits=10)
        if gid in seen:
            fail(f"{label}: duplicate schedule game ID {gid}")
        seen.add(gid)
        if int(gid[:4]) != season:
            fail(f"{label}: game_id {gid} belongs to another season")
        t = game_type(row["game_type"], label)
        if official_game_type(gid, label) != t:
            fail(f"{label}: game_id {gid} type disagrees with source game_type={row['game_type']!r}")
        if t not in (2, 3) or gid in EXCLUDED:
            continue
        dt = iso_date(row["game_date"], label)
        home = str(row["home_team_abbr"] or "").strip().upper()
        away = str(row["away_team_abbr"] or "").strip().upper()
        if not home or not away or home == away:
            fail(f"{label}: missing/identical home-away teams for {gid}")
        games[gid] = {"game_date": dt, "game_type": t, "home": home, "away": away}
    if not games:
        fail(f"schedule season_{season}.parquet: no regular/playoff games")
    return games


def normalize_orientation(raw: object, label: str) -> str:
    s = str(raw or "").strip().upper()
    if s in ("HOME", "H"):
        return "HOME"
    if s in ("AWAY", "A"):
        return "AWAY"
    fail(f"{label}: unknown home_away value {raw!r}")


def make_record(row: dict, season: int, schedule: dict[str, dict], source_row: int) -> dict | None:
    label = f"player_boxscores season_{season}.parquet row={source_row}"
    gid = integer_id(row["game_id"], label, digits=10)
    if gid in EXCLUDED:
        return None
    if int(gid[:4]) != season:
        fail(f"{label}: wrong game season for {gid}")
    if official_game_type(gid, label) not in (2, 3):
        return None  # preseason and All-Star excluded
    match = schedule.get(gid)
    if match is None:
        fail(f"{label}: NHL regular/playoff boxscore {gid} absent from this season schedule")
    pid = integer_id(row["player_id"], label)
    # SDV schedule and player boxscore dates are not always identical.
    # Use the audited schedule's date as canonical for the game ID, but
    # preserve the source player-boxscore date and flag every difference.
    source_player_date = iso_date(row["game_date"], label)
    date_offset = (date.fromisoformat(source_player_date) - date.fromisoformat(match["game_date"])).days
    date_match_status = ("MATCH" if date_offset == 0 else
                         "ONE_DAY_DIFFERENCE" if abs(date_offset) == 1 else
                         "LARGER_DATE_DIFFERENCE")
    side = normalize_orientation(row["home_away"], label)
    team = str(row["team_abbrev"] or "").strip().upper()
    expected = match["home"] if side == "HOME" else match["away"]
    if not team or team != expected:
        fail(f"{label}: {gid}/{pid} player team {team!r} differs from official schedule {expected!r}")
    position = str(row["position"] or "").strip().upper()
    goalie_stats_present = row["saves"] is not None or row["goals_against"] is not None
    skater_stats_present = row["shots_on_goal"] is not None or row["goals"] is not None
    if goalie_stats_present and skater_stats_present:
        fail(f"{label}: mixed skater/goalie outcome fields for {gid}/{pid}")
    if not goalie_stats_present and not skater_stats_present:
        fail(f"{label}: unclassified player role for {gid}/{pid}")
    is_goalie = goalie_stats_present
    if position in ("G", "GOALIE") and not is_goalie:
        fail(f"{label}: goalie position contradicts statistics for {gid}/{pid}")
    if is_goalie and position not in ("", "G", "GOALIE"):
        fail(f"{label}: skater position contradicts goalie statistics for {gid}/{pid}: {position!r}")
    status = toi_status(row["toi"], label)
    played = status == "PLAYED"
    o: dict = {
        "season": f"{season}-{str((season + 1) % 100).zfill(2)}",
        "game_id": gid,
        "game_date": match["game_date"],
        "source_player_game_date": source_player_date,
        "source_date_offset_days": date_offset,
        "source_date_match_status": date_match_status,
        "game_type": match["game_type"],
        "player_id": pid,
        "player_name": str(row["player_name"] or "").strip(),
        "sweater_number": str(row["sweater_number"] or "").strip(),
        "team_id": str(row["team_id"] or "").strip(),
        "team": team,
        "opponent_team": match["away"] if side == "HOME" else match["home"],
        "home_away": side,
        "position": position if position else ("G" if is_goalie else ""),
        "position_group": "GOALIE" if is_goalie else "SKATER",
        "toi": str(row["toi"] or "").strip(),
        "participation_status": status,
        "participated": int(played),
        "training_target_available": int(played),
        "goalie_starter_raw": (str(row.get("starter") or "").strip() if is_goalie else ""),
    }
    for field in (*SKATER_FIELDS, OPTIONAL_FIELD, *GOALIE_FIELDS, *ALL_ADDITIONAL_STATS):
        o[field] = None
    if played:
        if is_goalie:
            for field in GOALIE_FIELDS:
                o[field] = integer_stat(row[field], f"{label} {field}")
            for field in (*COMMON_ADDITIONAL_STATS, *GOALIE_ADDITIONAL_STATS):
                o[field] = source_stat(row[field])
        else:
            for field in SKATER_FIELDS:
                o[field] = integer_stat(row[field], f"{label} {field}")
            for field in (*COMMON_ADDITIONAL_STATS, *SKATER_ADDITIONAL_STATS):
                o[field] = source_stat(row[field])
            o[OPTIONAL_FIELD] = integer_stat(row[OPTIONAL_FIELD], f"{label} {OPTIONAL_FIELD}", allow_missing=True)
            if o[OPTIONAL_FIELD] is not None and o[OPTIONAL_FIELD] > o["goals"]:
                fail(f"{label}: power_play_goals exceeds goals for {gid}/{pid}")
            if o["points"] != o["goals"] + o["assists"]:
                fail(f"{label}: points != goals + assists for {gid}/{pid}")
    return o


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parquet_checked(pl, path: Path, required: tuple[str, ...]) -> list[dict]:
    if not path.is_file():
        fail(f"Missing required historical Parquet: {path}")
    df = pl.read_parquet(path)
    absent = sorted(set(required) - set(df.columns))
    if absent:
        fail(f"{path}: required columns missing: {absent}")
    return df.select(list(required)).to_dicts()


def build(history: Path, downloads: Path) -> None:
    if not history.is_dir():
        fail(f"Historical Parquet directory not found: {history}")
    if not downloads.is_dir():
        fail(f"Downloads folder does not exist: {downloads}")
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Existing Python environment requires polars. Nothing was installed.") from exc
    print(f"BUILDER | {BUILDER_VERSION} | accepts SportsDataverse R/P/PR/A game types", flush=True)
    summaries: list[dict] = []
    source_files: list[dict] = []
    totals = Counter()
    with tempfile.TemporaryDirectory(prefix="nhl_targets_build_", dir=downloads) as td:
        temp = Path(td)
        data_path = temp / f"{BASE_NAME}.csv"
        summary_path = temp / f"{BASE_NAME}_summary.csv"
        manifest_path = temp / f"{BASE_NAME}_manifest.json"
        dates_path = temp / f"{BASE_NAME}_date_discrepancies.csv"
        date_audit = Counter()
        all_source_dates = defaultdict(set)
        keys_seen: set[tuple[str, str]] = set()
        with data_path.open("w", encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for year in range(2009, 2026):
                sch = history / "schedules" / f"season_{year}.parquet"
                box = history / "player_boxscores" / f"season_{year}.parquet"
                print(f"SEASON {year}-{str((year+1)%100).zfill(2)} | reading local historical files", flush=True)
                schedules = read_parquet_checked(pl, sch, REQUIRED_SCHEDULE)
                games = parse_schedule(schedules, year)
                rows = read_parquet_checked(pl, box, REQUIRED_BOXSCORES)
                counts = Counter()
                covered: set[str] = set()
                season_keys: set[tuple[str, str]] = set()
                for ix, row in enumerate(rows, 1):
                    record = make_record(row, year, games, ix)
                    if record is None:
                        counts["source_rows_excluded"] += 1
                        continue
                    key = (record["game_id"], record["player_id"])
                    all_source_dates[record["game_id"]].add(record["source_player_game_date"])
                    if record["source_date_match_status"] != "MATCH":
                        counts["date_discrepant_rows"] += 1
                        if record["source_date_match_status"] == "LARGER_DATE_DIFFERENCE":
                            counts["large_date_discrepant_rows"] += 1
                        date_audit[(record["season"], record["game_id"], record["game_date"],
                                    record["source_player_game_date"], record["source_date_offset_days"],
                                    record["source_date_match_status"])] += 1
                    if key in season_keys or key in keys_seen:
                        fail(f"season {year}: duplicate player-game {key}")
                    season_keys.add(key)
                    keys_seen.add(key)
                    covered.add(record["game_id"])
                    writer.writerow({k: "" if v is None else v for k, v in record.items()})
                    counts["player_game_rows"] += 1
                    counts["skater_rows" if record["position_group"] == "SKATER" else "goalie_rows"] += 1
                    counts[record["participation_status"].lower()] += 1
                    if record["position_group"] == "SKATER" and record["participated"] == 1:
                        counts["skater_played"] += 1
                        if record[OPTIONAL_FIELD] is None:
                            counts["skater_played_missing_power_play_goals"] += 1
                    if record["position_group"] == "GOALIE" and record["participated"] == 1:
                        counts["goalie_played"] += 1
                missing_games = sorted(set(games) - covered)
                if missing_games:
                    fail(f"season {year}: {len(missing_games)} scheduled regular/playoff games have no player boxscore; first IDs: {missing_games[:15]}")
                summary = {
                    "season": f"{year}-{str((year+1)%100).zfill(2)}",
                    "official_scheduled_games": len(games),
                    "games_with_player_records": len(covered),
                    "source_player_rows": len(rows),
                    "source_rows_excluded": counts["source_rows_excluded"],
                    "player_game_rows": counts["player_game_rows"],
                    "skater_rows": counts["skater_rows"],
                    "goalie_rows": counts["goalie_rows"],
                    "skater_played": counts["skater_played"],
                    "goalie_played": counts["goalie_played"],
                    "played": counts["played"],
                    "zero_toi": counts["zero_toi"],
                    "unknown_toi": counts["unknown_toi"],
                    "skater_played_missing_power_play_goals": counts["skater_played_missing_power_play_goals"],
                    "date_discrepant_rows": counts["date_discrepant_rows"],
                    "large_date_discrepant_rows": counts["large_date_discrepant_rows"],
                }
                summaries.append(summary)
                totals.update({k: v for k, v in counts.items()})
                totals["games_with_player_records"] += len(covered)
                source_files.extend([
                    {"path": str(sch), "sha256": sha256(sch), "rows": len(schedules)},
                    {"path": str(box), "sha256": sha256(box), "rows": len(rows)},
                ])
                print(f"  games={len(covered)} rows={counts['player_game_rows']} played={counts['played']} zero_toi={counts['zero_toi']} unknown_toi={counts['unknown_toi']} missing_pp_goals={counts['skater_played_missing_power_play_goals']} differing_dates={counts['date_discrepant_rows']}", flush=True)
        with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        with dates_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DATE_AUDIT_FIELDS)
            writer.writeheader()
            for (season, gid, schedule_date, source_date, offset, status), nrows in sorted(date_audit.items()):
                writer.writerow({"season": season, "game_id": gid,
                                 "schedule_game_date": schedule_date,
                                 "source_player_game_date": source_date,
                                 "offset_days": offset, "affected_player_rows": nrows, "status": status})
        inconsistent_source_dates = sorted(gid for gid, dates in all_source_dates.items() if len(dates) > 1)
        incomplete = totals["unknown_toi"] > 0 or totals["skater_played_missing_power_play_goals"] > 0
        date_review = totals["large_date_discrepant_rows"] > 0 or bool(inconsistent_source_dates)
        manifest = {
            "builder_version": BUILDER_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": ("GENERATED_REQUIRES_DATE_REVIEW" if date_review else
                       "INCOMPLETE_SOURCE_FIELDS" if incomplete else
                       "GENERATED_WITH_DATE_OFFSETS_AND_DOCUMENTED_PPP_GAP" if date_audit else
                       "GENERATED_WITH_DOCUMENTED_PPP_GAP"),
            "input_root_read_only": str(history),
            "output_folder": str(downloads),
            "seasons": [s["season"] for s in summaries],
            "source": "SportsDataverse local historical Parquet; audited against official NHL boxscores except three documented exclusions",
            "game_types": {"2": "regular season", "3": "playoffs"},
            "excluded_game_ids": EXCLUDED,
            "grain": ["game_id", "player_id"],
            "training_target_available": "1 only for strictly positive time on ice, 0 for zero or unknown TOI; does not mean sportsbook eligibility",
            "zero_or_unknown_toi": "identity retained; all outcome values empty; do not train or grade as zero-stat player appearance",
            "available_skater_targets": [*SKATER_FIELDS, OPTIONAL_FIELD],
            "available_goalie_targets": list(GOALIE_FIELDS),
            "additional_observed_fields_not_separately_stat_audited": list(ALL_ADDITIONAL_STATS),
            "unavailable_targets": {
                "power_play_assists": "No verified per-player-game field in inspected historical Parquet",
                "power_play_points": "NOT DERIVED: power_play_goals alone is not power_play_points",
            },
            "date_policy": "Training game_date is from the audited SDV schedule keyed on official NHL game_id; source_player_game_date is preserved; offsets are not automatically assumed to be timezone errors",
            "date_discrepancy_game_count": len(set(key[1] for key in date_audit)),
            "date_discrepancy_row_count": sum(date_audit.values()),
            "large_date_discrepancy_row_count": totals["large_date_discrepant_rows"],
            "games_with_multiple_source_dates_count": len(inconsistent_source_dates),
            "games_with_multiple_source_dates": inconsistent_source_dates[:1000],
            "total_counts": dict(totals),
            "output_files": [f"{BASE_NAME}.csv", f"{BASE_NAME}_summary.csv", f"{BASE_NAME}_date_discrepancies.csv", f"{BASE_NAME}_manifest.json"],
            "source_files_sha256": source_files,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Publish only after every one of the 17 seasons passes structural checks.
        for path in (data_path, summary_path, dates_path, manifest_path):
            dest = downloads / path.name
            os.replace(path, dest)
            print(f"REPORT | {dest}", flush=True)
    print(f"STATUS | {manifest['status']} | rows={totals['player_game_rows']} | games={totals['games_with_player_records']}", flush=True)
    if incomplete:
        print("WARNING: Unknown TOI or missing power-play goals exists; inspect summary and manifest before using these targets.", file=sys.stderr, flush=True)
    if date_review:
        print("WARNING: Date inconsistencies larger than one day, or multiple source dates within one game, require review before training.", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path, help="Read-only SDV historical Parquet root")
    parser.add_argument("--downloads", required=True, type=Path, help="Only output folder")
    args = parser.parse_args()
    try:
        build(args.history, args.downloads)
    except Exception as exc:
        print(f"FAILED | {exc} | No completed output files were published by this run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
