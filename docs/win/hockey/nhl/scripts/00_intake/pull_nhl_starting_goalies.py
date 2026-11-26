#!/usr/bin/env python3
"""Scrape Daily Faceoff starting goalies, joining verified NHL game/team/player IDs.

Run from GitHub Actions or the repository root:
    python docs/win/hockey/nhl/scripts/00_intake/pull_nhl_starting_goalies.py

Requires the dated NHL schedule and team map. Validates goalie identities
against current official NHL team rosters; compares saved SDV goalie snapshots
for staleness. Initial captures write 33 columns. Later runs merge only
status/confirmation fields, and starter IDs/stats when the starter changes.
Existing game rows are immutable at/after their scheduled UTC start.
Outputs: 00_intake/lineups/goalies/YYYY-MM-DD_goalies.csv.
Optional --date YYYY-MM-DD; otherwise uses today's Eastern calendar date.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from contextlib import contextmanager
from html import unescape
import json
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCE_URL = "https://www.dailyfaceoff.com/starting-goalies"
NHL_API_ROOT = "https://api-web.nhle.com/v1"
BASE_DIR = Path(__file__).resolve().parents[2]  # docs/win/hockey/nhl
OUTPUT_DIR = BASE_DIR / "00_intake" / "lineups" / "goalies"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
ROSTER_DIR = BASE_DIR / "sdv" / "goalie"
TEAM_MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"

HEADERS = [
    "game_id",
    "CapturedUTC",
    "date",
    "time",
    "dateGmt",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "homeTeamAbbreviation",
    "awayTeamAbbreviation",
    "home_player_id",
    "away_player_id",
    "homeGoalieWins",
    "awayGoalieWins",
    "homeGoalieLosses",
    "awayGoalieLosses",
    "homeGoalieOvertimeLosses",
    "awayGoalieOvertimeLosses",
    "homeGoalieShutouts",
    "awayGoalieShutouts",
    "homeGoalieSavePercentage",
    "awayGoalieSavePercentage",
    "homeGoalieGoalsAgainstAvg",
    "awayGoalieGoalsAgainstAvg",
    "homeNewsStrengthId",
    "awayNewsStrengthId",
    "homeNewsStrengthName",
    "awayNewsStrengthName",
    "homeNewsDetails",
    "awayNewsDetails",
    "homeConfirmedAt",
    "awayConfirmedAt",
]

STATUS_SUFFIXES = ("NewsStrengthId", "NewsStrengthName", "NewsDetails", "ConfirmedAt")
STAT_SUFFIXES = ("GoalieWins", "GoalieLosses", "GoalieOvertimeLosses",
                 "GoalieShutouts", "GoalieSavePercentage", "GoalieGoalsAgainstAvg")
ISO_STAMP = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z"

REQUIRED_DFO_FIELDS = (
    "date", "time", "dateGmt", "homeTeamName", "awayTeamName",
    "homeTeamAbbreviation", "awayTeamAbbreviation",
)
SCHEDULE_FIELDS = (
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "home_team_abbrev", "away_team_abbrev", "home_team_id",
    "away_team_id", "start_time_utc",
)


class NextDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_next_data = False
        self.next_data_parts = []
        self.found = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.in_next_data = True
            self.found = True

    def handle_data(self, data):
        if self.in_next_data:
            self.next_data_parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "script":
            self.in_next_data = False


def read_games(html):
    parser = NextDataParser()
    parser.feed(html)
    if not parser.found:
        raise ValueError("Daily Faceoff __NEXT_DATA__ script not found")

    document = json.loads("".join(parser.next_data_parts))
    data = document["props"]["pageProps"]["data"]
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict):
        data = data.get("games")
    if not isinstance(data, list):
        raise ValueError("Expected a game list in pageProps.data")
    if not all(isinstance(game, dict) for game in data):
        raise ValueError("Game list contains a non-object record")
    return data


def read_csv(path, required_fields, allow_empty=False):
    if not path.is_file():
        raise FileNotFoundError(f"Required input file missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        records = list(reader)
    if not records and not allow_empty:
        raise ValueError(f"Required input file is empty: {path}")
    if any(None in row or any(v is None for v in row.values()) for row in records):
        raise ValueError(f"Malformed CSV row with unexpected extra cells: {path}")
    return records


def norm_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def require_text(record, field, context):
    value = str(record.get(field) or "").strip()
    if not value:
        raise ValueError(f"{context}: missing {field}")
    return value


def nhl_id(value, context):
    text = str(value or "").strip()
    if not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"{context}: invalid NHL ID {value!r}")
    return text


def normalized_date(value):
    raw = str(value or "").strip().replace("_", "-")
    return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()


def parse_utc(value, context):
    try:
        result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        return result.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid UTC timestamp {value!r}") from exc


def load_team_map(path=TEAM_MAP_PATH):
    rows = read_csv(
        path,
        ("league", "source", "alias", "canonical_team", "nhl_team_id", "nhl_abbrev"),
    )
    by_abbrev = {}
    by_official_alias = {}
    for row in rows:
        if row["league"].strip().casefold() != "nhl" or row["source"].strip() != "official_nhl":
            continue
        abbrev = require_text(row, "nhl_abbrev", str(path)).upper()
        team = {
            "id": nhl_id(row["nhl_team_id"], str(path)),
            "canonical": require_text(row, "canonical_team", str(path)),
            "alias": require_text(row, "alias", str(path)),
        }
        if abbrev in by_abbrev:
            raise ValueError(f"Duplicate official NHL team abbreviation {abbrev} in {path}")
        by_abbrev[abbrev] = team
        key = norm_name(team["alias"])
        if key in by_official_alias:
            raise ValueError(f"Duplicate official NHL team alias {team['alias']!r} in {path}")
        by_official_alias[key] = abbrev
    if not by_abbrev:
        raise ValueError(f"No official NHL team entries in {path}")
    return by_abbrev


def load_schedule(game_date, team_map, schedule_dir=SCHEDULE_DIR):
    path = schedule_dir / f"NHL_{game_date.replace('-', '_')}.csv"
    records = read_csv(path, SCHEDULE_FIELDS, allow_empty=True)
    index = {}
    game_ids = set()
    for number, row in enumerate(records, start=2):
        context = f"{path} row {number}"
        date = normalized_date(require_text(row, "game_date", context))
        if date != game_date:
            raise ValueError(f"{context}: wrong game_date {date} (expected {game_date})")
        home = require_text(row, "home_team_abbrev", context).upper()
        away = require_text(row, "away_team_abbrev", context).upper()
        if home == away or home not in team_map or away not in team_map:
            raise ValueError(f"{context}: invalid team abbreviations {home}/{away}")
        for side, abbr in (("home", home), ("away", away)):
            team = team_map[abbr]
            source_id = nhl_id(row[f"{side}_team_id"], context)
            if source_id != team["id"]:
                raise ValueError(f"{context}: {abbr} schedule team ID {source_id} differs from mapped NHL ID {team['id']}")
            name = require_text(row, f"{side}_team", context)
            if norm_name(name) != norm_name(team["alias"]):
                raise ValueError(f"{context}: unexpected schedule team name {name!r} for {abbr}")
        game_id = nhl_id(require_text(row, "game_id", context), context)
        if game_id in game_ids:
            raise ValueError(f"{context}: duplicate game_id {game_id}")
        game_ids.add(game_id)
        require_text(row, "game_time", context)
        parse_utc(require_text(row, "start_time_utc", context), context)
        key = (date, home, away)
        if key in index:
            raise ValueError(f"{context}: duplicate schedule matchup {key}")
        index[key] = row
    return index


def load_roster(abbrev, roster_dir=ROSTER_DIR):
    path = roster_dir / f"latest_{abbrev}_goalies_roster.csv"
    records = read_csv(path, ("id", "first_name_default", "last_name_default", "position_group"))
    names = {}
    ids = set()
    for number, row in enumerate(records, start=2):
        context = f"{path} row {number}"
        if require_text(row, "position_group", context).lower() != "goalies":
            raise ValueError(f"{context}: non-goalie in goalie roster")
        player_id = nhl_id(require_text(row, "id", context), context)
        if player_id in ids:
            raise ValueError(f"{context}: duplicate NHL player ID {player_id}")
        ids.add(player_id)
        full_name = " ".join((require_text(row, "first_name_default", context),
                              require_text(row, "last_name_default", context)))
        key = norm_name(full_name)
        if key in names:
            raise ValueError(f"{context}: ambiguous goalie name {full_name!r} in {abbrev} roster")
        names[key] = player_id
    return names


def fetch_nhl_json(url, attempts=3):
    """Load fresh NHL roster data; never substitute unverified cached player IDs."""
    import time

    last_error = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NHLStartingGoalies/2.0)",
                "Accept": "application/json",
            })
            with urlopen(request, timeout=25) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError,
                UnicodeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Official NHL API unavailable: {url}: {last_error}")


def load_official_goalies(abbrev, nhl_fetch=None):
    """Current NHL roster is authoritative for team membership and player IDs."""
    url = f"{NHL_API_ROOT}/roster/{abbrev}/current"
    payload = (nhl_fetch or fetch_nhl_json)(url)
    if not isinstance(payload, dict) or not isinstance(payload.get("goalies"), list):
        raise ValueError(f"{url}: missing goalies array")
    goalies = payload["goalies"]
    if not goalies:
        raise ValueError(f"{url}: no current goalies; cannot verify team membership")

    names = {}
    ids = set()
    for number, record in enumerate(goalies, start=1):
        context = f"{url} goalie {number}"
        if not isinstance(record, dict):
            raise ValueError(f"{context}: not an object")
        if record.get("positionCode") not in (None, "G"):
            raise ValueError(f"{context}: unexpected position {record['positionCode']!r}")
        player_id = nhl_id(record.get("id"), context)
        first, last = record.get("firstName"), record.get("lastName")
        if isinstance(first, dict):
            first = first.get("default")
        if isinstance(last, dict):
            last = last.get("default")
        if not isinstance(first, str) or not isinstance(last, str):
            raise ValueError(f"{context}: missing firstName/lastName")
        name = norm_name(first + " " + last)
        if not name:
            raise ValueError(f"{context}: missing normalized name")
        if name in names or player_id in ids:
            raise ValueError(f"{context}: duplicate official NHL goalie name/ID")
        names[name] = player_id
        ids.add(player_id)
    return names


def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def build_rows(games, schedule_dir=SCHEDULE_DIR, roster_dir=ROSTER_DIR,
               team_map_path=TEAM_MAP_PATH, nhl_fetch=None, *,
               captured_at=None):
    dates = set()
    keys = set()
    for number, game in enumerate(games, start=1):
        context = f"Daily Faceoff game {number}"
        for field in REQUIRED_DFO_FIELDS:
            require_text(game, field, context)
        dates.add(normalized_date(game["date"]))
        key = (str(game["date"]), str(game["homeTeamAbbreviation"]).strip().upper(),
               str(game["awayTeamAbbreviation"]).strip().upper())
        if key in keys:
            raise ValueError(f"{context}: duplicate source matchup {key}")
        keys.add(key)
    if len(dates) != 1:
        raise ValueError(f"Daily Faceoff contains multiple game dates: {sorted(dates)}")
    game_date = dates.pop()

    team_map = load_team_map(team_map_path)
    schedule = load_schedule(game_date, team_map, schedule_dir)
    roster_cache = {}
    official_roster_cache = {}
    captured = (captured_at or datetime.now(timezone.utc)).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = []
    matched_game_ids = set()

    for number, game in enumerate(games, start=1):
        context = f"Daily Faceoff game {number}"
        home = str(game["homeTeamAbbreviation"]).strip().upper()
        away = str(game["awayTeamAbbreviation"]).strip().upper()
        key = (normalized_date(game["date"]), home, away)
        match = schedule.get(key)
        if match is None:
            raise ValueError(f"{context}: no unique official NHL schedule matchup {key}")
        game_id = match["game_id"].strip()
        if game_id in matched_game_ids:
            raise ValueError(f"{context}: duplicate matched NHL game_id {game_id}")
        matched_game_ids.add(game_id)
        if str(game["time"]).strip() != match["game_time"].strip():
            raise ValueError(f"{context} game_id={game_id}: game time differs from official schedule")
        if parse_utc(game["dateGmt"], context) != parse_utc(match["start_time_utc"], context):
            raise ValueError(f"{context} game_id={game_id}: UTC start differs from official schedule")

        player_ids = {}
        for side, abbrev in (("home", home), ("away", away)):
            expected_name = team_map[abbrev]["canonical"]
            supplied_name = str(game[f"{side}TeamName"]).strip()
            if norm_name(supplied_name) != norm_name(expected_name):
                raise ValueError(f"{context}: DFO {side} team name {supplied_name!r} doesn't match {abbrev} ({expected_name})")
            goalie_name = str(game.get(f"{side}GoalieName") or "").strip()
            if not goalie_name:
                print(f"WARNING: {context} game_id={game_id}: no {side} goalie name from "
                      f"Daily Faceoff; {side}_player_id set to unknown", flush=True)
                player_ids[side] = "unknown"
                continue
            if abbrev not in official_roster_cache:
                official_roster_cache[abbrev] = load_official_goalies(abbrev, nhl_fetch)
            name_key = norm_name(goalie_name)
            player_id = official_roster_cache[abbrev].get(name_key)
            if player_id is None:
                print(f"WARNING: {context} game_id={game_id}: {side} goalie {goalie_name!r} "
                      f"is not on the official CURRENT NHL goalie roster for {abbrev}; "
                      f"{side}_player_id set to unknown", flush=True)
                player_ids[side] = "unknown"
                continue
            # SDV roster snapshots are diagnostic, not authoritative: the normal
            # pipeline does not regenerate them unless diagnostics are requested.
            if abbrev not in roster_cache:
                roster_path = roster_dir / f"latest_{abbrev}_goalies_roster.csv"
                try:
                    roster_cache[abbrev] = load_roster(abbrev, roster_dir)
                except (FileNotFoundError, ValueError, UnicodeError, OSError) as exc:
                    print(f"WARNING: SDV goalie snapshot unavailable/invalid for "
                          f"{abbrev}: {exc}; using official NHL roster", flush=True)
                    roster_cache[abbrev] = {}
            sdv_player_id = roster_cache[abbrev].get(name_key)
            if sdv_player_id != player_id:
                print(f"WARNING: stale SDV goalie roster for {abbrev}: "
                      f"{goalie_name!r} NHL_current_id={player_id} "
                      f"SDV_snapshot_id={sdv_player_id or 'missing'}; "
                      "using official NHL roster", flush=True)
            player_ids[side] = player_id

        row = {field: csv_cell(game.get(field)) for field in HEADERS}
        row.update({
            "game_id": game_id,
            "CapturedUTC": captured,
            "home_team_id": team_map[home]["id"],
            "away_team_id": team_map[away]["id"],
            "home_team": team_map[home]["canonical"],
            "away_team": team_map[away]["canonical"],
            "homeTeamAbbreviation": home,
            "awayTeamAbbreviation": away,
            "home_player_id": player_ids["home"],
            "away_player_id": player_ids["away"],
        })
        rows.append(row)

    return rows, game_date


def write_csv(rows, game_date, output_dir=OUTPUT_DIR, *, before_replace=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{game_date}_goalies.csv"
    temp_name = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_dir,
            prefix=".goalies_", suffix=".tmp", delete=False,
        ) as temp:
            temp_name = temp.name
            writer = csv.DictWriter(temp, fieldnames=HEADERS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if before_replace is not None:
            before_replace()
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination


def eastern_date(now):
    """Use NHL/DFO Eastern calendar date; no third-party tzdata needed on Windows."""
    try:
        return now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except ZoneInfoNotFoundError:
        # Current US DST rules, in force since 2007. Boundaries in UTC.
        year = now.year
        march = datetime(year, 3, 1, 7, tzinfo=timezone.utc)
        november = datetime(year, 11, 1, 6, tzinfo=timezone.utc)
        start = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
        end = november + timedelta(days=(6 - november.weekday()) % 7)
        return (now - timedelta(hours=4 if start <= now < end else 5)).date().isoformat()


def flat(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"(?i)\bSt\.\s+Louis\b", "St Louis", text).split())


def page_text(html):
    # Same visible-HTML extraction used by the successful PowerShell test.
    html = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", " ", html,
                  flags=re.I | re.S)
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    return flat(unescape(re.sub(r"<[^>]+>", " ", html)))


def game_key(game):
    return (normalized_date(game["date"]),
            str(game["homeTeamAbbreviation"]).strip().upper(),
            str(game["awayTeamAbbreviation"]).strip().upper())


def attach_confirmations(html, games, eligible_keys, observed_at):
    """Extract actual DFO status timestamps, never infer them from capture time."""
    page = page_text(html)
    starts = []
    for game in games:
        stamp = parse_utc(game["dateGmt"], "DFO start").isoformat(
            timespec="milliseconds").replace("+00:00", "Z")
        pattern = (re.escape(flat(str(game["awayTeamName"]))) + r"\s+at\s+" +
                   re.escape(flat(str(game["homeTeamName"]))) + r"\s+" + re.escape(stamp))
        matches = list(re.finditer(pattern, page, flags=re.I))
        if len(matches) == 1:
            starts.append((matches[0].start(), game_key(game)))
    blocks = {}
    if len(starts) == len(games) and len({i for i, _ in starts}) == len(games):
        starts.sort()
        for i, (pos, key) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(page)
            blocks[key] = page[pos:end]
    names = Counter(norm_name(g.get(f"{side}GoalieName"))
                    for g in games for side in ("home", "away"))
    for game in games:
        key = game_key(game)
        if key not in eligible_keys:
            continue
        for side in ("home", "away"):
            field = f"{side}ConfirmedAt"
            game[field] = ""
            if str(game.get(f"{side}NewsStrengthName") or "").strip().casefold() != "confirmed":
                continue
            name = str(game.get(f"{side}GoalieName") or "").strip()
            if not name:
                raise ValueError(f"{key} {side}: Confirmed without a goalie name")
            text = blocks.get(key)
            if text is None:
                if names[norm_name(name)] != 1:
                    raise ValueError(f"{key} {side}: ambiguous goalie name {name}")
                text = page
            pattern = re.escape(flat(name)) + r"\s+Confirmed\s+(" + ISO_STAMP + r")"
            values = set(re.findall(pattern, text, flags=re.I))
            # In a matchup containing the same name twice, never guess the side.
            other = "away" if side == "home" else "home"
            if norm_name(name) == norm_name(game.get(f"{other}GoalieName")):
                raise ValueError(f"{key}: duplicate goalie names within matchup")
            if len(values) != 1:
                raise ValueError(f"{key} {side} {name}: missing/ambiguous confirmation timestamp")
            stamp = values.pop()
            if not re.fullmatch(ISO_STAMP, stamp):
                raise ValueError(f"{key} {side}: invalid timestamp {stamp}")
            if parse_utc(stamp, f"{key} {side}") > observed_at:
                raise ValueError(f"{key} {side}: confirmation is later than capture time")
            game[field] = stamp


def read_existing(path, game_date):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADERS:
            raise ValueError(f"{path}: expected exact 33-column schema; original file retained")
        rows = list(reader)
    ids = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"{path}: malformed row")
        gid = nhl_id(row["game_id"], str(path))
        if gid in ids or normalized_date(row["date"]) != game_date:
            raise ValueError(f"{path}: duplicate game ID or incorrect date")
        parse_utc(row["dateGmt"], str(path))
        ids.add(gid)
    return rows


def as_text_row(row):
    return {field: str(row.get(field)) if row.get(field) is not None else ""
            for field in HEADERS}


def merge_rows(existing, incoming, now):
    """Retain all original rows; only approved fields may change before start."""
    incoming_by_id = {str(r["game_id"]): as_text_row(r) for r in incoming}
    if len(incoming_by_id) != len(incoming):
        raise ValueError("Duplicate incoming game IDs")
    output = []
    for old in existing:
        gid = old["game_id"]
        fresh = incoming_by_id.pop(gid, None)
        if now >= parse_utc(old["dateGmt"], gid) or fresh is None:
            output.append(old.copy())
            continue
        for field in ("date", "home_team_id", "away_team_id", "homeTeamAbbreviation",
                      "awayTeamAbbreviation"):
            if old[field] != fresh[field]:
                raise ValueError(f"{gid}: existing {field} differs from verified source")
        if parse_utc(old["dateGmt"], gid) != parse_utc(fresh["dateGmt"], gid):
            raise ValueError(f"{gid}: scheduled start changed; original row retained")
        row = old.copy()
        for side in ("home", "away"):
            id_field = f"{side}_player_id"
            # A missing/unresolved source identity must not overwrite a known goalie
            # or attach someone else's confirmation to the existing ID.
            if fresh[id_field] == "unknown" and old[id_field] != "unknown":
                print(f"WARNING: {gid} {side}: unresolved source goalie; retaining existing side", flush=True)
                continue
            if not fresh[f"{side}NewsStrengthName"] and old[f"{side}NewsStrengthName"]:
                print(f"WARNING: {gid} {side}: missing source status; retaining existing side", flush=True)
                continue
            if not fresh[f"{side}NewsStrengthId"] and old[f"{side}NewsStrengthId"]:
                print(f"WARNING: {gid} {side}: missing source status ID; retaining existing side", flush=True)
                continue
            if fresh[id_field] != old[id_field]:
                row[id_field] = fresh[id_field]
                for suffix in STAT_SUFFIXES:
                    row[side + suffix] = fresh[side + suffix]
            for suffix in STATUS_SUFFIXES:
                row[side + suffix] = fresh[side + suffix]
        output.append(row)
    # New rows are initial captures (even if the first run is after start).
    output.extend(incoming_by_id.values())
    return output


@contextmanager
def output_lock(output_dir):
    """Serialize runs using an OS lock that is released even after a crash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir.parent / ".starting_goalies.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fetch_page(game_date):
    request = Request(
        f"{SOURCE_URL}/{game_date}",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=45) as response:
        html = response.read().decode(response.headers.get_content_charset() or "utf-8")
    return html, datetime.now(timezone.utc)


def run(game_date, output_dir=OUTPUT_DIR, schedule_dir=SCHEDULE_DIR,
        roster_dir=ROSTER_DIR, team_map_path=TEAM_MAP_PATH, nhl_fetch=None,
        page_fetch=None, clock=None):
    clock = clock or (lambda: datetime.now(timezone.utc))
    game_date = normalized_date(game_date)
    destination = output_dir / f"{game_date}_goalies.csv"
    with output_lock(output_dir):
        existing = read_existing(destination, game_date)
        by_id = {r["game_id"]: r for r in existing}
        team_map = load_team_map(team_map_path)
        schedule = load_schedule(game_date, team_map, schedule_dir)
        scheduled_ids = {r["game_id"].strip() for r in schedule.values()}
        now = clock()
        if destination.exists() and scheduled_ids <= by_id.keys() and all(
                now >= parse_utc(r["dateGmt"], r["game_id"]) for r in existing):
            print(f"CSV unchanged (all rows locked): {destination}")
            return destination
        html, captured = (page_fetch or fetch_page)(game_date)
        games = read_games(html)
        heading_date = datetime.strptime(game_date, "%Y-%m-%d")
        heading = f"NHL Starting Goalies: {heading_date:%A, %B} {heading_date.day}"
        if heading.casefold() not in page_text(html).casefold():
            raise ValueError(f"Expected DFO page heading not found: {heading}")
        seen = set()
        eligible = []
        now = clock()
        for game in games:
            for field in REQUIRED_DFO_FIELDS:
                require_text(game, field, "Daily Faceoff")
            key = game_key(game)
            if key in seen or key not in schedule:
                raise ValueError(f"Duplicate or unmatched DFO game: {key}")
            seen.add(key)
            gid = schedule[key]["game_id"].strip()
            old = by_id.get(gid)
            if old is not None and now >= parse_utc(old["dateGmt"], gid):
                continue
            eligible.append(game)
        # Never silently publish an incomplete initial day or partial source page.
        # Previously captured rows remain valid if DFO omits a matchup later.
        missing = [key for key, row in schedule.items()
                   if key not in seen and row["game_id"].strip() not in by_id]
        if missing:
            raise ValueError(f"DFO missing uncaptured scheduled games: {missing}")
        incoming = []
        if eligible:
            attach_confirmations(html, games, {game_key(g) for g in eligible}, captured)
            incoming, source_date = build_rows(
                eligible, schedule_dir, roster_dir, team_map_path, nhl_fetch,
                captured_at=captured)
            if source_date != game_date:
                raise ValueError("DFO date differs from requested date")
        # Recheck scheduled start after all network requests / validations.
        merged = merge_rows(existing, incoming, clock())
        if destination.exists() and merged == existing:
            print(f"CSV unchanged: {destination}")
            return destination
        # Detect a row crossing its lock time during CSV staging, before replacement.
        changed_existing = [old for old, new in zip(existing, merged) if old != new]
        def before_replace():
            now = clock()
            if any(now >= parse_utc(row["dateGmt"], row["game_id"])
                   for row in changed_existing):
                raise ValueError("Scheduled start reached during write; original CSV retained. Rerun.")
        write_csv(merged, game_date, output_dir, before_replace=before_replace)
        print(f"CSV: {destination}")
        print(f"Games: {len(merged)} | Columns: {len(HEADERS)} | "
              f"New: {len(merged)-len(existing)} | Updated: {len(changed_existing)} | "
              f"Retained: {len(existing)-len(changed_existing)}")
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="NHL game date YYYY-MM-DD; defaults to today Eastern time")
    args = parser.parse_args()
    run(args.date or eastern_date(datetime.now(timezone.utc)))


if __name__ == "__main__":
    main()
