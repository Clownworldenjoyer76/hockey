#!/usr/bin/env python3
"""Standalone NHL DRatings scraper with browser evidence for the existing diagnostic workflow.

Run by .github/workflows/nhl_dratings_diagnostic.yml.
Does not import, execute, or modify hockey_drat_scraper.py.
Does not read schedules or contact the GitHub API.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://www.dratings.com/predictor/nhl-hockey-predictions/"
OUTPUT = Path("drat_diagnostic_output")
GAME_DATE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\b")


def write_text(name: str, content: str) -> None:
    (OUTPUT / name).write_text(content, encoding="utf-8")


def write_json(name: str, value: object) -> None:
    write_text(name, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events: list[str] = []
    report: dict[str, object] = {
        "url_requested": URL,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAILED",
        "browser_user_agent": "Chromium default (no forced User-Agent header)",
    }
    return_code = 1
    page = None
    browser = None
    resources = ExitStack()

    try:
        playwright = resources.enter_context(sync_playwright())
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("response", lambda response: events.append(
            f"HTTP {response.status} ({response.request.resource_type}) {response.url}"
        ) if response.request.resource_type in ("document", "xhr", "fetch") or response.status >= 400 else None)
        page.on("requestfailed", lambda request: events.append(
            f"REQUEST FAILED {request.url}: {request.failure}"
        ))
        page.on("pageerror", lambda error: events.append(f"JAVASCRIPT ERROR: {error}"))
        page.on("console", lambda message: events.append(
            f"CONSOLE ERROR: {message.text}"
        ) if message.type == "error" else None)

        # No User-Agent override: compare this browser's behavior against
        # the production scraper's hardcoded Windows Chrome identity.
        response = page.goto(URL, wait_until="load", timeout=45000)
        report["navigation_http_status"] = response.status if response else None
        report["final_url"] = page.url
        report["page_title"] = page.title()
        report["actual_user_agent"] = page.evaluate("navigator.userAgent")

        page.wait_for_selector("table", state="visible", timeout=30000)
        report["table_count"] = page.locator("table").count()
        report["visible_table_count"] = page.locator("table:visible").count()

        rows = [
            [cell.inner_text().strip() for cell in tr.query_selector_all("td")]
            for tr in page.query_selector_all("table tbody tr")
        ]
        game_rows = [row for row in rows if row and GAME_DATE.match(row[0])]
        write_json("raw_rows.json", rows)
        write_json("game_rows.json", game_rows)
        with (OUTPUT / "game_rows.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerows(game_rows)

        report["tbody_row_count"] = len(rows)
        report["game_row_count"] = len(game_rows)
        if not game_rows:
            raise RuntimeError("HTML table was visible, but no NHL game rows were found")
        report["result"] = "SUCCESS"
        return_code = 0

    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        if page is not None:
            for key, func in (
                ("final_url", lambda: page.url),
                ("page_title", lambda: page.title()),
                ("table_count", lambda: page.locator("table").count()),
                ("visible_table_count", lambda: page.locator("table:visible").count()),
                ("frame_urls", lambda: [frame.url for frame in page.frames]),
            ):
                try:
                    report[key] = func()
                except Exception as exc:
                    events.append(f"{key} ERROR: {exc}")
            try:
                write_text("page.html", page.content())
            except Exception as exc:
                events.append(f"HTML SAVE ERROR: {exc}")
            try:
                write_text("page_text.txt", page.locator("body").inner_text(timeout=10000))
            except Exception as exc:
                events.append(f"PAGE TEXT ERROR: {exc}")
            try:
                page.screenshot(path=str(OUTPUT / "screenshot.png"), full_page=True, timeout=15000)
            except Exception as exc:
                events.append(f"SCREENSHOT ERROR: {exc}")
            try:
                page.close()
            except Exception as exc:
                events.append(f"PAGE CLOSE ERROR: {exc}")
        if browser is not None:
            try:
                browser.close()
            except Exception as exc:
                events.append(f"BROWSER CLOSE ERROR: {exc}")
        try:
            resources.close()
        except Exception as exc:
            events.append(f"PLAYWRIGHT CLOSE ERROR: {exc}")
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json("report.json", report)
        write_text("browser_events.txt", "\n".join(events) + "\n")
        print(f"RESULT: {report['result']}")
        print(f"HTTP: {report.get('navigation_http_status', 'unknown')}")
        print(f"ALL TABLE ROWS: {report.get('tbody_row_count', 'unknown')}")
        print(f"NHL GAME ROWS: {report.get('game_row_count', 'unknown')}")
        print(f"EVIDENCE: {OUTPUT.resolve()}")
        if report.get("error"):
            print(f"ERROR: {report['error']}")
    return return_code


if __name__ == "__main__":
    sys.exit(run())
