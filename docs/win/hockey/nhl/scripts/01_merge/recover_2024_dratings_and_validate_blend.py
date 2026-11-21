#!/usr/bin/env python3
"""Recover public D-Ratings 2024-25 NHL completed-game prediction archive.

Join on OFFICIAL results by ET game date and explicit home/away team ID, then
verify final scores; do not infer orientation or fabricate pregame timestamps.
Calls existing V14 blend test with --validation-file when archive coverage passes.

The site's historical pages do NOT independently authenticate that the saved
probability was actually visible before puck drop. Accordingly the 2024 blend
selection remains PROVISIONAL regardless of accuracy. No production changes.

Uses the same Playwright Chromium dependency as the repository's production
D-Ratings scraper. Outputs ONLY beneath my_probability/independent_v14_dratings_blend.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

VERSION = "dratings_2024_archive_20261120_2_placeholder_fix"
BASE_URL = "https://www.dratings.com/predictor/nhl-hockey-predictions/completed/{}"
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")


def assert_true(condition, message):
    if not bool(condition):
        raise ValueError(message)


def plain(value):
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.casefold())
    return " ".join(s.split())


def load_map(root):
    src = root / "config" / "mapping" / "team_map_nhl.csv"
    assert_true(src.is_file(), f"Missing team identity map: {src}")
    by_source, by_abbr, canonical = {}, {}, {}
    with src.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("league") != "nhl":
                continue
            id_ = str(r.get("nhl_team_id") or "").strip()
            abbr = str(r.get("nhl_abbrev") or "").strip().upper()
            source = str(r.get("source") or "").strip()
            name = plain(r.get("alias") or "")
            # The repository's shared alias map intentionally includes
            # playoff placeholders (TBD, EC F1/F2, conference champion) with
            # no official team identity. They cannot match an NHL result.
            # Skip ONLY explicitly marked TBD placeholders; fail closed for
            # malformed real-team rows.
            if (source == "shared" and plain(r.get("canonical_team") or "") == "tbd"
                    and not id_ and not abbr):
                continue
            if not (id_.isdigit() and abbr and name):
                raise ValueError(f"Malformed team map row: {r}")
            k = (source, name)
            if k in by_source and by_source[k] != id_:
                raise ValueError(f"Conflicting team aliases: {k}")
            by_source[k] = id_
            if abbr in by_abbr and by_abbr[abbr] != id_:
                raise ValueError(f"Conflicting team abbreviations: {abbr}")
            by_abbr[abbr] = id_
            if source == 'official_nhl':
                canonical[id_] = r['canonical_team']
    assert_true(len(canonical) == 32, f"Expected 32 official NHL team identities, found {len(canonical)}")
    return by_source, by_abbr


def load_official(root, by_abbr):
    src = root / "my_probability" / "independent_v5" / "official_schedule_cache" / "season_2024_regular_games.csv"
    assert_true(src.is_file(), f"Missing independently fetched NHL 2024-25 results: {src}")
    df = pd.read_csv(src, dtype={'game_id': 'string'})
    required = {'game_id','game_date','home_team_full','away_team_full','home_score','away_score'}
    assert_true(required.issubset(df), f"Official history missing fields {sorted(required-set(df))}")
    assert_true(df.game_id.str.fullmatch(r'202402\d{4}').all(), "Official game IDs invalid")
    assert_true(df.game_id.is_unique and len(df) == 1312, "Official 2024-25 schedule not 1312 unique games")
    df['day'] = pd.to_datetime(df.game_date.str.replace('_','-',regex=False),errors='raise').dt.date
    df['home_id'] = df.home_team_full.astype(str).str.upper().map(by_abbr)
    df['away_id'] = df.away_team_full.astype(str).str.upper().map(by_abbr)
    assert_true(df.home_id.notna().all() and df.away_id.notna().all(), 'Official schedule has unmapped team abbreviation')
    df['home_score'] = pd.to_numeric(df.home_score,errors='raise').astype(int)
    df['away_score'] = pd.to_numeric(df.away_score,errors='raise').astype(int)
    assert_true((df.home_score != df.away_score).all(), 'Official score tie or invalid')
    keys=list(zip(df.day,df.away_id,df.home_id))
    assert_true(len(set(keys))==len(keys),'Duplicate official date/matchup keys')
    return {k: r._asdict() for k,r in zip(keys, df.itertuples(index=False))}, df


def parse_pair(value):
    return [p.strip() for p in str(value).split('\n') if p.strip()]


def parse_completed_row(cells):
    if len(cells) != 8:
        return None
    try:
        dtstr = str(cells[0]).replace('\n',' ').strip()
        dtstr = re.sub(r'\s+',' ',dtstr)
        # Repository's existing D-Ratings scraper uses UTC for source time.
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}\s*[AP]M)',dtstr,re.I)
        if not match:
            return None
        dt_utc = datetime.strptime(match.group(1)+' '+match.group(2).upper().replace('  ',' '),"%m/%d/%Y %I:%M %p").replace(tzinfo=UTC)
        day = dt_utc.astimezone(NY).date()
        team=parse_pair(cells[1])
        pp=parse_pair(cells[2])
        ss=parse_pair(cells[5])
        if len(team)!=2 or len(pp)!=2 or len(ss)!=2:
            return None
        p1,p2=[float(x.replace('%',''))/100. for x in pp]
        s1,s2=[int(x) for x in ss]
        if not (.001 < p1 < .999 and .001 < p2 < .999 and abs(p1+p2-1)<.012):
            return None
        if not (0<=s1<=30 and 0<=s2<=30 and s1!=s2):
            return None
        return {'day':day,'away_name':team[0],'home_name':team[1],
                'away_prob':p1,'home_prob':p2,'away_score':s1,'home_score':s2,
                'source_utc':dt_utc.isoformat()}
    except (ValueError,TypeError,OverflowError):
        return None


async def fetch_page(context, semaphore, number):
    async with semaphore:
        page = await context.new_page()
        try:
            url=BASE_URL.format(number)
            response=await page.goto(url,wait_until='domcontentloaded',timeout=45000)
            if response is None or response.status >= 400:
                return number,[],f'HTTP {response.status if response else "empty"}'
            await page.wait_for_selector('table tbody tr',timeout=22000)
            # Wait briefly for dynamically rendered completed rows.
            await page.wait_for_timeout(300)
            tables=await page.locator('table').evaluate_all("""(tables)=>tables.map(t=>({
                header:(t.querySelector('thead')?.innerText||'').trim(),
                rows:Array.from(t.querySelectorAll('tbody tr'),tr=>Array.from(tr.querySelectorAll('td'),td=>td.innerText.trim()))
            }))""")
            rows=[]
            for table in tables:
                header=table['header'].lower()
                if 'final' in header and 'loss' in header:
                    rows.extend(table['rows'])
            if not rows:
                # Legacy layout may have no descriptive headers; preserve only
                # canonical completed-game 8-column rows.
                for table in tables:
                    rows.extend(r for r in table['rows'] if len(r)==8)
            return number,rows,None
        except Exception as exc:
            return number,[],f'{type(exc).__name__}: {str(exc)[:160]}'
        finally:
            await page.close()


async def fetch_archive(start,end,parallel):
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise SystemExit('Playwright is missing; repository hockey_drat_scraper.py requires it') from exc
    output=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True)
        try:
            context=await browser.new_context(user_agent=USER_AGENT,locale='en-US')
            limiter=asyncio.Semaphore(parallel)
            tasks=[asyncio.create_task(fetch_page(context,limiter,n)) for n in range(start,end+1)]
            for idx,completed in enumerate(asyncio.as_completed(tasks),1):
                n,rows,error=await completed
                output.append({'page':n,'url':BASE_URL.format(n),'rows':rows,'error':error})
                if idx%15==0 or idx==len(tasks):
                    print(f'ARCHIVE {idx}/{len(tasks)} pages; completed-table rows fetched={sum(len(z["rows"]) for z in output)}',flush=True)
        finally:
            await browser.close()
    return sorted(output,key=lambda x:x['page'])


def recover(pages, games, by_source):
    matched={};errors=[];stats=Counter();unmatched=[]
    for page in pages:
        if page['error']:
            stats['failed_pages']+=1
            errors.append({'page':page['page'],'error':page['error']})
            continue
        stats['successful_pages']+=1
        for cells in page['rows']:
            game=parse_completed_row(cells)
            if not game:
                stats['unparseable']+=1
                continue
            if not (datetime(2024,9,1).date()<=game['day']<=datetime(2025,7,1).date()):
                stats['outside_period']+=1
                continue
            stats['period_parsed']+=1
            away_id=by_source.get(('dratings',plain(game['away_name'])))
            home_id=by_source.get(('dratings',plain(game['home_name'])))
            if away_id is None or home_id is None:
                stats['unmapped_team']+=1
                unmatched.append({'page':page['page'],'reason':'unmapped_team',**game})
                continue
            key=(game['day'],away_id,home_id)
            official=games.get(key)
            if official is None:
                stats['unknown_matchup']+=1
                unmatched.append({'page':page['page'],'reason':'unknown_matchup',**game})
                continue
            if (game['away_score']!=official['away_score'] or game['home_score']!=official['home_score']):
                # Critical: refuse using the probability if scores disagree.
                stats['score_conflict']+=1
                unmatched.append({'page':page['page'],'reason':'score_conflict',**game})
                continue
            row={'game_id':str(official['game_id']),'game_date':game['day'].strftime('%Y_%m_%d'),
                 'home_prob_moneyline':game['home_prob'],
                 'away_prob_moneyline':game['away_prob'],
                 'source_url':page['url'],
                 'archive_retrieved_at_utc':datetime.now(UTC).isoformat(),
                 'source_game_datetime_utc':game['source_utc'],
                 'original_pregame_observed_at_utc':'',
                 'timestamp_status':'unverified_original_pregame_capture',
                 'source_type':'D_Ratings_public_completed_archive'}
            gid=row['game_id']
            if gid in matched:
                old=matched[gid]
                if abs(old['home_prob_moneyline']-row['home_prob_moneyline'])>1e-9:
                    raise ValueError(f'D-Ratings public archive returned conflicting win probabilities for {gid}: {old["source_url"]} vs {row["source_url"]}')
                stats['duplicate_archive_same_value']+=1
                continue
            matched[gid]=row
            stats['matched_unique']+=1
    return matched,stats,errors,unmatched


def save_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    df.to_csv(tmp,index=False)
    tmp.replace(path)
    print(f'WROTE {path} | rows={len(df)}',flush=True)


def save_json(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
    tmp.replace(path)
    print(f'WROTE {path}',flush=True)


def self_test():
    # Exercise the actual load_map() branch, not just archived row parsing.
    # Shared playoff placeholders are expected; malformed real NHL teams
    # must remain fatal.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mapping = root / "config" / "mapping" / "team_map_nhl.csv"
        mapping.parent.mkdir(parents=True)
        header = ["league", "source", "alias", "canonical_team", "nhl_team_id", "nhl_abbrev"]
        rows = [dict(league="nhl", source="official_nhl", alias=f"Club {i}",
                     canonical_team=f"Club {i}", nhl_team_id=str(i),
                     nhl_abbrev=f"C{i:02d}") for i in range(1,33)]
        rows.extend(dict(league="nhl", source="shared", alias=alias,
                         canonical_team="TBD", nhl_team_id="", nhl_abbrev="")
                    for alias in ("eastern conference champion", "ec f1", "ec f2",
                                  "wc f1", "wc f2", "western conference champion", "TBD"))
        with mapping.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader(); writer.writerows(rows)
        by_source, by_abbr = load_map(root)
        assert_true(len(by_abbr) == 32 and ("shared", "tbd") not in by_source,
                    "Expected placeholder was not ignored")
        rows.append(dict(league="nhl", source="dratings", alias="Malformed Real Club",
                         canonical_team="Malformed Real Club", nhl_team_id="", nhl_abbrev=""))
        with mapping.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader(); writer.writerows(rows)
        try:
            load_map(root)
            raise AssertionError("Malformed real-team mapping was accepted")
        except ValueError as exc:
            assert_true("Malformed team map row" in str(exc), "Unexpected team map error")
    sample=['01/24/2025\n01:07 AM','Montreal Canadiens\nDetroit Red Wings',
            '46.4%\n53.6%','-140\n+120','+1.5\n-1.5','2\n4','-0.25','-0.45']
    x=parse_completed_row(sample)
    assert_true(x is not None and str(x['day'])=='2025-01-23','UTC -> ET date conversion failed')
    assert_true(abs(x['home_prob']-.536)<1e-10 and x['away_score']==2,'Probability or score extraction failed')
    assert_true(parse_completed_row(sample[:5]) is None,'Malformed row not rejected')
    assert_true(parse_completed_row(sample[:2]+['46.4%\n80%']+sample[3:]) is None,'Probability mismatch not rejected')
    games={(x['day'],'8','17'):{'game_id':'2024020754','away_score':2,'home_score':4}}
    names={('dratings','montreal canadiens'):'8',('dratings','detroit red wings'):'17'}
    pages=[{'page':150,'url':BASE_URL.format(150),'rows':[sample],'error':None}]
    matched,stats,errors,unmatched=recover(pages,games,names)
    assert_true(len(matched)==1 and matched['2024020754']['home_prob_moneyline']==.536,'Official-score join failed')
    duplicate=pages+[{'page':151,'url':BASE_URL.format(151),'rows':[sample],'error':None}]
    matched,stats,errors,unmatched=recover(duplicate,games,names)
    assert_true(len(matched)==1 and stats['duplicate_archive_same_value']==1,'Duplicate handling failed')
    wrong=sample.copy();wrong[5]='4\n2'
    matched,stats,errors,unmatched=recover([{'page':150,'url':BASE_URL.format(150),'rows':[wrong],'error':None}],games,names)
    assert_true(len(matched)==0 and stats['score_conflict']==1,'Incorrect orientation must reject')
    bad=sample.copy();bad[2]='50.0%\n50.0%'
    try:
        recover([{'page':150,'url':BASE_URL.format(150),'rows':[sample,bad],'error':None}],games,names)
        raise AssertionError('Conflicting archive probabilities did not fail')
    except ValueError as exc:
        assert_true('conflicting' in str(exc).lower(),'Unexpected error')
    print('SELF-TEST PASSED | mapped playoff placeholders, invalid team rejection, historical date, probability, scores, orientation, duplicates, conflicts')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nhl-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--start-page',type=int,default=95)
    ap.add_argument('--end-page',type=int,default=205)
    ap.add_argument('--parallel',type=int,default=3)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    assert_true(0<=args.start_page<=args.end_page<=500,'Invalid page range')
    assert_true(1<=args.parallel<=5,'Invalid parallelism')
    root=args.nhl_root
    if not (root/'config'/'mapping'/'team_map_nhl.csv').is_file():
        # Stored under .../nhl/scripts/01_merge/, so parents[2] = .../nhl.
        raise FileNotFoundError(f'--nhl-root must point to NHL root, got {root}')
    by_source,by_abbr=load_map(root)
    games,official=load_official(root,by_abbr)
    pages=asyncio.run(fetch_archive(args.start_page,args.end_page,args.parallel))
    found,stats,errors,unmatched=recover(pages,games,by_source)
    out=root/'my_probability'/'independent_v14_dratings_blend'
    report=out/'recovered_2024_archive'
    write=out/'2024_recovered_dratings.csv'
    source=pd.DataFrame(sorted(found.values(),key=lambda x:(x['game_date'],x['game_id'])))
    if source.empty:
        source=pd.DataFrame(columns=['game_id','game_date','home_prob_moneyline','away_prob_moneyline','source_url','timestamp_status'])
    inventory=pd.DataFrame([{'page':p['page'],'url':p['url'],'completed_rows':len(p['rows']),'error':p['error'] or ''} for p in pages])
    save_csv(inventory,report/'archive_page_inventory.csv')
    save_csv(pd.DataFrame(unmatched),report/'unmatched_games.csv')
    save_csv(source,write)
    manifest={'version':VERSION,'archive_pages':f'{args.start_page}-{args.end_page}',
              'official_games':len(official),'matching_games':len(source),
              'coverage':len(source)/len(official),
              'source_win_prob_capture_timestamp_verified':False,
              'source_pregame_status':'NOT_AUTHENTICATED',
              'source_website':'D-Ratings historical completed-games archive',
              'source_retrieved_at_utc':datetime.now(UTC).isoformat(),
              'page_fetch_errors':errors,'counts':dict(stats),
              'production_changed':False}
    save_json(manifest,report/'provenance_and_coverage.json')
    print(f'ARCHIVE MATCHED {len(source)}/{len(official)} official 2024-25 NHL games | coverage={len(source)/len(official):.1%}',flush=True)
    print('WARNING: Archive displays past predictions, but original pregame capture timestamps are NOT VERIFIED.',flush=True)
    if stats['failed_pages']>0:
        print(f'WARNING: {stats["failed_pages"]} archive pages could not be retrieved; investigate report.',flush=True)
    if len(source)<1000 or len(source)/len(official)<.90:
        print('V14 BLEND SELECTION NOT RUN: insufficient independently matched historical D-Ratings coverage.',flush=True)
        return
    from subprocess import run
    v14=Path(__file__).with_name('test_my_probability_v14_dratings_blend.py')
    assert_true(v14.is_file(),f'V14 is required for blend evaluation: {v14}')
    cmd=[sys.executable,str(v14),'--validation-file',str(write)]
    print('RUN V14 matched 2024-25 validation, with unverified pregame provenance.',flush=True)
    child=run(cmd,check=False)
    if child.returncode:
        raise SystemExit(f'V14 validation returned error code {child.returncode}')
    print('DONE: provisional 2024 archive-based blend selection only. No adoption or production changes.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(1)
