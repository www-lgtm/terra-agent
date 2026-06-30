#!/usr/bin/env python3
"""Scrape operator → base skill mappings from PRTS Wiki (prts.wiki).

PRTS Wiki stores base skills in two separate locations:

  1. Operator pages (e.g., /w/德克萨斯 → action=raw):
        {{后勤技能
        |后勤技能1-1=恩怨
        |后勤技能1-1阶段=精英0
        |后勤技能2-1=默契
        |后勤技能2-1阶段=精英2
        }}

  2. Skill definition store (/w/后勤技能一览/store → action=raw):
        {{后勤技能信息/store
        |技能名=恩怨
        |房间=贸易站
        |技能图标=bskill_tra_texas1
        |技能描述=当与拉普兰德在同一个贸易站时...
        }}

This script:
  1. Fetches all operator page titles from Category:干员
  2. Downloads raw wikitext for each operator page
  3. Parses CharinfoV2 (rarity, faction) and 后勤技能 templates
  4. Parses the skill store for facility → icon → description
  5. Cross-references to produce operator_base_skills.json

Usage:
    python scripts/scrape_prts_base_skills.py

Output:
    src/knowledge/arknights/operator_base_skills.json

Requires: Python 3.11+, httpx (or urllib), no other dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────

PRTS_API = "https://prts.wiki/api.php"
PRTS_RAW = "https://prts.wiki/index.php"
OUT_DIR = Path(__file__).parent.parent / "src" / "knowledge" / "arknights"
OUT_FILE = OUT_DIR / "operator_base_skills.json"
DELAY = 0.5          # seconds between page fetches (be polite)
BATCH_SIZE = 50      # progress log every N operators
MAX_OPS = 0          # 0 = all operators, set to limit for testing

# Facility name normalization (PRTS → internal keys)
FACILITY_MAP: dict[str, str] = {
    "控制中枢": "Control",
    "发电站": "Power",
    "制造站": "Mfg",
    "贸易站": "Trade",
    "宿舍": "Dorm",
    "加工站": "Processing",
    "办公室": "Office",
    "训练室": "Training",
    "会客室": "Reception",
}


# ── HTTP helpers ──────────────────────────────────────────────────

def _http_get(url: str, retries: int = 3) -> str:
    """GET a URL, return response body as string. Retry on failure."""
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "TerraAgent/0.1 (data-scraper)"})
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("HTTP error (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return ""


def _api_get(params: dict[str, str]) -> dict:
    """Call PRTS MediaWiki API, return parsed JSON dict."""
    query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
    url = f"{PRTS_API}?{query}&format=json"
    text = _http_get(url)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _raw_page(title: str) -> str:
    """Get raw wikitext for a page by title."""
    url = f"{PRTS_RAW}?title={quote(title)}&action=raw"
    return _http_get(url)


# ── Wikitext template parser ──────────────────────────────────────

def _parse_template_params(text: str) -> dict[str, str]:
    """Parse MediaWiki template parameters from a single template block.

    Handles format like:
        {{TemplateName
        |param1=value1
        |param2=value2 with {{nested}} ignored
        }}

    Returns {param_name: param_value} dict.
    """
    params: dict[str, str] = {}
    # Strip surrounding {{...}} including nested
    # Match |key=value pairs, using regex
    pattern = re.compile(r'^\|([^=]+)=(.*)$', re.MULTILINE)
    for match in pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        # Strip any remaining template markup and HTML tags
        value = re.sub(r'\{\{[^}]+\}\}', '', value)
        value = re.sub(r'<[^>]+>', '', value)
        params[key] = value
    return params


def _extract_template_blocks(text: str, template_name: str) -> list[str]:
    """Extract all {{template_name ... }} blocks from wikitext.

    Handles nested templates by counting braces.
    """
    blocks: list[str] = []
    start_pattern = "{{" + template_name
    pos = 0
    while True:
        idx = text.find(start_pattern, pos)
        if idx == -1:
            break
        brace_count = 0
        end = idx
        for i in range(idx, len(text)):
            if text[i:i+2] == "{{":
                brace_count += 1
            elif text[i:i+2] == "}}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 2
                    break
        blocks.append(text[idx:end])
        pos = end
    return blocks


# ── Data extraction ───────────────────────────────────────────────

def fetch_operator_list() -> list[str]:
    """Fetch all operator page titles from Category:干员 via MediaWiki API.

    Excludes subcategories, redirects, and non-operator pages.
    """
    logger.info("Fetching operator list from Category:干员...")
    operators: list[str] = []
    cmcontinue: str | None = None

    while True:
        params: dict[str, str] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:干员",
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = _api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            title = m["title"]
            # Skip subcategories and special pages
            if title.startswith("Category:") or title.startswith("Template:"):
                continue
            if "/" in title:  # Sub-pages like "干员/语音"
                continue
            operators.append(title)

        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
            logger.info("  ... %d operators so far, fetching next page", len(operators))
        else:
            break

    logger.info("Found %d operator pages", len(operators))
    return operators


def parse_skill_store(raw_text: str) -> dict[str, dict[str, str]]:
    """Parse 后勤技能一览/store page into {skill_name: {room, icon, description}}.

    Keys are full skill names like "恩怨" or "恩怨(精英0)".
    """
    logger.info("Parsing skill store...")
    blocks = _extract_template_blocks(raw_text, "后勤技能信息/store")
    skills: dict[str, dict[str, str]] = {}

    for block in blocks:
        params = _parse_template_params(block)
        name = params.get("技能名", "")
        if not name:
            continue
        skills[name] = {
            "room": FACILITY_MAP.get(params.get("房间", ""), params.get("房间", "")),
            "icon": params.get("技能图标", ""),
            "description": params.get("技能描述", ""),
        }

    logger.info("Parsed %d skill definitions", len(skills))
    return skills


def parse_operator_page(raw_text: str) -> dict[str, Any]:
    """Parse an operator's raw wikitext page.

    Returns:
        {
            "name": "德克萨斯",
            "rarity": 5,
            "faction": "企鹅物流",
            "base_skills": [
                {
                    "skill_name": "恩怨",
                    "unlock_condition": "精英0",
                    "skill_slot": 1,
                },
                ...
            ]
        }
    """
    result: dict[str, Any] = {
        "name": "",
        "rarity": 0,
        "faction": "",
        "base_skills": [],
    }

    # Parse CharinfoV2 for rarity and faction
    charinfo_blocks = _extract_template_blocks(raw_text, "CharinfoV2")
    if charinfo_blocks:
        char_params = _parse_template_params(charinfo_blocks[0])
        rarity_str = char_params.get("稀有度", "1")
        try:
            result["rarity"] = int(rarity_str)
        except ValueError:
            result["rarity"] = 1
        result["faction"] = char_params.get("所属国家", "") or char_params.get("所属组织", "")
        result["name"] = char_params.get("干员名", "")

    # Parse 后勤技能 for base skill assignments
    logistics_blocks = _extract_template_blocks(raw_text, "后勤技能")
    for block in logistics_blocks:
        params = _parse_template_params(block)

        # Find all skill entries: 后勤技能A-B=技能名 and 后勤技能A-B阶段=精英0/1/2
        skill_entries: dict[str, dict[str, str]] = {}
        for key, value in params.items():
            # Match patterns like "1-1", "2-1" etc.
            match = re.match(r'后勤技能(\d+)-(\d+)$', key)
            if match:
                slot = match.group(1)
                level = match.group(2)
                entry_key = f"{slot}-{level}"
                if entry_key not in skill_entries:
                    skill_entries[entry_key] = {}
                skill_entries[entry_key]["name"] = value

            stage_match = re.match(r'后勤技能(\d+)-(\d+)阶段$', key)
            if stage_match:
                slot = stage_match.group(1)
                level = stage_match.group(2)
                entry_key = f"{slot}-{level}"
                if entry_key not in skill_entries:
                    skill_entries[entry_key] = {}
                skill_entries[entry_key]["unlock"] = value

        # Convert to base_skills list
        for entry_key, entry_data in skill_entries.items():
            slot, _ = entry_key.split("-")
            skill_name = entry_data.get("name", "")
            unlock = entry_data.get("unlock", "")
            if skill_name:
                result["base_skills"].append({
                    "skill_name": skill_name,
                    "unlock_condition": unlock,
                    "skill_slot": int(slot),
                })

    return result


# ── Cross-referencing ─────────────────────────────────────────────

def cross_reference(
    operators: list[dict[str, Any]],
    skill_store: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Cross-reference operator skills with skill store to build final data.

    For each operator, for each skill name, try to match against the skill store.
    The skill store has entries like "恩怨" and "恩怨(精英0)".
    The operator has skill_name="恩怨" and unlock_condition="精英0".

    Returns list of operator dicts ready for JSON serialization.
    """
    logger.info("Cross-referencing %d operators with %d skills...",
                len(operators), len(skill_store))

    result: list[dict[str, Any]] = []
    matched = 0
    unmatched = 0

    for op in operators:
        op_skills: list[dict[str, Any]] = []
        for bs in op.get("base_skills", []):
            skill_name = bs["skill_name"]
            unlock = bs.get("unlock_condition", "")

            # Try different name formats to match skill store
            store_entry = None
            candidates = [
                f"{skill_name}({unlock})",          # "恩怨(精英0)"
                f"{skill_name}",                    # "恩怨"
                f"{skill_name}({unlock}化)",        # rare format
            ]
            for cand in candidates:
                if cand in skill_store:
                    store_entry = skill_store[cand]
                    break

            if store_entry:
                op_skills.append({
                    "skill_id": store_entry.get("icon", ""),
                    "facility": store_entry.get("room", ""),
                    "skill_name": skill_name,
                    "unlock_condition": unlock,
                    "description": store_entry.get("description", ""),
                })
                matched += 1
            else:
                # Skill not found in store — still record it
                op_skills.append({
                    "skill_id": "",
                    "facility": "",
                    "skill_name": skill_name,
                    "unlock_condition": unlock,
                    "description": "",
                })
                unmatched += 1
                logger.debug("Unmatched skill: %s for operator %s",
                           skill_name, op.get("name", "?"))

        if op_skills:
            result.append({
                "id": "",
                "name": op.get("name", ""),
                "rarity": op.get("rarity", 1),
                "faction": op.get("faction", ""),
                "base_skills": op_skills,
            })

    logger.info("Cross-reference: %d matched, %d unmatched skills", matched, unmatched)
    return result


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Phase 1: Fetch and parse skill store
    logger.info("=" * 60)
    logger.info("Phase 1: Fetching skill definitions (后勤技能一览/store)")
    logger.info("=" * 60)

    store_raw = _raw_page("后勤技能一览/store")
    if not store_raw:
        logger.error("Failed to fetch skill store page")
        return 1
    skill_store = parse_skill_store(store_raw)

    # Phase 2: Fetch operator list
    logger.info("=" * 60)
    logger.info("Phase 2: Fetching operator list")
    logger.info("=" * 60)

    operator_titles = fetch_operator_list()
    if MAX_OPS > 0:
        operator_titles = operator_titles[:MAX_OPS]
        logger.info("Limited to %d operators (MAX_OPS=%d)", len(operator_titles), MAX_OPS)

    # Phase 3: Fetch and parse each operator page
    logger.info("=" * 60)
    logger.info("Phase 3: Fetching operator pages")
    logger.info("=" * 60)

    operators: list[dict[str, Any]] = []
    skipped = 0

    for i, title in enumerate(operator_titles):
        raw_text = _raw_page(title)
        if not raw_text:
            skipped += 1
            continue

        op_data = parse_operator_page(raw_text)
        if op_data["base_skills"]:
            operators.append(op_data)

        if (i + 1) % BATCH_SIZE == 0:
            logger.info("Processed %d/%d operators (%d have base skills)",
                       i + 1, len(operator_titles), len(operators))

        time.sleep(DELAY)

    logger.info("Fetched %d operators (%d have base skills, %d skipped)",
                len(operators), len([o for o in operators if o["base_skills"]]), skipped)

    # Phase 4: Cross-reference
    logger.info("=" * 60)
    logger.info("Phase 4: Cross-referencing operator skills with store")
    logger.info("=" * 60)

    final_operators = cross_reference(operators, skill_store)

    # Phase 5: Build output
    logger.info("=" * 60)
    logger.info("Phase 5: Writing output")
    logger.info("=" * 60)

    # Build name index for O(1) lookup
    by_name: dict[str, dict[str, Any]] = {}
    for op in final_operators:
        name = op["name"]
        if name:
            by_name[name] = op

    output = {
        "description": (
            "干员→基建技能映射。数据来源：PRTS Wiki（prts.wiki），"
            f"自动抓取于 {time.strftime('%Y-%m-%d')}。"
            "skill_id cross-reference base_skills.json → skills[].id"
        ),
        "source": "https://prts.wiki/w/后勤技能一览",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_operators": len(final_operators),
        "operators": final_operators,
        "by_name": by_name,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    file_size = OUT_FILE.stat().st_size
    logger.info("Output: %s", OUT_FILE)
    logger.info("  %d operators, %d total skills, %.1f KB",
                len(final_operators),
                sum(len(o["base_skills"]) for o in final_operators),
                file_size / 1024,
             )

    # Quick stats
    facilities = {}
    for op in final_operators:
        for skill in op.get("base_skills", []):
            fac = skill.get("facility", "Unknown")
            facilities[fac] = facilities.get(fac, 0) + 1
    logger.info("Skills by facility: %s",
                ", ".join(f"{k}: {v}" for k, v in sorted(facilities.items())))

    return 0


if __name__ == "__main__":
    sys.exit(main())
