"""Unit tests for skill Markdown parser."""

import pytest
from src.skills.parser import parse_skill_md


SAMPLE_SKILL = """---
name: farm-gt-6
description: "Farm GT-6 for sugar"
tags: [farm, sugar, sanity]
game: arknights
---

# Farm GT-6

## When to Use
- 需要糖

## Steps
1. adb_tap('作战')
2. adb_tap('物资筹备')
3. adb_swipe('下滑', 'find GT-6')
4. adb_tap('GT-6')

## Pitfalls
- 弹窗可能遮挡按钮
"""


def test_parse_frontmatter():
    result = parse_skill_md(SAMPLE_SKILL)
    assert result["name"] == "farm-gt-6"
    assert result["description"] == "Farm GT-6 for sugar"
    assert result["tags"] == ["farm", "sugar", "sanity"]
    assert result["game"] == "arknights"


def test_parse_steps():
    result = parse_skill_md(SAMPLE_SKILL)
    assert len(result["steps"]) == 4
    assert result["steps"][0]["tool"] == "adb_tap"
    assert result["steps"][0]["args"] == ["作战"]


def test_parse_pitfalls():
    result = parse_skill_md(SAMPLE_SKILL)
    assert len(result["pitfalls"]) == 1
    assert "弹窗" in result["pitfalls"][0]


def test_parse_empty_frontmatter():
    result = parse_skill_md("# Just a title\n\nNo frontmatter here.")
    assert result["name"] == ""
    assert result["steps"] == []


def test_parse_raw_body():
    result = parse_skill_md(SAMPLE_SKILL)
    assert "## When to Use" in result["body"]
    assert result["raw"] == SAMPLE_SKILL


def test_parse_type_field():
    """Verify type field is read from frontmatter, defaulting based on verified."""
    result = parse_skill_md(SAMPLE_SKILL)
    # verified=false, no explicit type → defaults to guide
    assert result["type"] == "guide"
    assert result["verified"] is False
