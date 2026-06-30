"""BaseScheduler — Arknights base scheduling intelligence tool.

Parses user intent, enumerates all product configurations, computes the
Pareto frontier, and produces an optimal schedule. When the user's goal
is vague ("搓玉为主也要练级"), uses LLM reasoning (via the agent loop)
to select the best tradeoff point on the frontier.

Architecture:
  - Goal parsing: regex (fast, no LLM)
  - Box parsing: regex + elite level extraction
  - Optimization: pure Python (Pareto frontier via BaseOptimizer)
  - LLM role: interpret fuzzy intent → pick best Pareto point (via user message injection)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.intelligence.base import (
    IntelligenceContext,
    IntelligenceResult,
    IntelligenceTool,
)

logger = logging.getLogger(__name__)

# ── Intent parsing patterns ────────────────────────────────────────

# (regex, goal_key, description, (orundum_w, lmd_w, combat_record_w), mood_threshold)
GOAL_PATTERNS: list[tuple[str, str, str, tuple[float, float, float], float]] = [
    (r"全力搓玉|纯搓玉|只要合成玉|最大化合成玉", "orundum_max",
     "全力搓玉", (0.70, 0.20, 0.10), 0.35),
    (r"搓玉.*也要.*练级|搓玉.*也要.*经验|搓玉.*也要.*升级|搓玉.*也要.*龙门|搓玉为主|边搓.*边练",
     "mixed_orundum_upgrade",
     "搓玉为主，兼顾练级", (0.50, 0.30, 0.20), 0.40),
    (r"搓玉|合成玉|抽卡|源石碎片", "orundum_max",
     "全力搓玉", (0.60, 0.25, 0.15), 0.35),
    (r"平衡.*练级|均衡.*发展|又要.*又要|龙门币.*经验|练级.*龙门币", "balanced",
     "均衡发展", (0.30, 0.40, 0.30), 0.50),
    (r"龙门币|钱|赤金|lmd", "lmd_max",
     "最大化龙门币", (0.05, 0.80, 0.15), 0.35),
    (r"作战记录|经验|录像带", "combat_record_max",
     "最大化作战记录", (0.05, 0.15, 0.80), 0.35),
    (r"平衡|平均|兼顾", "balanced",
     "均衡发展", (0.30, 0.40, 0.30), 0.50),
]

# Mood threshold overrides — explicit user intent
_MOOD_OVERRIDE_RE = re.compile(
    r'心情[降到留]?\s*(\d+)\s*%|保留\s*(\d+)\s*%|长班|短班|保守|激进',
)
_MOOD_KEYWORDS: dict[str, float] = {
    "保守": 0.60, "短班": 0.60,
    "长班": 0.30, "激进": 0.25,
}

LAYOUT_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b333\b|3贸易.*3制造|3制造.*3贸易", "333", "333（3贸3制3电）"),
    (r"\b243\b|2贸易.*4制造|4制造.*2贸易", "243", "243（2贸4制3电）"),
    (r"\b252\b|2贸易.*5制造|5制造.*2贸易", "252", "252（2贸5制2电）"),
    (r"\b153\b|1贸易.*5制造|5制造.*1贸易", "153", "153（1贸5制3电）"),
]


class BaseScheduler(IntelligenceTool):
    """Arknights base scheduling optimizer — two-phase interaction:

    Phase 1: User describes goal + box → comparison table + recommendation.
    Phase 2: User says "展示方案3" → full schedule detail for that solution.
    """

    # ── Session cache for two-phase flow ──
    # Keyed by thread_id so concurrent agents don't cross-contaminate.
    _cache: dict[int, dict] = {}

    @classmethod
    def _get_cache(cls) -> dict | None:
        """Get the cache for the current thread. Returns None if no cache."""
        import threading
        return cls._cache.get(threading.current_thread().ident)

    @classmethod
    def _set_cache(cls, data: dict) -> None:
        """Set the cache for the current thread."""
        import threading
        cls._cache[threading.current_thread().ident] = data

    def can_handle(self, task: str) -> bool:
        # ── Phase 2 follow-up: always allow through ──
        # "展示方案3" / "按推荐的" / "选第二个" — user is selecting from
        # previously-computed plans.  Don't block these with keyword or
        # data checks — the data was present when Phase 1 ran.
        if re.search(
            r'(?:展示方案|看看方案|查看方案|方案\s*\d+|按推荐|选推荐|推荐方案|'
            r'就这个|就它|就选这个|听你的|执行推荐|用推荐|用默认|'
            r'第\s*[一二三四五六七八九\d]+\s*个)',
            task,
        ):
            return True

        keywords = [
            "基建", "排班", "制造站", "贸易站", "发电站", "排",
            "控制中枢", "宿舍", "会客室", "搓玉", "合成玉", "龙门币",
            "作战记录", "赤金", "源石碎片", "干员分配", "布局",
            "练级", "升级", "经验", "base", "farming", "orundum", "shift",
            "方案",  # "方案" catches general mentions
        ]
        task_lower = task.lower()
        if not any(kw in task_lower for kw in keywords):
            return False

        # ── Quick shift exclusion ──
        # "换班" = daily quick shift using MAA default mode.  Do NOT run
        # the full optimizer pipeline.  The agent should call
        # base_shift_maa(mode='default') directly.
        # "排班" / "出方案" / "新方案" = full scheduler pipeline.
        _is_schedule = bool(re.search(
            r'排班|出.{0,2}方案|重新排|基建安排|重排|新排|新方案',
            task,
        ))
        if not _is_schedule:
            return False

        # ── Data gate: box data is required to produce plans.
        # Depot data is checked inside analyze() — if missing, it shows
        # a blocker telling the LLM to read the warehouse. Don't gate
        # on depot here or the blocker will never be seen.
        if not self._has_box_data():
            return False

        return True

    @staticmethod
    def _has_box_data() -> bool:
        """Check if operator box data is available (cache, session, or memory)."""
        if BaseScheduler._cache and BaseScheduler._cache.get("box"):
            return True
        try:
            from src.intelligence.arknights.base_chain import SESSION_DIR, read_box_file
            if SESSION_DIR.exists():
                for s in sorted(SESSION_DIR.iterdir(),
                                key=lambda p: p.stat().st_mtime, reverse=True):
                    if s.is_dir() and (s / "box.json").exists():
                        box = read_box_file(s.name)
                        if box and len(box) > 2:
                            return True
                        break
        except Exception:
            pass
        return False

    @staticmethod
    def _has_depot_data() -> bool:
        """Check if warehouse/depot data is available (cache or session file)."""
        if BaseScheduler._cache and BaseScheduler._cache.get("depot_stock"):
            return True
        try:
            from src.intelligence.arknights.base_chain import SESSION_DIR
            if SESSION_DIR.exists():
                for s in sorted(SESSION_DIR.iterdir(),
                                key=lambda p: p.stat().st_mtime, reverse=True):
                    if s.is_dir() and (s / "warehouse.json").exists():
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _has_fresh_box(task: str, ctx: IntelligenceContext) -> bool:
        """Check if the task explicitly provides box data (not stale auto-load).

        "排班" requires fresh scan — old session data from a previous account is
        useless.  Returns True only if the user explicitly provided box data in
        this message (inline names, JSON file, or session reference).
        """
        # Explicit session reference
        if re.search(r'session[：:\s]*(\S+)', task):
            return True
        # JSON file reference
        if re.search(r'(?:box|干员|文件)[：:\s]*(\S+\.json)', task, re.IGNORECASE):
            return True
        # Inline list with elite markers: "德克萨斯(E2)、能天使(E2)"
        elite_names = re.findall(
            r'[一-鿿]{2,4}\s*[（(]\s*(?:精英)?\s*[Ee]?\d\s*[）)]',
            task,
        )
        if len(elite_names) >= 3:
            return True
        # Inline with prefix: "我的干员: ..."
        if re.search(r'我的干员|我有|box[：:\s]|干员[：:\s]', task):
            return True
        # Scan results in this very message (agent just did scan_operator_box)
        if re.search(r'scan_depot|scan_operator_box|box.*scan|扫描结果', task):
            return True
        return False

    # ── Selection intent patterns (Chinese ordinal + recommendation selection) ──
    # Ordered from most-specific to catch-all, first match wins.
    _SELECTION_PATTERNS: list[tuple[str, int]] = [
        # "方案3" / "方案 3" — explicit plan number
        (r'方案\s*(\d+)', -1),  # -1 = use captured group
        # "第3个" / "第三个" / "第 3 个"
        (r'第\s*([一二三四五六七八九\d]+)\s*个', -1),
        # "选第一个" / "展示第一个"
        (r'[选展示]\s*第\s*([一二三四五六七八九\d]+)\s*个', -1),
    ]
    _CN_DIGITS: dict[str, str] = {
        '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
    }
    # Keywords that mean "pick the recommended one"
    _RECOMMENDED_KEYWORDS: list[str] = [
        '按推荐', '选推荐', '推荐方案', '推荐的', '推荐吧',
        '就这个', '就它', '就选这个', '就选它', '听你的',
        '执行推荐', '用推荐', '用默认',
    ]

    def _match_selection(self, task: str) -> int | None:
        """Extract plan index from user selection intent. Returns 0-based index or None."""
        for pattern, group in self._SELECTION_PATTERNS:
            m = re.search(pattern, task)
            if m:
                if group == -1:
                    raw = m.group(1)
                    raw = self._CN_DIGITS.get(raw, raw)
                    try:
                        return max(0, int(raw) - 1)
                    except ValueError:
                        return None
                return group
        for kw in self._RECOMMENDED_KEYWORDS:
            if kw in task:
                return 0  # best / recommended
        return None

    def analyze(self, ctx: IntelligenceContext, task: str) -> IntelligenceResult | None:
        from src.intelligence.arknights.base_optimizer import BaseOptimizer, ParetoSolution

        # ── Phase 2: user selected a specific plan ──
        plan_index = self._match_selection(task)
        if plan_index is not None:
            return self._phase2_show_solution(plan_index)

        # ── New schedule vs daily shift ──
        # "排班" = create a NEW schedule → MUST re-scan box + depot.
        # "换班" = daily shift using existing plan → use cache fine.
        _is_new_schedule = bool(re.search(
            r'排班|出方案|重新排|基建安排|重排|新排|新方案',
            task,
        )) and not re.search(r'换班|轮换|换岗', task)

        if _is_new_schedule and not self._has_fresh_box(task, ctx):
            layout, layout_desc = self._parse_layout(task, "")
            goal_key, goal_desc, _, _ = self._parse_goal(task)
            return IntelligenceResult(
                recommendation=(
                    f"检测到基建排班需求（目标：**{goal_desc}**），但系统需要当前账号的最新数据。\n\n"
                    "请按顺序执行：\n"
                    "1. **先读仓库** — 调用 scan_depot() 获取赤金/源岩/源石碎片库存\n"
                    "2. **再扫 Box** — 导航到干员列表 → 排序等级 → 展开职业筛选 → scan_operator_box()\n"
                    "3. 两个都完成后系统会自动计算最优排班方案\n\n"
                    "⛔ 不要直接调 base_shift_maa——那不是用来排班的，是换班用的。"
                ),
                confidence=0.9,
                source="knowledge",
            )

        # ── Phase 1: full optimization ──
        # 1. Parse goal with weights
        goal_key, goal_desc, weights, mood_threshold = self._parse_goal(task)

        # 2. Parse layout
        layout, layout_desc = self._parse_layout(task, goal_key)

        # 3. Parse resource inventory
        from src.intelligence.arknights.base_optimizer import parse_inventory
        inventory = parse_inventory(task)

        # 4. Extract operator box
        operator_box = self._extract_box(task, ctx)
        if not operator_box:
            return IntelligenceResult(
                recommendation=(
                    f"检测到基建排班需求（目标：**{goal_desc}**），但未识别到您的干员列表。\n\n"
                    "请通过以下方式提供干员box：\n"
                    "  1. **直接列出**：如「我的干员：德克萨斯(E2)、能天使(E2)、清流(E1)...」\n"
                    "  2. **JSON文件**：如「box文件：my_box.json」\n"
                    "  3. **全box截图**：说「从截图识别box」让我OCR识别\n\n"
                    "💡 支持标注精英等级：德克萨斯(E2) 表示精英2级\n"
                    f"推荐布局：{layout_desc}"
                ),
                confidence=0.3,
                source="knowledge",
            )

        # 5. Run Pareto optimization
        try:
            optimizer = BaseOptimizer(ctx.knowledge)
            depot_stock = (BaseScheduler._get_cache() or {}).get("depot_stock")

            # ── Auto-load depot stock from latest chain session ──
            # When the user hasn't explicitly scanned their warehouse, try
            # loading from a previous session's scan result.  This lets the
            # scheduler use real inventory data even on first use.
            from src.intelligence.arknights.base_chain import SESSION_DIR
            if depot_stock is None and SESSION_DIR.exists():
                sessions = sorted(
                    [s for s in SESSION_DIR.iterdir()
                     if s.is_dir() and (s / "warehouse.json").exists()],
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                if sessions:
                    try:
                        import json
                        wh_path = sessions[0] / "warehouse.json"
                        wh_data = json.loads(wh_path.read_text(encoding="utf-8"))
                        from src.games.arknights.operators import MaterialStock
                        depot_stock = MaterialStock(
                            items=wh_data.get("items", {}),
                            lmd=wh_data.get("lmd", 0),
                            scanned_at=wh_data.get("scanned_at", ""),
                        )
                        logger.info(
                            "Auto-loaded depot stock from %s: %d items, LMD=%d",
                            sessions[0].name, len(depot_stock.items), depot_stock.lmd,
                        )
                    except Exception as e:
                        logger.debug("Could not auto-load depot stock: %s", e)

            # ── Inventory-aware weight adjustment ──
            # Adjust goal weights based on warehouse stock so the recommended
            # plan favors what the player actually needs.
            from src.intelligence.arknights.base_optimizer import BaseOptimizer as BO
            inv_sort = BO._inventory_sort_adjustment(inventory, depot_stock)
            adj_weights = (
                weights[0] * inv_sort[0],
                weights[1] * inv_sort[1],
                weights[2] * inv_sort[2],
            )

            frontier = optimizer.solve_pareto(operator_box, layout, num_shifts=0,
                                               sort_weights=adj_weights,
                                               inventory=inventory,
                                               mood_threshold=mood_threshold,
                                               material_stock=depot_stock)

            # Use knee-point for balanced/mixed, weighted-sum for extreme goals
            if goal_key in ("balanced", "mixed_orundum_upgrade"):
                best = optimizer.solve_balanced(frontier)
            else:
                best = optimizer.solve_with_weights(frontier, *adj_weights)

        except Exception as e:
            logger.error("Optimization failed: %s", e, exc_info=True)
            return IntelligenceResult(
                recommendation=f"排班计算失败: {e}\n请检查干员名称是否正确。",
                confidence=0.0,
                source="knowledge",
            )

        if not best or not frontier:
            return IntelligenceResult(
                recommendation="在当前布局下，您的干员box无法填满任何配置。"
                               "建议尝试更小的布局或扩充box。",
                confidence=0.1,
                source="knowledge",
            )

        # Guard: all-zero frontier — happens when operator names don't match
        # any skills in the KB or curated data (e.g. heuristic matched common
        # words as fake operator names).
        _ZERO_THRESHOLD = 0.005
        if all(
            s.orundum_eff <= _ZERO_THRESHOLD
            and s.lmd_eff <= _ZERO_THRESHOLD
            and s.combat_record_eff <= _ZERO_THRESHOLD
            for s in frontier
        ):
            resolved_names = ", ".join(list(operator_box.keys())[:8])
            return IntelligenceResult(
                recommendation=(
                    f"检测到 {len(operator_box)} 个干员名称"
                    f"（{resolved_names}），"
                    f"但无法在数据库中匹配到任何基建技能。\n\n"
                    "这通常是因为：\n"
                    "1. **干员名称识别错误** — 请用标准中文名直接列出，"
                    "如：「我的干员：德克萨斯(E2)、清流(精英1)、能天使(E2)」\n"
                    "2. **格式不完整** — 确保精英等级已标注，部分技能需精1/精2解锁\n"
                    "3. **AI接收到了错误文本** — 重新发送完整的干员名单试试\n\n"
                    "💡 确认干员列表后，说「**重新排班**」重新计算。"
                ),
                confidence=0.05,
                source="knowledge",
            )

        operators_resolved = optimizer._resolve_operators(operator_box)

        # ── Cache for Phase 2 follow-up (class-level so base_shift_tool can read it) ──
        # Preserve depot_stock across cache rebuilds — it's set by the depot
        # scan tool and should survive Phase 2 lookups.
        _existing_depot = (BaseScheduler._get_cache() or {}).get("depot_stock")
        if depot_stock is not None:
            _existing_depot = depot_stock  # use newly-loaded data
        BaseScheduler._set_cache({
            "frontier": frontier,
            "best": best,
            "box": operator_box,
            "goal_desc": goal_desc,
            "goal_key": goal_key,
            "layout_desc": layout_desc,
            "operators": operators_resolved,
            "inventory": inventory,
            "depot_stock": _existing_depot,
        })

        # ── Count feasible solutions (for interactive selection) ──
        from src.intelligence.arknights.base_optimizer import BaseOptimizer
        _feasible = []
        for fi, s in enumerate(frontier):
            bal = BaseOptimizer.check_resource_balance(s, inventory, depot_stock=depot_stock)
            if not bal["warnings"]:
                _feasible.append((fi, s))

        # ── Depot scan blocker ──────────────────────────────────────
        # When there is no warehouse data, ALL plan suggestions are blind
        # guesses.  Return ONLY the blocker — NO comparison table, NO plan
        # details, NO schedule.  If the LLM sees plan data it will try to
        # be "helpful" and recommend one instead of scanning.
        if depot_stock is None:
            logger.info(
                "BaseScheduler: depot_stock missing — "
                "returning depot blocker (no resource data)"
            )
            return IntelligenceResult(
                recommendation=(
                    "## 🛑 缺少仓库数据，无法精准排班\n\n"
                    f"已加载 {len(operator_box)} 名干员，但**没有仓库库存数据**"
                    "（龙门币、赤金、固源岩数量未知）。\n\n"
                    "### 现在必须做这件事，不要做其他任何事：\n\n"
                    "**调用 `scan_depot()` 一键扫描仓库资源。**\n\n"
                    "不要 ask_user。不要分析方案。不要给建议。先 scan_depot。"
                ),
                confidence=0.60,
                source="knowledge",
            )

        # ── Phase 1 output: comparison table + best-plan detail ──
        comparison = self._format_phase1(
            best, frontier, operator_box, goal_desc, layout_desc,
            operators_resolved, inventory, goal_key=goal_key,
            depot_stock=depot_stock,
        )
        best_detail = self._format_solution_detail(
            best, 1, operator_box, operators_resolved, inventory,
        )
        recommendation = (
            comparison
            + "\n---\n"
            + "## ⭐ 推荐方案完整排班\n"
            + best_detail
        )

        # Only recommend SUSTAINABLE plans (closed-loop LMD balance ≥ 0).
        _sustainable = [(fi, s) for fi, s in _feasible if s.sustain_verdict == "ok"]
        _unsustainable = [(fi, s) for fi, s in _feasible if s.sustain_verdict != "ok"]

        if len(_sustainable) >= 1:
            # ── Filter to goal-relevant plans ──
            _goal_relevant: list = []
            _goal_irrelevant: list = []
            if goal_key in ("orundum_max", "mixed_orundum_upgrade"):
                for fi, s in _sustainable[:8]:
                    if s.daily_orundum > 0:
                        _goal_relevant.append((fi, s))
                    else:
                        _goal_irrelevant.append((fi, s))
                if not _goal_relevant:
                    _goal_relevant = _sustainable[:8]
                    _goal_irrelevant = []
            else:
                _goal_relevant = _sustainable[:8]

            # Find the recommended plan's actual frontier index.
            best_bal = BaseOptimizer.check_resource_balance(best, inventory, depot_stock=depot_stock)
            if best_bal["warnings"] and _goal_relevant:
                rec_plan = _goal_relevant[0][1]
            elif best.sustain_verdict != "ok" and _goal_relevant:
                rec_plan = _goal_relevant[0][1]
            else:
                rec_plan = best

            plan_options = []
            plan_index_map: dict[int, int] = {}
            rec_display_num = 1
            for display_i, (fi, s) in enumerate(_goal_relevant):
                display_num = display_i + 1
                plan_index_map[display_i] = fi
                if s is rec_plan:
                    rec_display_num = display_num

                def _fmt_orundum(v: float) -> str:
                    if v <= 0: return "—"
                    if v < 100: return f"{v:.0f}/d"
                    return f"{v/1000:.1f}k/d"
                def _fmt_lmd(v: float) -> str:
                    if v <= 0: return "—"
                    if v < 10000: return f"{v:.0f}/d"
                    return f"{v/10000:.2f}w/d"
                def _fmt_cr(v: float) -> str:
                    if v <= 0: return "—"
                    return f"{v:.1f}/d"

                config_str = self._short_config(s.config)
                team_lines = self._summarize_teams(s)
                op_str = "、".join(team_lines[:3]) if team_lines else "—"

                net_tag = ""
                if s.sustain_verdict != "ok":
                    net_tag = " 🔴不可持续"

                plan_options.append(
                    f"方案#{display_num} | ⚙{config_str} | "
                    f"合成玉{_fmt_orundum(s.daily_orundum)} "
                    f"龙门币{_fmt_lmd(s.daily_lmd)} "
                    f"作战记录{_fmt_cr(s.daily_combat_record)}"
                    f"{net_tag} | {op_str}"
                )
            extra_note = ""
            if _unsustainable:
                extra_note = (
                    f"\n\n> ⚠️ 另有 {len(_unsustainable)} 个不可持续方案已自动排除"
                    f"（LMD亏空/理智不足）。"
                )
            if _goal_irrelevant:
                extra_note += (
                    f"\n> 💡 另有 {len(_goal_irrelevant)} 个纯龙门币方案"
                    f"（无合成玉产出），说「显示全部方案」可查看。"
                )
            # ── Build cultivation summary for injection ──
            # Use the actual recommended plan (rec_plan from selection logic)
            _cult_plan = rec_plan if rec_plan else best
            cult = self._get_cultivation_suggestions(
                operator_box, operators_resolved, _cult_plan,
                frontier=frontier, material_stock=depot_stock,
            )
            cult_summary = ""
            if cult:
                skill_suggestions = [c for c in cult if "★★" in c[3]][:8]
                if skill_suggestions:
                    import re as _re_cult
                    lines = ["| 干员 | 当前 | 升级 | 产出提升 | 可进方案 | 替换 | 费用 |"]
                    lines.append("|------|------|------|----------|----------|------|------|")
                    for s in skill_suggestions:
                        name, cur, target, benefit = s
                        # "★★ +3500龙门币/d — ✅可执行 → 进⭐方案#1 Trade 替XX (已入选)"
                        m_delta = _re_cult.search(r'\+\S+/d|\+\S+%无人机', benefit)
                        delta_str = m_delta.group(0) if m_delta else "?"
                        m_plan = _re_cult.search(r'方案#\d+', benefit)
                        plan_str = m_plan.group(0) if m_plan else "?"
                        m_role = _re_cult.search(r'(?:已入选|可替补)', benefit)
                        role_str = m_role.group(0) if m_role else "—"
                        # Cost is between " — " and "→"
                        cost_str = benefit.split(" — ")[-1].split(" →")[0] if " — " in benefit else "—"
                        lines.append(f"| {name} | {cur} | {target} | {delta_str} | {plan_str} | {role_str} | {cost_str} |")
                    cult_summary = "\n".join(lines)

            # Save plan_index mapping for _write_custom_plan resolution
            _cache = BaseScheduler._get_cache() or {}
            _cache["plan_index_map"] = plan_index_map
            BaseScheduler._set_cache(_cache)

            # ── Replace recommendation: use plan_options as the single list.
            # Two numbering systems (comparison table raw fi vs. plan_options
            # sequential 1..N) side by side confuses the LLM. Drop the old.
            wh = self._format_warehouse_context(depot_stock, inventory)
            rec_header = (
                (wh + "\n---\n\n" if wh else "") +
                f"## 🏗️ 基建排班 — 方案对比\n\n"
                f"- **目标**: {goal_desc}\n"
                f"- **布局**: {layout_desc}\n"
                f"- **干员数**: {len(operator_box)} 名\n"
                f"- **可用方案**: {len(_goal_relevant)} 个\n\n"
                "## ⭐ 推荐方案完整排班\n" +
                self._format_solution_detail(
                    rec_plan, rec_display_num, operator_box, operators_resolved, inventory)
            )
            recommendation = (
                rec_header +
                "\n\n---\n"
                "## 🛑 停！在用户选择之前不要执行任何操作！\n\n"
                "**你的下一步 MUST 是调用 ask_user() 工具。** 不要点击弹窗、不要导航、不要启动游戏。\n\n"
                f"**在 ask_user 中逐字复制以下方案列表**（每个方案已包含产品配置+产出+干员团队，不要自己编造）：\n\n"
                f"**{len(_goal_relevant)} 个可选方案：**\n\n" +
                "\n".join(f"- **{p}**" for p in plan_options) +
                extra_note +
                (f"\n\n### 🎓 培养建议\n{cult_summary}" if cult_summary else "") +
                f"\n\n推荐 ⭐方案#{rec_display_num}。"
                f"用户选方案N → base_shift_maa(mode='custom', plan_index=N-1)。"
                f"「按推荐的」→ plan_index={rec_display_num - 1}。"
            )
        elif len(_feasible) > 1:
            # All feasible plans are unsustainable — warn user
            plan_options = []
            for fi, s in _feasible[:8]:
                config_str = self._short_config(s.config)
                def _fmt_orundum(v: float) -> str:
                    if v <= 0: return "—"
                    if v < 100: return f"{v:.0f}/d"
                    return f"{v/1000:.1f}k/d"
                def _fmt_lmd(v: float) -> str:
                    if v <= 0: return "—"
                    if v < 10000: return f"{v:.0f}/d"
                    return f"{v/10000:.2f}w/d"
                plan_options.append(
                    f"方案#{fi+1} | ⚙{config_str} | "
                    f"合成玉{_fmt_orundum(s.daily_orundum)} "
                    f"龙门币{_fmt_lmd(s.daily_lmd)}"
                    f" 🔴 不可持续"
                )
            recommendation += (
                "\n\n---\n"
                f"## 🛑 警告！所有方案均不可持续\n\n"
                f"当前无方案能实现 LMD 自给自足。建议调整目标（如降低搓玉比例）或先攒龙门币。\n\n"
                f"以下是 {len(_feasible)} 个方案供参考（均需外部输血）：\n\n"
                + "\n".join(f"- **{p}**" for p in plan_options)
            )
        else:
            # Only 1 feasible solution — auto-execute
            recommendation += (
                "\n\n---\n"
                "> ⚡ 仅1个可行方案，可直接执行 base_shift_maa(mode='custom', plan_index=0)。"
            )

        confidence = min(0.95, 0.5 + 0.5 * (best.coverage if best else 0))

        logger.info(
            "BaseScheduler phase1: %d ops, goal=%s, layout=%s, frontier=%d points",
            len(operator_box), goal_key, layout, len(frontier),
        )

        return IntelligenceResult(
            recommendation=recommendation,
            confidence=confidence,
            source="knowledge",
        )

    # ── Parsing ──────────────────────────────────────────────────

    def _parse_goal(
        self, task: str,
    ) -> tuple[str, str, tuple[float, float, float], float]:
        for pattern, key, desc, w, mood in GOAL_PATTERNS:
            if re.search(pattern, task):
                mood = self._override_mood(task, mood)
                return key, desc, w, mood
        mood = self._override_mood(task, 0.35)
        return "orundum_max", "全力搓玉（默认）", (0.60, 0.25, 0.15), mood

    def _override_mood(self, task: str, default: float) -> float:
        """Extract explicit mood threshold from user text.

        Supports: "心情降到30%", "保留40%心情", "长班", "保守"
        """
        m = _MOOD_OVERRIDE_RE.search(task)
        if m:
            if m.group(1):
                return int(m.group(1)) / 100.0
            if m.group(2):
                return int(m.group(2)) / 100.0
            if m.group(0) in _MOOD_KEYWORDS:
                return _MOOD_KEYWORDS[m.group(0)]
        return default

    def _parse_layout(self, task: str, goal_key: str) -> tuple[str, str]:
        for pattern, layout, desc in LAYOUT_PATTERNS:
            if re.search(pattern, task):
                return layout, desc
        return "243", "243（2贸4制3电）"

    def _extract_box(self, task: str, ctx: IntelligenceContext) -> dict[str, int | dict]:
        """Extract {operator_name: elite_level | {elite,level,...}} from user input.

        Parses:
          - JSON file: "box文件：my_box.json"
          - Chain session: "session:base_20260605_120000" → reads box.json
          - Inline text: "德克萨斯(E2)、清流(E2)"
        """
        # Pattern 1: Chain session reference
        session_match = re.search(r'session[：:\s]*(\S+)', task)
        if session_match:
            sid = session_match.group(1).strip().rstrip('/')
            from src.intelligence.arknights.base_chain import read_box_file
            box = read_box_file(sid)
            if box and len(box) > 0:
                logger.info("Loaded %d operators from chain session %s", len(box), sid)
                return box

        # Pattern 2: JSON file
        json_match = re.search(r'(?:box|干员|文件)[：:\s]*(\S+\.json)', task, re.IGNORECASE)
        if json_match:
            return self._load_box_json(json_match.group(1))

        # Pattern 3: Inline list with elite levels
        for prefix in [r'我的干员[：:\s]*', r'我有[：:\s]*', r'box[：:\s]*', r'干员[：:\s]*']:
            inline = re.search(prefix + r'(.+?)(?:[。\n]|$)', task)
            if inline:
                return self._parse_box_text(inline.group(1))

        # Pattern 4a: Heuristic — names with explicit elite markers
        # Only fires when the user provides elite-tagged names (E0/E1/E2/精英1 etc).
        # This avoids matching common words like "根据", "给我" as fake operators.
        elite_names = re.findall(r'[一-鿿]{2,4}\s*[（(]\s*(?:精英)?\s*[Ee]?\d\s*[）)]', task)
        if len(elite_names) >= 3:
            return self._parse_box_text("、".join(elite_names))

        # Pattern 4b: Bare Chinese text — only match if we have MANY names AND
        # most of them are known operators (validated against KB).
        # Otherwise session auto-load takes precedence.
        bare_names = re.findall(r'[一-鿿]{2,4}(?:\([Ee]?\d\)|（(?:精英)?\d）)?', task)
        if len(bare_names) >= 20:
            # Check how many of these are actually known operators
            known_count = 0
            try:
                if ctx and ctx.knowledge:
                    for n in bare_names[:50]:
                        n_clean = re.sub(r'[（(].*[）)]', '', n)
                        data = ctx.knowledge.get(
                            "arknights", "operator_base_skills",
                            key=n_clean, key_field="name",
                        )
                        if data:
                            known_count += 1
            except Exception:
                pass
            if known_count >= 10:
                return self._parse_box_text("、".join(bare_names))

        # Pattern 5: Auto-detect latest chain session
        from pathlib import Path
        from src.intelligence.arknights.base_chain import SESSION_DIR, read_box_file
        if SESSION_DIR.exists():
            sessions = sorted(SESSION_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for s in sessions:
                if s.is_dir() and (s / "box.json").exists():
                    box = read_box_file(s.name)
                    if box:
                        # Guard: a box with ≤2 operators is likely stale/incomplete.
                        # Real box scans produce 50+ operators. Skip sessions that
                        # were created with corrupt/empty data from a previous run.
                        if len(box) <= 2:
                            logger.warning(
                                "Skipping session %s — only %d operators (stale/corrupt)",
                                s.name, len(box),
                            )
                            continue
                        logger.info("Auto-loaded %d operators from latest session %s", len(box), s.name)
                        return box
                    break

        # Pattern 6: Search memory DB for recent box data
        # When a user previously scanned their box, it's saved as a memory.
        # Look for memories containing operator lists with elite markers.
        try:
            from src.memory.memory_db import memory_db
            from src.memory.fts5_utils import build_search_terms, safe_fts5_term

            terms = build_search_terms("box 干员 E2 E1")
            if terms:
                safe_terms = [safe_fts5_term(t) for t in terms]
                safe_terms = [t for t in safe_terms if t]
                if safe_terms:
                    fts5_query = ' OR '.join(safe_terms)
                    rows = memory_db.conn.execute(
                        """SELECT m.body, m.created
                           FROM memories_fts f
                           JOIN memories_data m ON f.rowid = m.id
                           WHERE memories_fts MATCH ? AND m.game = 'arknights'
                             AND m.deleted_at IS NULL
                           ORDER BY f.rank
                           LIMIT 5""",
                        (fts5_query,),
                    ).fetchall()

                    for row in rows:
                        body = row["body"] or ""
                        # Try to parse as inline box text
                        names_with_elite = re.findall(
                            r'[一-鿿]{2,4}\s*[（(]\s*(?:精英)?\s*[Ee]?\d\s*[）)]',
                            body,
                        )
                        if len(names_with_elite) >= 3:
                            logger.info(
                                "Loaded %d operators from memory DB (created %s)",
                                len(names_with_elite), row["created"],
                            )
                            return self._parse_box_text("、".join(names_with_elite))

                        # Try parsing formatted box dumps: "E2 LV90: 维什戴尔、酒神"
                        e2_match = re.findall(r'[Ee](\d)\s*(?:LV\d+\s*)?[:：]\s*([一-鿿]{2,4}(?:、[一-鿿]{2,4})*)', body)
                        if e2_match:
                            parsed: dict[str, int] = {}
                            for e_str, names_str in e2_match:
                                elite = int(e_str)
                                for name in re.split(r'[、，,\s]+', names_str):
                                    name = name.strip()
                                    if name and len(name) >= 2:
                                        parsed[name] = max(parsed.get(name, 0), elite)
                            if len(parsed) >= 3:
                                logger.info(
                                    "Loaded %d operators from memory DB formatted dump",
                                    len(parsed),
                                )
                                return parsed
        except Exception as e:
            logger.debug("Memory DB box search skipped: %s", e)

        return {}

    @staticmethod
    def _parse_box_text(text: str) -> dict[str, int]:
        """Parse operator list — delegates to shared parser in base_optimizer."""
        from src.intelligence.arknights.base_optimizer import parse_operator_box
        return parse_operator_box(text)

    def _load_box_json(self, filepath: str) -> dict[str, int | dict]:
        import json
        path = Path(filepath)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ops = data.get("operators", data.get("box", []))
            if isinstance(ops, dict):
                # {name: {elite, level, ...}} or {name: elite_int}
                return ops
            # Legacy list format: [{"name": "银灰", "elite": 2}, ...]
            result: dict[str, int | dict] = {}
            for op in ops:
                if isinstance(op, dict):
                    name = op.get("name", "")
                    if name:
                        result[name] = {
                            "elite": op.get("elite", op.get("elite_level", 2)),
                            "level": op.get("level", 1),
                            "potential": op.get("potential", 0),
                            "rarity": op.get("rarity", 0),
                        }
                elif isinstance(op, str):
                    result[op] = 2
            return result
        except Exception as e:
            logger.error("Failed to load box JSON: %s", e)
            return {}

    # ── Output formatting ────────────────────────────────────────

    # ── Phase 1: comparison table + recommendation (no schedule detail) ──

    def _format_phase1(
        self,
        best,
        frontier: list,
        box: dict[str, int],
        goal_desc: str,
        layout_desc: str,
        operators=None,
        inventory=None,
        goal_key: str = "",
        depot_stock: 'MaterialStock | None' = None,
    ) -> str:
        """Phase 1: comparison table of feasible options + operator improvement tips."""
        from src.intelligence.arknights.base_optimizer import BaseOptimizer

        # Goal relevance masks — a "搓玉" user shouldn't see LMD-only configs.
        # Only solutions that make sense for the stated goal are shown.
        _GOAL_MASKS: dict[str, dict] = {
            "orundum_max":         {"orundum_min": 0.01, "label": "搓玉"},
            "mixed_orundum_upgrade": {"orundum_min": 0.01, "label": "搓玉+练级"},
            "lmd_max":             {"lmd_min": 0.01, "label": "龙门币"},
            "combat_record_max":   {"cr_min": 0.01, "label": "作战记录"},
            "balanced":            {},  # show everything
        }
        mask = _GOAL_MASKS.get(goal_key, {"orundum_min": 0.01})

        # Only show feasible, goal-relevant configs
        feasible: list[tuple[int, object]] = []
        for fi, s in enumerate(frontier):
            bal = BaseOptimizer.check_resource_balance(s, inventory, depot_stock=depot_stock)
            if bal["warnings"]:
                continue
            if mask.get("orundum_min") and s.orundum_eff < mask["orundum_min"]:
                continue
            if mask.get("lmd_min") and s.lmd_eff < mask["lmd_min"]:
                continue
            if mask.get("cr_min") and s.combat_record_eff < mask["cr_min"]:
                continue
            feasible.append((fi, s))

        # If recommended solution is infeasible, pick the closest feasible one
        best_bal = BaseOptimizer.check_resource_balance(best, inventory, depot_stock=depot_stock)
        if best_bal["warnings"] and feasible:
            # Find feasible solution closest to the theoretical best
            def _dist_from_best(s: object) -> float:
                return (
                    abs(s.orundum_eff - best.orundum_eff)
                    + abs(s.lmd_eff - best.lmd_eff)
                    + abs(s.combat_record_eff - best.combat_record_eff)
                )
            display_best = min((s for _, s in feasible), key=_dist_from_best)
        else:
            display_best = best

        # ── Warehouse context block (rich, multi-item) ──
        warehouse_ctx = self._format_warehouse_context(depot_stock, inventory)
        has_warehouse = bool(warehouse_ctx)

        lines = [
            f"## 🏗️ 基建排班 — 方案对比",
            f"",
        ]

        if has_warehouse:
            lines.append(warehouse_ctx)
            lines.append("---")
            lines.append("")

        lines.extend([
            f"- **目标**: {goal_desc}",
            f"- **布局**: {layout_desc}",
            f"- **干员数**: {len(box)} 名",
            f"- **可用方案**: {len(feasible)} 个",
            f"",
        ])

        if not feasible:
            lines.append("⚠️ 当前无可完全可行的方案。请尝试调整目标或补充干员。")
            return "\n".join(lines)

        # ── Comparison table ──
        # Split into sustainable vs unsustainable
        sustainable = [(fi, s) for fi, s in feasible if s.sustain_verdict == "ok"]
        unsustainable = [(fi, s) for fi, s in feasible if s.sustain_verdict != "ok"]

        # If recommended is unsustainable, pick best sustainable as display_best
        if display_best.sustain_verdict != "ok" and sustainable:
            def _dist_from_best_sus(s: object) -> float:
                return (
                    abs(s.orundum_eff - best.orundum_eff)
                    + abs(s.lmd_eff - best.lmd_eff)
                    + abs(s.combat_record_eff - best.combat_record_eff)
                )
            display_best = min((s for _, s in sustainable), key=_dist_from_best_sus)

        lines.append("| # | 合成玉/天 | 龙门币/天 | 作战记录/天 | 产品配置 | 关键干员/团队 | 闭环 + 库存 |")
        lines.append("|---|-----------|-----------|-------------|----------|--------------|------------|")

        # Show sustainable plans first, then unsustainable with warnings
        shown_sustainable = 0
        for fi, s in (sustainable + unsustainable)[:8]:
            marker = "⭐" if s is display_best else f"#{fi+1}"
            config_str = self._short_config(s.config)
            team_lines = self._summarize_teams(s)
            op_summary = "、".join(team_lines[:2]) if team_lines else "-"

            # Supply chain + sustainability indicator
            bal = BaseOptimizer.check_resource_balance(s, inventory, depot_stock=depot_stock)
            sr = bal.get("stone_ratio", 1.0)
            gr = bal.get("gold_ratio", 1.0)
            od = sum(1 for f, p in s.config.rooms if p == "Orundum")
            lm = sum(1 for f, p in s.config.rooms if p == "LMD")

            chain_parts = []
            # ── Warehouse-aware stockpile analysis ──
            stock_note = ""
            if depot_stock and not depot_stock.is_empty():
                # Gold stockpile → how many days of LMD trade
                gold = depot_stock.get_any("赤金")
                gold_daily = bal.get("gold_daily_deficit", 0)
                if lm > 0 and gold > 0 and gold_daily > 0:
                    gold_days = gold / gold_daily
                    if gold_days < 3:
                        stock_note += f" 金仅{gold_days:.0f}d"
                    elif gold_days < 14:
                        stock_note += f" 金{gold_days:.0f}d"
                # Rock stockpile → how many days of orundum trade
                rocks = depot_stock.get_any("固源岩")
                orundum_orders = s.daily_orundum / 20.0 if s.daily_orundum > 0 else 0
                rock_daily = orundum_orders * 4  # 2 stones/order × 2 rocks/stone
                if rock_daily > 0 and rocks > 0:
                    rock_days = rocks / rock_daily
                    if rock_days < 3:
                        stock_note += f" 石仅{rock_days:.0f}d"
                    elif rock_days < 14:
                        stock_note += f" 石{rock_days:.0f}d"

            if od > 0:
                chain_parts.append(f"石{sr:.0%}" if sr < 1.0 else "石✓")
            if lm > 0:
                chain_parts.append(f"金{gr:.0%}" if gr < 1.0 else "金✓")

            # Sustainability tag
            if s.sustain_verdict == "ok":
                if od > 0:
                    chain_parts.append(f"💰{s.sustain_lmd_balance:+.0f}/d")
                chain_icon = " ".join(chain_parts) if chain_parts else "✅"
            elif s.sustain_verdict == "lmd_deficit":
                chain_parts.append(f"🔴亏{abs(s.sustain_lmd_balance):.0f}/d")
                chain_icon = " ".join(chain_parts)
            elif s.sustain_verdict == "both":
                chain_parts.append(f"🔴🔴双亏")
                chain_icon = " ".join(chain_parts)
            else:
                chain_icon = " ".join(chain_parts) if chain_parts else "⚠️"

            # Combine chain icon with stock note
            full_chain = chain_icon + stock_note

            # Format daily output with readable units
            def _fmt_orundum(v: float) -> str:
                if v <= 0: return "—"
                if v < 100: return f"~{v:.0f}"
                return f"~{v/1000:.1f}k"
            def _fmt_lmd(v: float) -> str:
                if v <= 0: return "—"
                if v < 10000: return f"~{v:.0f}"
                return f"~{v/10000:.2f}w"
            def _fmt_cr(v: float) -> str:
                if v <= 0: return "—"
                return f"~{v:.1f}"

            lines.append(
                f"| {marker} | {_fmt_orundum(s.daily_orundum)} | "
                f"{_fmt_lmd(s.daily_lmd)} | {_fmt_cr(s.daily_combat_record)} | "
                f"{config_str} | "
                f"{op_summary} | {full_chain} |"
            )
            if s.sustain_verdict == "ok":
                shown_sustainable += 1
        lines.append(f"⭐ = 推荐方案（偏好「{goal_desc}」）")
        lines.append("")
        # ── All rooms accounted for verification ──
        # Count rooms for the recommended plan and list them explicitly so
        # the LLM (and user) can see all rooms are filled.
        best_room_desc = self._describe_all_rooms(display_best)
        if best_room_desc:
            lines.append(f"**推荐方案完整房间**（{best_room_desc[0]}间，全部排满无空缺）：")
            lines.append(best_room_desc[1])
            lines.append("")

        # ── Warehouse-aware recommendation ──
        if depot_stock and not depot_stock.is_empty():
            warehouse_verdict = self._format_warehouse_recommendation(
                depot_stock, display_best, goal_key)
            if warehouse_verdict:
                lines.append("")
                lines.append(warehouse_verdict)

        # ── Unsustainable plan warnings ──
        if unsustainable:
            lines.append("")
            lines.append("### 🔴 闭环警告 — 以下方案不可持续")
            lines.append("")
            for fi, s in unsustainable[:4]:
                verdict_label = {
                    "lmd_deficit": f"日亏 {abs(s.sustain_lmd_balance):.0f} 龙门币",
                    "sanity_impossible": f"需 {s.sustain_sanity_cost:.0f} 理智/天刷1-7",
                    "both": f"日亏 {abs(s.sustain_lmd_balance):.0f} LMD + 需 {s.sustain_sanity_cost:.0f} 理智",
                }.get(s.sustain_verdict, "不可持续")
                lines.append(
                    f"- **方案#{fi+1}**: {verdict_label}。"
                    f"搓玉的成本(200LMD/石+2固源岩/石)超出产出，"
                    f"需要外部输血才能维持。"
                )
            lines.append("")
            lines.append(
                "> 💡 搓玉的隐含成本：每搓1个源石碎片需要 **200龙门币 + 2个固源岩**。"
                "纯搓玉方案虽然合成玉产出高，但龙门币会持续失血。"
                "**兼顾龙门币产出的方案才能真正闭环循环。**"
            )
            lines.append("")

        lines.append(f"👉 选择方案说「**展示方案N**」查看完整排班表。")
        lines.append("")

        # ── Orundum chain check ──
        # When the user wants orundum (搓玉), verify the recommended config
        # actually has the OriginStone→Orundum production chain.
        has_orundum_trade = any(
            f == "Trade" and p == "Orundum" for f, p in display_best.config.rooms
        )
        has_originium_mfg = any(
            f == "Mfg" and p == "OriginStone" for f, p in display_best.config.rooms
        )
        is_orundum_goal = "搓玉" in goal_desc or "orundum" in goal_desc.lower()
        if is_orundum_goal and not has_originium_mfg:
            lines.append("### ⚠️ 方案优化提示")
            if has_orundum_trade:
                lines.append("- 当前推荐方案包含合成玉贸易站，但**缺少源石碎片制造**——合成玉需要源石碎片作为原料")
            else:
                lines.append("- 当前推荐方案**不含源石碎片制造→合成玉供应链**")
            lines.append("- 搓玉方案需要：制造站产**源石碎片** + 贸易站产**合成玉**，二者缺一不可")
            lines.append("- 如果看不到搓玉方案的选项，通常是Box中缺少有源石碎片制造技能的干员")
            lines.append("- 确保「基建排班」相关干员已录入Box后，说「**展示方案N**」可以查看")
            lines.append("")

        # ── Inventory status ──
        display_bal = BaseOptimizer.check_resource_balance(display_best, inventory, depot_stock=depot_stock)
        has_stockpile = inventory and not inventory.is_empty()
        if has_stockpile:
            lines.append("### 📦 库存状态")
            if inventory.puregold > 0:
                lines.append(f"- 赤金: **{inventory.puregold}个**")
            if inventory.origin_stone > 0:
                lines.append(f"- 源石碎片: **{inventory.origin_stone}个**")
            gsd = display_bal.get("gold_stockpile_days")
            ssd = display_bal.get("stone_stockpile_days")
            if gsd is not None:
                lines.append(f"- 库存可支撑当前消耗约 **{gsd:.0f}天**")
            if ssd is not None:
                lines.append(f"- 源石碎片库存可支撑约 **{ssd:.0f}天**")
            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # ── 模板缺口分析：差一步就能用 vs 缺关键干员 ──
        # ═══════════════════════════════════════════════════════════════
        _stock = depot_stock or ((BaseScheduler._get_cache() or {}).get("depot_stock"))
        gap_analysis = self._format_template_gap_analysis(
            box, operators, material_stock=_stock)
        if gap_analysis:
            lines.append(gap_analysis)
            lines.append("")

        # ═══════════════════════════════════════════════════════════════
        # ── 培养建议：干员升级收益 + 投入分析 ──
        # ═══════════════════════════════════════════════════════════════
        cultivations = self._get_cultivation_suggestions(
            box, operators, display_best, frontier=frontier, material_stock=_stock)
        if cultivations:
            stars3_list = [c for c in cultivations if "★★★" in c[3]]
            stars2_list = [c for c in cultivations if "★★" in c[3] and "★★★" not in c[3]]
            display = stars3_list[:6] + stars2_list[:6]
            seen_names: set[str] = set()
            deduped: list = []
            for item in display:
                if item[0] not in seen_names:
                    deduped.append(item)
                    seen_names.add(item[0])
            display = deduped[:10]

            stars_total = sum(1 for _ in cultivations if "★★★" in _[3])
            stars2_total = len(stars2_list)
            affordable = sum(1 for _ in cultivations if "✅可执行" in _[3])
            needs_lmd = sum(1 for _ in cultivations if "⚠️缺钱" in _[3])

            # ── Summary banner ──
            banner_parts: list[str] = []
            if affordable > 0:
                banner_parts.append(f"✅ **{affordable} 名可立即执行**（材料充足）")
            if needs_lmd > 0:
                banner_parts.append(f"⚠️ **{needs_lmd} 名缺龙门币**")
            if stars_total > 0:
                banner_parts.append(f"🥇 **{stars_total} 名升E1减心情**（工时+33%）")
            if stars2_total > 0 and len(banner_parts) < 3:
                banner_parts.append(f"🎯 **{stars2_total} 名解锁新技能**")
            banner = " | ".join(banner_parts)

            lines.append("### 🎓 培养建议 — 投入优先级排序")
            lines.append("")
            lines.append(f"> {banner}")
            lines.append("")
            lines.append("| 优先级 | 干员 | 当前 | 建议 | 收益 |")
            lines.append("|--------|------|------|------|------|")
            for rank, (name, current, target, benefit) in enumerate(display, 1):
                # Determine priority icon
                if "★★★" in benefit and "已入选" in benefit:
                    prio = "🔥🔥🔥"
                elif "★★★" in benefit:
                    prio = "🔥🔥"
                elif "★★" in benefit and "已入选" in benefit:
                    prio = "🔥🔥"
                elif "★★" in benefit:
                    prio = "🔥"
                else:
                    prio = "⭐"
                lines.append(
                    f"| {prio} #{rank} | {name} | {current} | {target} | {benefit} |"
                )
            lines.append("")

            # ── Actionable summary ──
            if affordable > 0:
                afford_names = [
                    c[0] for c in cultivations[:15]
                    if "✅可执行" in c[3]
                ][:5]
                if afford_names:
                    lines.append(f"**现在就能升**：{'、'.join(afford_names)}")
            if needs_lmd > 0:
                lmd_names = [
                    c[0] for c in cultivations[:15]
                    if "⚠️缺钱" in c[3]
                ][:5]
                if lmd_names:
                    lines.append(f"**需要先刷龙门币**：{'、'.join(lmd_names)}")
            lines.append("")
        else:
            warnings = self._collect_warnings(display_best, box)
            if warnings:
                lines.append("### 🎯 提升建议（干员培养方向）")
                lines.append("")
                lines.append("以下干员解锁更高基建技能后可显著提升效率：")
                lines.append("")
                for w in warnings[:5]:
                    lines.append(f"- {w}")
                lines.append("")

        return "\n".join(lines)

    # ── Phase 2: full schedule detail ──

    def _phase2_show_solution(self, n: int) -> IntelligenceResult | None:
        """Retrieve cached solution #n and show its full schedule detail."""
        _cache = BaseScheduler._get_cache()
        if not _cache or "frontier" not in _cache:
            return IntelligenceResult(
                recommendation="⚠️ 尚未进行排班计算。请先描述你的基建目标和干员列表。",
                confidence=0.0,
                source="knowledge",
            )

        _cache = BaseScheduler._get_cache()
        frontier = _cache["frontier"]
        if n < 0 or n >= len(frontier):
            return IntelligenceResult(
                recommendation=f"⚠️ 方案 #{n+1} 不存在。目前有 {len(frontier)} 个方案（#1-#{len(frontier)}）。",
                confidence=0.0,
                source="knowledge",
            )

        solution = frontier[n]
        return IntelligenceResult(
            recommendation=self._format_solution_detail(
                solution, n + 1,
                _cache["box"],
                _cache["operators"],
                _cache["inventory"],
            ),
            confidence=min(0.95, 0.5 + 0.5 * solution.coverage),
            source="knowledge",
        )

    def _format_solution_detail(
        self,
        solution,
        solution_num: int,
        box: dict[str, int],
        operators,
        inventory,
    ) -> str:
        """Phase 2 output: full detailed schedule for one Pareto solution."""
        from src.intelligence.arknights.base_optimizer import BaseOptimizer

        config_str = self._short_config(solution.config)
        ns = len(solution.shifts) if solution.shifts else 0
        sh = solution.shifts[0].duration_hours if solution.shifts else 0

        lines = [
            f"## 📋 方案 #{solution_num} — 详细排班表",
            f"",
            f"- **产品配置**: {config_str}",
            f"- **预期日产**:",
        ]
        # Concrete daily output (primary)
        if solution.daily_orundum > 0:
            lines.append(f"  - 合成玉 **~{solution.daily_orundum:.0f}/天**")
        if solution.daily_lmd > 0:
            if solution.daily_lmd >= 10000:
                lines.append(f"  - 龙门币 **~{solution.daily_lmd/10000:.1f}万/天**")
            else:
                lines.append(f"  - 龙门币 **~{solution.daily_lmd:.0f}/天**")
        if solution.daily_combat_record > 0:
            lines.append(f"  - 作战记录 **~{solution.daily_combat_record:.0f}/天**")
        # Normalized percentages (secondary reference)
        lines.append(f"- **归一化效率**: 合成玉 {solution.orundum_eff:.0%} | 龙门币 {solution.lmd_eff:.0%} | 作战记录 {solution.combat_record_eff:.0%}")
        lines.append(f"- **覆盖率**: {solution.coverage:.0%}")
        lines.append(f"- **班次**: {ns} × {sh:.0f}h")
        lines.append(f"")

        # ── Closed-loop sustainability ──
        if solution.daily_orundum > 0:
            lines.append("### 🔄 闭环可持续性分析")
            lines.append("")
            # LMD balance
            if solution.sustain_lmd_balance >= 0:
                lines.append(f"- ✅ **LMD自给自足**：日产 {solution.daily_lmd:.0f} → "
                             f"搓石成本 {solution.daily_lmd - solution.sustain_lmd_balance:.0f} → "
                             f"净盈余 **+{solution.sustain_lmd_balance:.0f}/天**")
            else:
                lines.append(f"- 🔴 **LMD亏损**：日产 {solution.daily_lmd:.0f} → "
                             f"搓石成本 {solution.daily_lmd - solution.sustain_lmd_balance:.0f} → "
                             f"净亏损 **-{abs(solution.sustain_lmd_balance):.0f}/天**")
                lines.append(f"  - ⚠️ 每天亏 {abs(solution.sustain_lmd_balance)/10000:.1f} 万龙门币，不能长久维持！")
            # Rock/sanity demand
            if solution.sustain_rock_demand > 0:
                lines.append(f"- 固源岩需求 **{solution.sustain_rock_demand:.0f}个/天** "
                             f"→ 1-7刷理智 **{solution.sustain_sanity_cost:.0f}/天**")
                if solution.sustain_sanity_cost <= 240:
                    lines.append(f"  - ✅ 在自然回复({240}/天)范围内")
                elif solution.sustain_sanity_cost <= 300:
                    lines.append(f"  - ⚠️ 略超自然回复，配合每周理智药可维持")
                else:
                    lines.append(f"  - 🔴 远超自然回复({240}/天)，需大量碎石或理智药")
            # Summary verdict
            if solution.sustain_verdict == "ok":
                lines.append(f"")
                lines.append(f"**结论：此方案可闭环循环运行，无需外部输血。** ✅")
            elif solution.sustain_verdict == "lmd_deficit":
                lines.append(f"")
                lines.append(f"**结论：龙门币持续失血，不可长久。建议每 2-3 天切换为纯龙门币方案「回血」。** ⚠️")
            else:
                lines.append(f"")
                lines.append(f"**结论：此方案不可持续，需要频繁外部补充 LMD 和固源岩。** 🔴")
            lines.append("")

        # Schedule tables
        if solution.shifts:
            for shift in solution.shifts:
                lines.append(f"### {shift.name}（{shift.duration_hours:.0f}h）")
                lines.append("")
                lines.append("| 设施 | 产品 | 推荐干员 | 效率 |")
                lines.append("|------|------|----------|------|")
                for room in shift.rooms:
                    ops_str = "、".join(op.name for op in room.operators) if room.operators else "（空缺）"
                    eff = room.total_efficiency()
                    lines.append(
                        f"| {room.facility}{room.index+1} | {room.product} | "
                        f"{ops_str} | {eff:.1f}% |"
                    )
                lines.append("")

        # Elite warnings
        warnings = self._collect_warnings(solution, box)
        if warnings:
            lines.append("### ⚠️ 精英等级提醒")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Drone guidance
        drone_recs = self._drone_guidance(solution)
        if drone_recs:
            lines.append("### 🚁 无人机分配建议")
            for rec in drone_recs:
                lines.append(f"- {rec}")
            lines.append("")

        # Resource chain
        depot_stock = (BaseScheduler._get_cache() or {}).get("depot_stock")
        balance = BaseOptimizer.check_resource_balance(solution, inventory, depot_stock=depot_stock)
        if balance.get("warnings") or (inventory and not inventory.is_empty()):
            lines.append("### ⚖️ 资源链")
            for w in balance["warnings"]:
                lines.append(f"- {w}")
            gsd = balance.get("gold_stockpile_days")
            ssd = balance.get("stone_stockpile_days")
            if gsd is not None:
                lines.append(f"- 库存可支撑约 **{gsd:.0f}天**")
            if ssd is not None:
                lines.append(f"- 源石碎片库存可支撑约 **{ssd:.0f}天**")
            if not balance["warnings"]:
                lines.append("- ✅ 资源链健康")
            lines.append("")

        # Morale
        # ── Morale-driven schedule output ──
        if solution.schedule_mode == "morale_driven" and solution.rest_groups:
            lines.append("### ⏱️ 心情驱动换班计划")
            lines.append("")
            lines.append("| 组 | 设施 | 产品 | 干员数 | 最长工时 | 最少休息 | 换班频率 |")
            lines.append("|---|------|------|--------|----------|----------|----------|")
            for g in solution.rest_groups:
                cycle = g.work_duration + g.rest_duration
                cycle_str = f"每{cycle:.1f}h换一次" if cycle > 0 else "-"
                lines.append(
                    f"| {chr(65+g.group_id)}组 | {g.facility} | {g.product} | "
                    f"{len(g.operators)} | {g.work_duration:.1f}h | "
                    f"{g.rest_duration:.1f}h | {cycle_str} |"
                )
            lines.append("")
            if solution.work_to_rest_ratio > 0:
                lines.append(f"- ⚖️ 工休比: **{solution.work_to_rest_ratio:.1f}:1**"
                             f"（工作{solution.work_to_rest_ratio:.1f}h需要休息1h）")
            lines.append("")

            # ── Strategic scheduling advice ──
            # When work:rest ≤ 2:1, the base has significant downtime unless
            # operators are split into staggered A/B teams.
            if solution.work_to_rest_ratio < 2.0 and len(solution.rest_groups) >= 2:
                total_ops_needed = sum(len(g.operators) for g in solution.rest_groups)
                lines.append("### 🎯 交错排班策略")
                lines.append("")
                if total_ops_needed * 2 <= len(solution.operator_names):
                    lines.append(
                        f"⚠️ 当前工休比仅 {solution.work_to_rest_ratio:.1f}:1，"
                        f"全组同时上下班将导致约 **{100/(solution.work_to_rest_ratio+1):.0f}%** 时间设施闲置。"
                    )
                    lines.append("")
                    lines.append(
                        f"💡 **建议启用交错排班**：将 {total_ops_needed} 名干员分为 A/B 两组，"
                        f"A 组上班时 B 组在宿舍恢复，实现 **100% 设施在线率**。"
                    )
                    lines.append(
                        f"你 Box 中有 {len(solution.operator_names)} 名干员，"
                        f"足够支持 {total_ops_needed * 2} 人的双组轮换。"
                    )
                else:
                    lines.append(
                        f"⚠️ 当前工休比仅 {solution.work_to_rest_ratio:.1f}:1，"
                        f"设施在线率约 **{100/(solution.work_to_rest_ratio+1):.0f}%**。"
                    )
                    lines.append(
                        f"需要 {total_ops_needed * 2} 名干员才能实现 100% 在线（当前 Box 有 "
                        f"{len(solution.operator_names)} 名）。建议优先培养减心情消耗的干员提升工休比。"
                    )
                lines.append("")
                # Suggest concrete operators to recruit/upgrade for better ratio
                morale_tips = []
                for g in solution.rest_groups[:4]:
                    high_drain_ops = [op for op in g.operators
                                     if op.morale_drain_per_hour(g.facility) > 0.7]
                    if high_drain_ops:
                        names = "、".join(op.name for op in high_drain_ops[:3])
                        morale_tips.append(
                            f"- {g.facility}: {names} 心情消耗较高，"
                            f"替换为减心情干员可延长工时"
                        )
                if morale_tips:
                    lines.append("**降低心情消耗的建议**：")
                    lines.extend(morale_tips[:5])
                    lines.append("")
            lines.append("> 💡 任一组员心情降至50%时整组换班，宿舍满员后恢复速度最快。")

            # ── 24h timeline (event-driven) ──
            lines.append("")
            lines.append("### 🕐 24小时换班时间线")
            lines.append("")
            if solution.rest_groups:
                # Each group alternates work/rest. Start all working.
                lines.append("| 时间 | 事件 |")
                lines.append("|------|------|")
                # Simulate: collect all transition points
                events: list[tuple[float, str]] = []
                for g in solution.rest_groups:
                    # Work→Rest→Work→Rest→... cycles within 24h
                    if g.work_duration <= 0 and g.rest_duration <= 0:
                        continue
                    t = 0.0
                    working = True
                    while t < 24.0:
                        if working:
                            dur = max(g.work_duration, 0.01)  # guard against zero
                            end = t + dur
                            if end <= 24.0:
                                events.append((t,
                                    f"⚡ {chr(65+g.group_id)}组上班 → {g.facility}/{g.product}"))
                            t = end
                        else:
                            dur = max(g.rest_duration, 0.01)
                            end = t + dur
                            if t < 24.0:
                                events.append((t,
                                    f"🛌 {chr(65+g.group_id)}组休息 → 宿舍{g.rest_dorm_index+1}"))
                            t = end
                        working = not working
                events.sort()
                for t, desc in events[:16]:
                    h = int(t)
                    m = int((t - h) * 60)
                    lines.append(f"| {h:02d}:{m:02d} | {desc} |")
                lines.append("")
                lines.append("> 💡 时间从「执行排班」时刻开始算。例如现在执行排班，00:00=现在，04:30=4.5小时后。")

            # ── Agent scheduling instructions ──
            if solution.rest_groups:
                # Build per-group timeline, then group simultaneous events
                # into per-facility cron tasks.
                group_events: list[tuple[float, str, 'RestGroup']] = []
                for g in solution.rest_groups:
                    if g.work_duration <= 0 and g.rest_duration <= 0:
                        continue
                    t = g.work_duration
                    while t < 24.0:
                        group_events.append((t, "rest", g))
                        t += max(g.rest_duration, 0.01)
                        if t < 24.0:
                            group_events.append((t, "work", g))
                        t += max(g.work_duration, 0.01)
                group_events.sort(key=lambda x: x[0])

                # Deduplicate times and collect facilities per time
                facility_by_time: dict[float, set[str]] = {}
                for t, _, g in group_events:
                    if t not in facility_by_time:
                        facility_by_time[t] = set()
                    facility_by_time[t].add(g.facility)

                if facility_by_time:
                    lines.append("")
                    lines.append("### 🤖 Agent 定时换班指令")
                    lines.append("")
                    lines.append("执行首次排班后，为每个时间点创建一次性 Cron 任务。")
                    lines.append("**每个任务仅操作需要换的设施，不碰仍在工作的房间。**")
                    lines.append("")
                    lines.append("| 延迟 | rotation_index | 操作设施 | base_shift_maa 调用 |")
                    lines.append("|------|----------------|----------|---------------------|")
                    import datetime as _dt
                    for idx, t in enumerate(sorted(facility_by_time.keys())[:8], start=1):
                        h = int(t)
                        m = int((t - h) * 60)
                        delay_sec = int(t * 3600)
                        facs = sorted(facility_by_time[t])
                        fac_str = ",".join(facs)
                        now = _dt.datetime.now()
                        fire_at = now + _dt.timedelta(seconds=delay_sec)
                        cron = f"{fire_at.minute} {fire_at.hour} {fire_at.day} {fire_at.month} *"
                        lines.append(
                            f"| +{h}h{m:02d}min | {idx} | {fac_str} | "
                            f"`base_shift_maa(rotation_index={idx}, facility='{fac_str}')` |"
                        )
                    lines.append("")
                    lines.append("> **CronCreate 参数**: `recurring: false, durable: true`")
                    lines.append("> `plan_index=0`（默认），`rotation_index` 按表格递增 1→2→3…")

        # ── Dorm assignments ──
        if solution.dorm_snapshots:
            lines.append("### 🛌 宿舍分配（按恢复效率排序）")
            lines.append("")
            lines.append("| 宿舍 | 干员 | 人均恢复速度 |")
            lines.append("|------|------|-------------|")
            for snap in solution.dorm_snapshots:
                if not snap.operators:
                    continue
                ops_str = "、".join(snap.operators[:5])
                if len(snap.operators) > 5:
                    ops_str += f"... ({len(snap.operators)}人)"
                avg_recovery = (
                    sum(snap.per_op_recovery.values()) / len(snap.per_op_recovery)
                    if snap.per_op_recovery else 0.75
                )
                lines.append(
                    f"| 宿舍{snap.dorm_index+1} | {ops_str} | {avg_recovery:.2f}/h |"
                )
            lines.append("")

        # ── Fia charges ──
        if solution.fia_charges:
            lines.append("### ⚡ 菲亚梅塔充能计划")
            lines.append("")
            lines.append("| 时间 | 目标干员 | 系数 | 油门 |")
            lines.append("|------|----------|------|------|")
            for fc in solution.fia_charges:
                throttle_label = {1: "全油门", 2: "半油门", 3: "轻油门"}.get(
                    fc.throttle, str(fc.throttle))
                lines.append(
                    f"| +{fc.charge_time_h:.1f}h | {fc.operator_name} | "
                    f"{fc.coefficient:.2f} | {throttle_label} |"
                )
            lines.append("")

        # ── Template gap analysis (same as Phase 1) ──
        depot_stock = (BaseScheduler._get_cache() or {}).get("depot_stock")
        gap_analysis = self._format_template_gap_analysis(
            box, operators, material_stock=depot_stock)
        if gap_analysis:
            lines.append(gap_analysis)
            lines.append("")

        # ── Cultivation suggestions ──
        cached_frontier = (BaseScheduler._get_cache() or {}).get("frontier")
        cultivations = self._get_cultivation_suggestions(
            box, operators, solution, frontier=cached_frontier, material_stock=depot_stock)
        if cultivations:
            stars_total = sum(1 for _ in cultivations if "★★★" in _[3])
            affordable = sum(1 for _ in cultivations if "✅可执行" in _[3])
            needs_lmd = sum(1 for _ in cultivations if "⚠️缺钱" in _[3])
            has_coeff = sum(1 for _ in cultivations if "系数" in _[3])

            banner_parts: list[str] = []
            if affordable > 0:
                banner_parts.append(f"✅ **{affordable} 名可立即执行**")
            if needs_lmd > 0:
                banner_parts.append(f"⚠️ **{needs_lmd} 名缺龙门币**")
            if has_coeff > 0:
                banner_parts.append(f"💡 **{has_coeff} 名可参与菲亚梅塔充能**")
            banner = " | ".join(banner_parts)

            lines.append("### 🎓 培养建议 — 提升本方案效率")
            lines.append("")
            if banner:
                lines.append(f"> {banner}")
                lines.append("")
            lines.append("| 优先级 | 干员 | 当前 | 建议 | 收益 |")
            lines.append("|--------|------|------|------|------|")
            for rank, (name, current, target, benefit) in enumerate(cultivations[:10], 1):
                if "★★★" in benefit and "已入选" in benefit:
                    prio = "🔥🔥🔥"
                elif "★★★" in benefit:
                    prio = "🔥🔥"
                elif "★★" in benefit and "已入选" in benefit:
                    prio = "🔥🔥"
                elif "★★" in benefit:
                    prio = "🔥"
                else:
                    prio = "⭐"
                lines.append(
                    f"| {prio} #{rank} | {name} | {current} | {target} | {benefit} |"
                )
            lines.append("")

            # Actionable summary
            if affordable > 0:
                afford_names = [
                    c[0] for c in cultivations[:15]
                    if "✅可执行" in c[3]
                ][:5]
                if afford_names:
                    lines.append(f"**现在就能升**：{'、'.join(afford_names)}")
            if needs_lmd > 0:
                lmd_names = [
                    c[0] for c in cultivations[:15]
                    if "⚠️缺钱" in c[3]
                ][:5]
                if lmd_names:
                    lines.append(f"**需要先刷龙门币**：{'、'.join(lmd_names)}")
            if stars_total > 0:
                lines.append(f"**🥇 强烈建议优先升精英的 {stars_total} 名干员** — 立减心情消耗+33%工时")
            lines.append("")

        if operators and solution.shifts:
            morale = BaseOptimizer.analyze_morale_sustainability(solution.shifts, operators)
            if morale.get("risks"):
                lines.append("### 💚 心情/换班提醒")
                for risk in morale["risks"][:5]:
                    lines.append(f"- {risk}")
                lines.append("")
            else:
                total_used = morale.get("total_operators_used", 0)
                lines.append(f"💚 所有 {total_used} 名干员心情可持续完成本方案。")
                lines.append("")

        # Navigation
        _cache = BaseScheduler._get_cache()
        frontier = _cache.get("frontier", []) if _cache else []
        if frontier:
            lines.append(f"👉 查看其他方案说「**展示方案N**」（共 {len(frontier)} 个），或说「多搓玉少练级」重新计算偏好。")

        return "\n".join(lines)

    # ── Shared helpers ──

    @staticmethod
    def _describe_all_rooms(solution, show_fixed: bool = False) -> tuple[int, str] | None:
        """List all rooms in a solution, skipping fixed facilities by default.

        Returns (room_count, per-room description) to prove all rooms are filled.
        Production rooms only (Trade/Mfg/Power) unless show_fixed=True.
        """
        if not solution or not solution.shifts:
            return None
        shift = solution.shifts[0]
        prod_facs = {"Trade", "Mfg", "Power"}
        lines: list[str] = []
        room_count = 0

        for room in shift.rooms:
            if not show_fixed and room.facility not in prod_facs:
                continue
            ops_str = "、".join(op.name for op in room.operators[:3]) if room.operators else "空缺"
            if len(room.operators) > 3:
                ops_str += f"…({len(room.operators)}人)"
            eff = room.total_efficiency()
            room_count += 1
            lines.append(
                f"  - {room.facility}{room.index+1} [{room.product}] "
                f"{ops_str} ({eff:.0f}%)"
            )

        return (room_count, "\n".join(lines))

    @staticmethod
    def _short_config(config) -> str:
        """Compact config description with room count verification."""
        from src.intelligence.arknights.base_optimizer import ProductConfig
        counters: dict[str, dict[str, int]] = {}
        for facility, product in config.rooms:
            counters.setdefault(facility, {})
            counters[facility][product] = counters[facility].get(product, 0) + 1
        parts: list[str] = []
        total_rooms = 0
        for fac in sorted(counters):
            items = [f"{p}×{c}" for p, c in sorted(counters[fac].items())]
            parts.append(f"{fac}({'+'.join(items)})")
            total_rooms += sum(c for _, c in sorted(counters[fac].items()))
        rooms_label = f" 共{total_rooms}间"
        return f"[{rooms_label}] " + "  ".join(parts)

    @staticmethod
    def _collect_warnings(best, box: dict[str, int]) -> list[str]:
        if not best or not best.shifts:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for shift in best.shifts:
            for room in shift.rooms:
                for w in room.warnings:
                    if w not in seen:
                        seen.add(w)
                        out.append(w)
        return out[:8]

    @staticmethod
    def _get_cultivation_suggestions(
        box: dict,
        operators: list,
        solution,
        frontier: list | None = None,
        material_stock: 'MaterialStock | None' = None,
    ) -> list[tuple[str, str, str, str]]:
        """Generate cultivation suggestions for ALL box operators.

        Scans every E0/E1 operator, checks ALL Pareto-frontier plans to find
        where each operator's unlocked skills would improve efficiency, and
        produces concrete recommendations like "升E1 → 可进方案#5 Mfg/PureGold".

        Uses universal game mechanics as fallback when detailed skill data
        (elite_required, morale_mod) is unavailable:
          - E0→E1: universal -0.25/h morale drain reduction
          - E1→E2: universal -0.25/h morale drain reduction + skill unlock

        Returns list of (name, current_state, suggested_action, benefit_with_cost),
        sorted by: in-solution first, then ★★★ > ★★ > ★, by impact.
        """
        if not box or not operators or not solution:
            return []
        op_map = {op.name: op for op in operators}

        from src.games.arknights.operators import get_elite_cost

        # ── Build per-plan room efficiency benchmarks ──
        # For each plan in the frontier, compute the weakest operator
        # per (facility, product) so we can tell if a promoted operator
        # would be an upgrade in that plan.
        plans_to_check = frontier if frontier else [solution]
        plan_benchmarks: list[dict] = []
        # plan_benchmarks[i] = {(fac, prod): (worst_eff, worst_op_name), ...}
        for plan_sol in plans_to_check:
            bm: dict = {}
            for g in plan_sol.rest_groups:
                key = (g.facility, g.product)
                if g.operators:
                    worst_op = min(g.operators, key=lambda o: o.efficiency_for(g.product, g.facility))
                    worst_eff = worst_op.efficiency_for(g.product, g.facility)
                    bm[key] = (worst_eff, worst_op.name)
                else:
                    bm[key] = (0.0, "空缺")
            plan_benchmarks.append(bm)

        # ── Detect data quality ──
        any_elite_required = any(
            sk.elite_required > 0
            for op in operators for sk in op.skills
            if op.name in box
        )
        logger.debug(
            "Cultivation: any_elite_required=%s, plans=%d",
            any_elite_required, len(plans_to_check),
        )

        suggestions: list[tuple[str, str, str, str]] = []

        # Build set of operators already in the recommended solution
        in_solution_set: set[str] = set()
        for g in solution.rest_groups:
            for op in g.operators:
                in_solution_set.add(op.name)

        EFF_GAIN_THRESHOLD = 3.0   # minimum efficiency gain to suggest a replacement

        for name, val in sorted(box.items()):
            op = op_map.get(name)
            if not op:
                continue
            elite = op.elite_level
            level = op.level or 0
            rarity = op.rarity

            # Skip operators already maxed out
            if elite >= 2:
                has_locked = any(
                    sk.elite_required > elite
                    or (sk.level_required > 1 and level > 0 and level < sk.level_required)
                    for sk in op.skills
                )
                if not has_locked:
                    continue

            # Skip operators with zero base skills
            has_any_base_skill = any(
                sk.facility in ("Trade", "Mfg", "Power", "Control",
                                "Office", "Reception", "Dorm")
                for sk in op.skills
            )
            if not has_any_base_skill:
                continue

            # ── Cost estimation helper ──
            def _cost_tag(target_elite: int) -> str:
                cost = get_elite_cost(rarity, target_elite)
                if not cost:
                    return ""
                total_lmd = cost.lmd + cost.level_lmd
                parts: list[str] = []
                if material_stock and not material_stock.is_empty():
                    if material_stock.lmd >= total_lmd:
                        parts.append("钱✓")
                    else:
                        missing = total_lmd - material_stock.lmd
                        if missing >= 10000:
                            parts.append(f"缺{missing/10000:.1f}万钱")
                        else:
                            parts.append(f"缺{missing}钱")
                else:
                    if total_lmd >= 10000:
                        parts.append(f"{total_lmd/10000:.1f}万钱")
                    else:
                        parts.append(f"{total_lmd}钱")
                if cost.chip_type and cost.chip_count:
                    cname = cost.chip_type
                    if material_stock and not material_stock.is_empty():
                        if material_stock.has(cname, cost.chip_count):
                            parts.append(f"{cname}✓")
                        else:
                            have = material_stock.items.get(cname, 0)
                            parts.append(f"缺{cost.chip_count - have}个")
                    else:
                        parts.append(f"{cname}×{cost.chip_count}")
                summary = " + ".join(parts)
                if material_stock and not material_stock.is_empty():
                    if material_stock.lmd > 0 and material_stock.lmd < total_lmd:
                        return f" — ⚠️缺钱（需{summary}）"
                    return f" — ✅可执行（{summary}）"
                return f" — 💰需{summary}"

            in_sol = name in in_solution_set

            # ═════════════════════════════════════════════════════════
            # (A) Elite promotion → unlocks concrete efficiency skills.
            # "工时增加" is NOT output improvement — skip morale-only suggestions.
            # Only suggest if the promoted operator would gain actual
            # efficiency (skill unlock) usable in at least one frontier plan.
            # ═════════════════════════════════════════════════════════

            # ═════════════════════════════════════════════════════════
            # (B) Skill Unlock — find which frontier plan benefits most
            # ═════════════════════════════════════════════════════════
            if elite < 2:
                next_elite = min(elite + 1, 2)

                # Compute operator's efficiency contribution per (facility, product)
                # after promotion (conservative: use current values if elite_required missing)
                op_fac_eff: dict[tuple[str, str], float] = {}
                for sk in op.skills:
                    if sk.facility not in ("Trade", "Mfg", "Power"):
                        continue
                    if any_elite_required and sk.elite_required > next_elite:
                        continue
                    for prod_key, eff_val in sk.efficiency.items():
                        if not isinstance(eff_val, (int, float)) or eff_val <= 0:
                            continue
                        if prod_key == "all":
                            for p in ("LMD", "Orundum", "PureGold", "CombatRecord", "OriginStone", "Drone"):
                                if sk.facility == "Trade" and p not in ("LMD", "Orundum"):
                                    continue
                                if sk.facility == "Mfg" and p not in ("PureGold", "CombatRecord", "OriginStone"):
                                    continue
                                if sk.facility == "Power" and p != "Drone":
                                    continue
                                k = (sk.facility, p)
                                op_fac_eff[k] = op_fac_eff.get(k, 0.0) + eff_val
                        else:
                            k = (sk.facility, prod_key)
                            op_fac_eff[k] = op_fac_eff.get(k, 0.0) + eff_val

                if op_fac_eff:
                    # Find the best plan+room match across the entire frontier
                    best_match = None  # (plan_num, fac, prod, gain, replaced_op)
                    for pi, bm in enumerate(plan_benchmarks):
                        for (fac, prod), (worst_eff, worst_name) in bm.items():
                            promoted_eff = op_fac_eff.get((fac, prod), 0.0)
                            gain = promoted_eff - worst_eff
                            if gain > EFF_GAIN_THRESHOLD:
                                if best_match is None or gain > best_match[3]:
                                    plan_num = pi + 1 if frontier else 1
                                    best_match = (plan_num, fac, prod, gain, worst_name)

                    if best_match:
                        plan_num, fac, prod, gain, replaced = best_match
                        is_best_plan = (plan_num == 1)
                        plan_hint = f"⭐方案#{plan_num}" if is_best_plan else f"方案#{plan_num}"
                        tag = "已入选" if in_sol else "可替补"

                        # Convert efficiency gain → concrete daily output delta.
                        # Trade: 10 orders/d base, LMD=1000/order, Orundum=20/order.
                        # Mfg:   20 items/d base, 1 item each (gold, CR, stone).
                        # Power: drone speed — not convertible, show % directly.
                        if fac == "Trade" and prod == "LMD":
                            daily_delta = gain * 100  # 10 * gain% * 1000 / 100
                            output_str = f"+{daily_delta:.0f}龙门币/d" if daily_delta >= 10 else f"+{daily_delta:.0f}龙门币/d"
                        elif fac == "Trade" and prod == "Orundum":
                            daily_delta = gain * 2   # 10 * gain% * 20 / 100
                            output_str = f"+{daily_delta:.0f}合成玉/d"
                        elif fac == "Mfg" and prod == "PureGold":
                            daily_delta = gain * 0.2  # 20 * gain% / 100
                            output_str = f"+{daily_delta:.1f}赤金/d"
                        elif fac == "Mfg" and prod == "CombatRecord":
                            daily_delta = gain * 0.2
                            output_str = f"+{daily_delta:.1f}作战记录/d"
                        elif fac == "Mfg" and prod == "OriginStone":
                            daily_delta = gain * 0.2
                            output_str = f"+{daily_delta:.1f}源石碎片/d"
                        elif fac == "Power":
                            output_str = f"+{gain:.0f}%无人机加速"
                        else:
                            output_str = f"+{gain:.0f}%效率"

                        benefit = output_str
                        benefit += _cost_tag(next_elite)
                        # Internal tier marker for sorting — not displayed in table
                        tier_marker = "★★"
                        suggestions.append((
                            name,
                            f"E{elite} Lv{level or '?'}",
                            f"升E{next_elite}",
                            f"{tier_marker} {benefit} → 进{plan_hint} {fac} 替{replaced} ({tag})",
                        ))

            # ═════════════════════════════════════════════════════════
            # (C) Level-gated skills
            # ═════════════════════════════════════════════════════════
            for sk in op.skills:
                if sk.level_required > 1 and (level <= 0 or level < sk.level_required):
                    remaining = max(1, sk.level_required - max(level, 1))
                    new_eff = sum(
                        v for v in sk.efficiency.values()
                        if isinstance(v, (int, float))
                    )
                    if new_eff > 0:
                        benefit = (
                            f"解锁 {sk.name} (+{new_eff:.0f}%) ★"
                            f" — 需约{remaining * 2}张作战记录"
                        )
                        suggestions.append((
                            name,
                            f"E{elite} Lv{level}",
                            f"≥Lv{sk.level_required} (+{remaining})",
                            benefit,
                        ))
                    break

        # ── Sorting ──
        import re as _re_mod
        def _sort_key(item):
            nm = item[0]
            benefit = item[3]
            # Priority 0: in the current solution
            in_sol = 1 if nm in in_solution_set else 2
            # ★★ = skill-unlock with concrete efficiency gain (top priority)
            # ★ = level-gated skill (lower priority)
            tier = 0 if "★★" in benefit else 1
            m = _re_mod.search(r'\+(\d+)%', benefit)
            impact = -int(m.group(1)) if m else 0
            return (in_sol, tier, impact, nm)
        suggestions.sort(key=_sort_key)

        # Filter: only return skill-unlock (★★) suggestions, capped at 20.
        # Level-gated (★) are kept as fallback if too few skill-unlock exist.
        skill_unlocks = [s for s in suggestions if "★★" in s[3]]
        level_gated = [s for s in suggestions if "★" in s[3] and "★★" not in s[3]]
        top = skill_unlocks[:20]
        if len(top) < 5:
            top += level_gated[:10]
        return top[:30]


    @staticmethod
    def _summarize_teams(solution) -> list[str]:
        """Extract which community teams / key operators are used in a solution.

        Returns short labels like ['交际花组', '红云+蛇屠箱', '德克萨斯 135%']
        for the phase-1 comparison table.
        """
        from src.intelligence.arknights.base_optimizer import COMMUNITY_TEAM_TEMPLATES
        lines: list[str] = []

        if not solution.shifts:
            return lines

        # Collect all operator names used across all shifts
        all_used: set[str] = set()
        for shift in solution.shifts:
            for room in shift.rooms:
                for op in room.operators:
                    all_used.add(op.name)

        # Check which community teams are fully present
        for tmpl in sorted(COMMUNITY_TEAM_TEMPLATES, key=lambda t: t.tier):
            if set(tmpl.members).issubset(all_used):
                lines.append(tmpl.name)
                if len(lines) >= 3:
                    break

        # If no recognized teams, show top efficiency operators
        if not lines:
            top_ops: list[tuple[str, float]] = []
            for shift in solution.shifts:
                for room in shift.rooms:
                    for op in room.operators:
                        eff = room.total_efficiency()
                        if eff > 80:
                            top_ops.append((op.name, eff))
            top_ops.sort(key=lambda x: -x[1])
            seen = set()
            for name, my_eff in top_ops:
                if name not in seen:
                    seen.add(name)
                    lines.append(f"{name} {my_eff:.0f}%")
                if len(lines) >= 3:
                    break

        return lines

    @staticmethod
    def _format_template_gap_analysis(
        box: dict,
        operators: list,
        material_stock: 'MaterialStock | None' = None,
    ) -> str:
        """Analyze community team templates vs player's box.

        Shows three categories:
        - ✅ Ready: all members available at required elite level
        - 🔶 Nearly ready: 1-2 elite promotions away (shows cost)
        - 💡 Missing key operator: missing 1-2 members (shows who to recruit)

        Sorted by tier (T0 first), then by closeness to ready.
        """
        from src.intelligence.arknights.base_optimizer import (
            COMMUNITY_TEAM_TEMPLATES, BaseOptimizer,
        )
        from src.games.arknights.operators import get_elite_cost

        if not box or not operators:
            return ""

        op_map = {op.name: op for op in operators}
        box_names = set(box.keys())

        ready: list[dict] = []
        nearly_ready: list[dict] = []  # just need elite promotions
        missing_op: list[dict] = []    # missing 1-2 operators

        tier_order = {"T0": 0, "T1": 1, "T2": 2, "": 3}

        for tmpl in COMMUNITY_TEAM_TEMPLATES:
            have = []
            need_elite = []    # (name, current_elite, required_elite)
            missing = []       # operator not in box

            for member_name in tmpl.members:
                if member_name not in box_names:
                    missing.append(member_name)
                    continue
                op = op_map.get(member_name)
                if op is None:
                    missing.append(member_name)
                    continue
                member_elite = op.elite_level
                min_elite = tmpl.requires_elite.get(member_name, 0)
                if member_elite < min_elite:
                    need_elite.append((member_name, member_elite, min_elite))
                have.append(member_name)

            total_cost_lmd = 0
            cost_parts: list[str] = []
            can_afford = True
            for name, cur_e, req_e in need_elite:
                op = op_map.get(name)
                rarity = op.rarity if op else 1
                for target_e in range(cur_e + 1, req_e + 1):
                    cost = get_elite_cost(rarity, target_e)
                    if cost:
                        total_cost_lmd += cost.lmd + cost.level_lmd
                        if material_stock and not material_stock.is_empty():
                            if material_stock.lmd < total_cost_lmd:
                                can_afford = False
                            if cost.chip_type and cost.chip_count:
                                if not material_stock.has(cost.chip_type, cost.chip_count):
                                    can_afford = False

            # Determine category
            if len(missing) == 0 and len(need_elite) == 0:
                ready.append({
                    "template": tmpl,
                    "have": have,
                })
            elif len(missing) == 0 and len(need_elite) <= 2:
                nearly_ready.append({
                    "template": tmpl,
                    "have": have,
                    "need_elite": need_elite,
                    "total_cost_lmd": total_cost_lmd,
                    "can_afford": can_afford,
                    "gap_count": len(need_elite),
                })
            elif len(missing) <= 2 and len(need_elite) <= 2:
                missing_op.append({
                    "template": tmpl,
                    "have": have,
                    "missing": missing,
                    "need_elite": need_elite,
                    "total_cost_lmd": total_cost_lmd,
                    "can_afford": can_afford,
                    "gap_count": len(missing) + len(need_elite),
                })

        # Sort: ready T0 first, then nearly-ready by gap_count, then missing by gap_count
        ready.sort(key=lambda x: tier_order.get(x["template"].tier, 3))
        nearly_ready.sort(key=lambda x: (
            tier_order.get(x["template"].tier, 3),
            x["gap_count"],
            -x["template"].equiv_eff,
        ))
        missing_op.sort(key=lambda x: (
            tier_order.get(x["template"].tier, 3),
            x["gap_count"],
            -x["template"].equiv_eff,
        ))

        lines: list[str] = []

        # ── Section: Nearly Ready (just elite up) ──
        if nearly_ready:
            lines.append("### 🔶 差一步就能用的顶级模板")
            lines.append("")
            lines.append("| 模板 | 等效效率 | 已有干员 | 需要培养 | 成本 |")
            lines.append("|------|----------|----------|----------|------|")
            for item in nearly_ready[:6]:
                tmpl = item["template"]
                have_str = "、".join(item["have"])
                elite_str = "、".join(
                    f"{n}(E{c}→E{r})" for n, c, r in item["need_elite"]
                )
                if item["can_afford"]:
                    cost_str = f"✅ {item['total_cost_lmd']/10000:.1f}万"
                elif material_stock and not material_stock.is_empty():
                    cost_str = f"⚠️需{item['total_cost_lmd']/10000:.1f}万"
                else:
                    cost_str = f"💰约{item['total_cost_lmd']/10000:.1f}万"
                lines.append(
                    f"| **{tmpl.name}** (T{tmpl.tier.replace('T','')}) "
                    f"| {tmpl.equiv_eff:.0f}% "
                    f"| {have_str} "
                    f"| {elite_str} "
                    f"| {cost_str} |"
                )
            lines.append("")
            lines.append("> 💡 以上模板只需**提升精英等级**即可激活，不需要新干员！")
            lines.append("")

        # ── Section: Missing key operators ──
        if missing_op:
            lines.append("### 🎯 缺关键干员的顶级模板")
            lines.append("")
            lines.append("| 模板 | 等效效率 | 已有 | 缺失 | 获取途径 |")
            lines.append("|------|----------|------|------|----------|")
            for item in missing_op[:8]:
                tmpl = item["template"]
                have_str = "、".join(item["have"]) if item["have"] else "无"
                missing_str = "、".join(item["missing"])
                # Try to provide acquisition hints
                acq_hints: list[str] = []
                for m in item["missing"]:
                    op = op_map.get(m)
                    rarity = op.rarity if op else 1
                    if rarity <= 3:
                        acq_hints.append(f"{m}(公开招募)")
                    elif rarity == 4:
                        acq_hints.append(f"{m}(公招/标准寻访)")
                    elif rarity == 5:
                        acq_hints.append(f"{m}(公开招募/寻访)")
                    else:
                        acq_hints.append(f"{m}(寻访)")
                lines.append(
                    f"| **{tmpl.name}** (T{tmpl.tier.replace('T','')}) "
                    f"| {tmpl.equiv_eff:.0f}% "
                    f"| {have_str} "
                    f"| {missing_str} "
                    f"| {'、'.join(acq_hints)} |"
                )
            lines.append("")

        # ── Section: Already ready (show top 3) ──
        if ready:
            top_ready = ready[:3]
            lines.append("### ✅ 已解锁的顶级模板")
            lines.append("")
            names = "、".join(
                f"**{r['template'].name}**({r['template'].equiv_eff:.0f}%)"
                for r in top_ready
            )
            lines.append(f"你的Box已完全解锁: {names}")
            lines.append("")

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _format_warehouse_context(
        depot_stock: 'MaterialStock | None',
        inventory: 'Inventory | None',
    ) -> str:
        """Build a rich warehouse snapshot for base scheduling decisions.

        Uses depot_stock.get_any() which searches across MAA raw names and
        common Chinese aliases, so items like "固源岩组" (MAA name) are
        correctly matched for the "固源岩" query.
        """
        if not depot_stock or depot_stock.is_empty():
            return ""

        items = depot_stock.items
        lmd = depot_stock.lmd

        # ── Key materials for base production chain ──
        BASE_CHAIN_MATERIALS = [
            ("赤金",       "贸易站→龙门币的原料",     100, 30, 10),
            ("固源岩",     "1-7刷石 → 源石碎片 → 合成玉", 100, 30, 10),
            ("源石碎片",   "制造站产→贸易站搓合成玉",  20, 10, 5),
            ("装置",       "制造站产线+精英化材料",    30, 15, 5),
            ("异铁",       "精英化材料",              30, 15, 5),
            ("异铁组",     "精英化材料",              15, 5, 2),
            ("糖",         "精英化材料",              30, 15, 5),
            ("聚酸酯",     "精英化材料",              30, 15, 5),
            ("酮凝集组",   "精英化材料",              15, 5, 2),
        ]

        # ── Sum up all combat records as "作战记录" ──
        total_cr = sum(
            qty for name, qty in items.items()
            if "作战记录" in name
        )

        # ── Exact rock count (固源岩 only, not 固源岩组) ──
        exact_rocks = depot_stock.get_any("固源岩")

        chain_lines: list[str] = []
        for item_key, hint, high, medium, low in BASE_CHAIN_MATERIALS:
            qty = depot_stock.get_any(item_key)
            if qty == 0:
                continue
            if qty >= high:
                tag = "充裕"
            elif qty >= medium:
                tag = "尚可"
            elif qty >= low:
                tag = "偏少"
            else:
                tag = "紧张"
            chain_lines.append(f"| {item_key} | **{qty}** | {tag} | {hint} |")

        # ── LMD tier ──
        if lmd >= 500000:
            lmd_tier = "💰💰💰 充裕 — 够用很久"
        elif lmd >= 200000:
            lmd_tier = "💰💰 尚可 — 够1-2周"
        elif lmd >= 100000:
            lmd_tier = "💰 偏少 — 精二一个六星都不够"
        elif lmd > 0:
            lmd_tier = "🔴 见底 — 需要优先刷钱本"
        else:
            lmd_tier = "⚠️ 未扫描到（需从主界面读取）"

        # ── Chip summary ──
        chip_total = sum(
            qty for name, qty in items.items()
            if "芯片" in name
        )
        chip_detail: list[str] = []
        for name, qty in sorted(items.items()):
            if "芯片" in name and qty > 0:
                chip_detail.append(f"`{name}×{qty}`")
        chip_preview = "  ".join(chip_detail[:10]) if chip_detail else "无"

        lines: list[str] = []
        lines.append("### 📦 仓库全景（MAA 扫描）")
        lines.append("")

        if lmd > 0:
            lines.append(f"- **龙门币**：**{lmd/10000:.1f}万** — {lmd_tier}")
        else:
            lines.append(f"- **龙门币**：{lmd_tier}")
        lines.append(f"- **材料种类**：{len(items)} 种已扫描")
        if total_cr > 0:
            lines.append(f"- **作战记录库存**：{total_cr} 张（各级合计）")
        lines.append("")

        if chain_lines:
            lines.append("**基建生产链材料**：")
            lines.append("")
            lines.append("| 材料 | 库存 | 评级 | 用途 |")
            lines.append("|------|------|------|------|")
            lines.extend(chain_lines)
            lines.append("")

        if chip_total > 0:
            lines.append(f"**芯片库存**（{chip_total}个合计）：{chip_preview}")
            lines.append("")

        # ── Quick verdict ──
        verdicts: list[str] = []
        gold = depot_stock.get_any("赤金")
        rocks = exact_rocks if exact_rocks > 0 else depot_stock.get_any("固源岩")
        stones = depot_stock.get_any("源石碎片")

        if gold >= 100:
            verdicts.append("✅ 赤金充裕，**赤金→龙门币链**无忧")
        elif gold >= 30:
            verdicts.append("⚠️ 赤金一般，建议至少保持1个赤金制造站")
        elif gold > 0:
            verdicts.append("🔴 赤金紧张，**优先制造赤金**")
        else:
            verdicts.append("❓ **未扫描到赤金数量**，建议回到仓库「全部」页扫描")

        if rocks >= 100:
            verdicts.append("✅ 固源岩充裕，搓玉无忧")
        elif rocks >= 30:
            verdicts.append("⚠️ 固源岩一般，搓玉需配合刷1-7")
        elif rocks > 0:
            verdicts.append("🔴 固源岩很少，**搓玉前需大量刷石头**")

        if stones >= 20:
            verdicts.append("✅ 源石碎片有库存，开启贸易站即可搓玉")
        elif stones > 0:
            verdicts.append("⚠️ 源石碎片库存有限，需先制造再贸易")
        else:
            verdicts.append("❓ 未扫描到源石碎片")

        if 0 < lmd < 50000:
            verdicts.append("🔴 龙门币见底，**强烈建议优先龙门币方案回血**")

        if verdicts:
            lines.append("**仓库结论**：")
            for v in verdicts:
                lines.append(f"- {v}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_warehouse_recommendation(
        depot_stock: 'MaterialStock',
        best,
        goal_key: str,
    ) -> str:
        """Produce warehouse-based recommendations for the selected plan.

        Analyzes what the player has vs what the plan needs, giving concrete
        advice on stockpile sustainability and immediate actions.
        """
        if not depot_stock or depot_stock.is_empty() or best is None:
            return ""
        items = depot_stock.items
        lmd = depot_stock.lmd

        notes: list[str] = []

        # ── Gold stockpile vs LMD trade ──
        gold = depot_stock.get_any("赤金")
        lmd_rooms = sum(1 for r in best.config.rooms
                       if r[0] == "Trade" and r[1] == "LMD")
        if lmd_rooms > 0 and gold > 0:
            # ~20 gold consumed per LMD room per day
            daily_gold = lmd_rooms * 20
            gold_days = gold / daily_gold
            if gold_days < 3:
                notes.append(
                    f"🔴 赤金仅 **{gold}个**（够{gold_days:.1f}天），"
                    f"**必须保持至少{lmd_rooms+1}个赤金制造站**才能不断链"
                )
            elif gold_days < 10:
                notes.append(
                    f"⚠️ 赤金 **{gold}个**（够{gold_days:.0f}天），"
                    f"建议保持赤金制造站运转"
                )
            else:
                notes.append(
                    f"✅ 赤金 **{gold}个**（够{gold_days:.0f}天），"
                    f"短期无忧"
                )

        # ── Rock stockpile vs orundum chain ──
        rocks = depot_stock.get_any("固源岩")
        orundum_daily = best.daily_orundum
        if orundum_daily > 0 and rocks > 0:
            orders = orundum_daily / 20
            daily_rocks = orders * 4  # 2 stones/order × 2 rocks/stone
            rock_days = rocks / daily_rocks if daily_rocks > 0 else 999
            if rock_days < 3:
                notes.append(
                    f"🔴 固源岩仅 **{rocks}个**（够{rock_days:.1f}天搓玉），"
                    f"**需要立刻刷1-7！** 每天需{orders*2:.0f}个源石碎片={daily_rocks:.0f}个固源岩"
                )
            elif rock_days < 10:
                notes.append(
                    f"⚠️ 固源岩 **{rocks}个**（够{rock_days:.0f}天搓玉），"
                    f"记得日常刷1-7补充"
                )
            else:
                notes.append(
                    f"✅ 固源岩 **{rocks}个**（够{rock_days:.0f}天搓玉），"
                    f"搓玉自由"
                )

        # ── LMD analysis ──
        if lmd > 0:
            daily_lmd_net = best.sustain_lmd_balance
            if daily_lmd_net < 0 and lmd < abs(daily_lmd_net) * 3:
                notes.append(
                    f"🔴 LMD仅 **{lmd/10000:.1f}万**，当前方案日亏{abs(daily_lmd_net):.0f}，"
                    f"**3天内就会见底**。建议切换到有LMD产出的方案"
                )
            elif daily_lmd_net < 0 and lmd < abs(daily_lmd_net) * 14:
                days = lmd / abs(daily_lmd_net)
                notes.append(
                    f"⚠️ LMD **{lmd/10000:.1f}万**，日亏{abs(daily_lmd_net):.0f}，"
                    f"可撑{days:.0f}天。需在{days*0.7:.0f}天内切换到回血方案"
                )
            elif daily_lmd_net >= 0 and lmd < 50000:
                notes.append(
                    f"⚠️ LMD仅 **{lmd/10000:.1f}万**，虽然本方案有正收益，"
                    f"但基数太低。建议先全力刷几天钱本回血"
                )

        # ── Chip stock for elite promotions ──
        if goal_key in ("balanced", "mixed_orundum_upgrade"):
            chip_total = sum(
                qty for name, qty in items.items()
                if "芯片" in name
            )
            if chip_total >= 10:
                notes.append(
                    f"✅ 芯片库存充裕（{chip_total}个），有培养新干员的余裕"
                )
            elif chip_total >= 4:
                notes.append(
                    f"💡 芯片库存 **{chip_total}个**，可以精选1-2名干员培养"
                )
            elif chip_total > 0:
                notes.append(
                    f"⚠️ 芯片库存仅 **{chip_total}个**，培养干员需先刷芯片本"
                )

        if not notes:
            return ""

        lines: list[str] = []
        lines.append("### 🏪 基于你的仓库 — 具体建议")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _drone_guidance(best) -> list[str]:
        """Recommend drone allocation for optimal daily output.

        Community consensus (公孙长乐): drones should go to trade rooms
        with 龙舌兰 (Investment skill — 4-gold-bar orders) first,
        because each drone accelerates a larger order value. After that,
        direct to the highest-efficiency Mfg room.
        """
        if not best or not best.shifts:
            return []

        product_priority: dict[str, str] = {
            "OriginStone": "源石碎片制造（→合成玉供应链）",
            "PureGold": "赤金制造（→龙门币收益）",
            "CombatRecord": "作战记录制造（→经验收益）",
        }

        # Find trade rooms with 龙舌兰 investment skill
        dragon_tongue_rooms: list[tuple[str, int, str, float]] = []
        all_mfg: list[tuple[str, int, str, float]] = []

        for shift in best.shifts:
            for room in shift.rooms:
                if room.facility == "Trade" and room.operators:
                    for op in room.operators:
                        for s in op.skills:
                            if "投资" in str(s.name) and s.elite_required <= op.elite_level:
                                dragon_tongue_rooms.append(
                                    (shift.name, room.index, room.product, room.total_efficiency())
                                )
                                break
                elif room.facility == "Mfg" and room.operators:
                    all_mfg.append(
                        (shift.name, room.index, room.product, room.total_efficiency())
                    )

        recs: list[str] = []

        # Priority 1: Dragon tongue trade rooms (龙舌兰组)
        if dragon_tongue_rooms:
            top = max(dragon_tongue_rooms, key=lambda x: x[3])
            recs.append(
                f"**优先将无人机导向 {top[0]} 的 Trade{top[1]+1}**（龙舌兰投资组，"
                f"4赤金订单价值最高），当前效率 {top[3]:.0f}%"
            )

        # Priority 2: Highest-efficiency Mfg room
        all_mfg.sort(key=lambda x: -x[3])
        if all_mfg:
            top_mfg = all_mfg[0]
            product_label = product_priority.get(top_mfg[2], top_mfg[2])
            prefix = "其次" if dragon_tongue_rooms else "**优先"
            suffix = "**" if not dragon_tongue_rooms else ""
            recs.append(
                f"{prefix}将无人机导向 {top_mfg[0]} 的 Mfg{top_mfg[1]+1}{suffix}"
                f"（{product_label}），当前效率 {top_mfg[3]:.0f}%"
            )

        return recs
