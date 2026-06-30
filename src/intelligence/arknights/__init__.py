"""Arknights-specific intelligence tools.

Implemented:
    base_optimizer.py  — Product config enumeration + Pareto frontier + SA solver
    base_scheduler.py  — IntelligenceTool wrapper (goal parsing, box extraction)
    base_chain.py      — Chain conductor: scan → schedule → deploy orchestration

Chained skills (data/skills/arknights/):
    box-scan.md        — Scan operator list, OCR names + detect elite rank
    base-deploy.md     — Execute base shift deployment from schedule

These consume the knowledge base (base_skills.json, recruit_tags.json,
stages.json, materials.json, operator_base_skills.json) + relevant skills
+ memories to provide data-driven recommendations to the LLM agent.

Planned:
    recruiter.py       — Recruitment tag optimization
    planner.py         — Operator promotion material planning
"""
