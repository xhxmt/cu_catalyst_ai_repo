# Development Workflow

## Purpose
This document defines how Google Antigravity, Codex, and Claude Code work together in this repository.
The goal is to keep implementation fast while preserving review quality, traceability, and predictable change scope.

## Tool Responsibilities
### Google Antigravity
Google Antigravity is the primary implementation environment. It is responsible for active coding, interactive testing, browser-based verification, and producing reviewable artifacts such as screenshots, verification notes, or implementation summaries.

### Codex
Codex is the default planning and repository-structure assistant. It should be used first for multi-step work, risky changes, refactors, and tasks that touch multiple files. Codex should usually inspect the repository, produce a bounded implementation plan, list affected files, identify invariants, propose tests, and point out missing repository guidance. Codex may edit code only when explicitly instructed.

### Claude Code
Claude Code is the review and quality-gate assistant. It should be used after implementation to review correctness, architecture drift, missing tests, edge cases, maintainability issues, and security risks. Claude Code hooks may run standard checks such as formatting, linting, tests, or secret scanning.

## Standard Task Flow
### Low-complexity task
Use Antigravity to implement the change directly. After the change, use Claude Code to perform a review and run checks.

### Medium-complexity task
Ask Codex for a plan first. Review the plan. Let Antigravity implement the approved plan. Then let Claude Code review the result and run checks.

### High-complexity or high-risk task
Ask Codex for a detailed implementation plan first. Review and approve the plan manually. Let Antigravity implement the work in small steps. Run Claude Code review after each major step and again before merge. Manually spot-check the final result.

## Complexity Guidelines
A task should be treated as medium or high complexity when it changes multiple files, affects public interfaces, modifies schema or persistence logic, touches auth or permissions, changes deployment or CI behavior, introduces new dependencies, or requires a migration or refactor.

## Required Deliverables per Task
Every completed task should leave behind the following:
1. a short implementation summary
2. a list of changed files
3. commands run for verification
4. test outcomes
5. remaining risks or follow-up items

For UI, browser, workflow, or API-journey changes, include a screenshot or browser walkthrough artifact when possible.

## Merge Gate
A branch is ready to merge only when all of the following are true:
1. the implementation matches the approved scope
2. required checks pass
3. Claude Code review has no unresolved blocking issue
4. a human has completed a quick spot verification

## Workspace Recommendations for Antigravity
It is recommended to keep three working contexts:

### Build Workspace
Use for active coding, local runs, and primary implementation.

### Verify Workspace
Use for browser checks, UI flow validation, API verification, and artifact collection.

### Ops Workspace
Use for documentation lookup, environment inspection, and connected tools.

## Review Cadence
For large tasks, do not wait until the end to review everything. Review at the end of each meaningful step. This reduces the chance of broad architectural drift and makes fixes smaller.

## Escalation Rules
When a review identifies a major issue:
- If the issue is local and mechanical, return it to Antigravity for a focused fix.
- If the issue suggests the plan was flawed, ask Codex to produce a narrower repair plan before more code is changed.

## Documentation Maintenance
When recurring mistakes appear, update `AGENTS.md`, `.agents/skills/`, `.claude/commands/`, or this workflow file so the guidance improves over time.
