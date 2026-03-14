# Project Rules

## Mission
Build a reproducible pure-Python research workflow for Cu catalyst screening.

The main priority is an end-to-end ML loop:
data -> cleaning -> features -> training -> explanation -> report

DFT automation is included as a modular extension, but it is not the first milestone.

This repository is developed with Google Antigravity as the primary IDE and execution environment.
Codex is used for planning, task decomposition, repository guidance, and reusable agent skills.
Claude Code is used for review, automated checks, merge gating, and quality control.

The default engineering goal is to deliver reproducible, high-quality changes with minimal risk, clear tests, explicit review notes, and stable research outputs.

## Roles

### Antigravity
- Primary implementation agent
- Main owner of feature branches during active development
- Responsible for code changes, local runs, browser verification when applicable, and artifact generation

### Codex
- Planning and decomposition by default
- Maintains repository guidance through `AGENTS.md` and `.agents/skills/`
- May create focused patches only when explicitly requested

### Claude Code
- Review and gatekeeping by default
- Responsible for correctness review, risk review, test review, security review, and merge checks
- Hooks may run automated checks, but should not silently rewrite production code without approval

## Default Workflow
1. Read the request and identify the affected area.
2. For any medium or high complexity task, ask Codex to produce an implementation plan before editing.
3. Approve the plan before coding.
4. Let Antigravity implement the task on a dedicated feature branch.
5. Require Claude Code review before merge.
6. Merge only after checks, review, and manual spot verification pass.

## When Planning Is Mandatory
Planning is required when any of the following is true:
- More than one production file will change
- Public interfaces may change
- The task touches schema, target definitions, data contracts, or reusable training interfaces
- Database, auth, permissions, secrets, infra, payments, or deletion flows are touched
- The task includes refactoring, migration, or performance-sensitive logic
- The task affects build, deployment, or test infrastructure
- The task changes DFT generation, parsing, or dataset append rules

## Working Style
- Read before editing
- Plan first for tasks touching multiple files
- Prefer small, testable changes
- Preserve file formats, schema, and column names unless explicitly changing them
- Prefer minimal diffs over broad rewrites
- Preserve public interfaces unless the task explicitly allows changing them
- Avoid introducing new frameworks or heavy dependencies without written justification
- Keep configuration in dedicated config files whenever possible
- Do not hardcode secrets, tokens, local absolute paths, or machine-specific settings
- Add docstrings to new nontrivial modules and functions
- Add type hints to new Python functions whenever practical

## Branch Rules
- One main writing agent per feature branch
- Antigravity owns feature branch writes by default
- Codex and Claude Code should stay read-only unless explicitly asked to patch
- Do not let multiple agents perform overlapping large edits on the same branch

## Required Checks
Before considering a task done, run the relevant subset of the following:
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run pytest`
- typecheck when applicable
- integration tests when affected
- secret scan for config or infra changes
- dependency or security audit when dependencies change

Also ensure:
- training metrics are saved under `reports/tables/`
- figures are saved under `reports/figures/`

## Data Rules
- Raw data is append-only
- Cleaned and processed data must be versionable
- Unknown units or missing provenance must be flagged for review
- Never mix target definitions silently
- Never mix incompatible calculation settings into the same training target without explicit documentation
- Schema changes must be reflected in validation logic and tests
- Experimental feedback data must be stored with provenance and version tags

## Model Rules
- Keep a baseline model available at all times
- Use the same evaluation protocol across models
- Save MAE, RMSE, and R2 for every run
- Produce parity and learning-curve plots for every train run
- If explainability is enabled, save the related outputs in a stable and reproducible location
- Do not silently change target columns, split logic, or evaluation metrics

## DFT Rules
- Generation, parsing, and sanity checks must stay separate
- Do not auto-submit HPC jobs in the default workflow
- Suspicious parsed results must be flagged, not silently accepted
- DFT modules are extensions to the ML workflow and must not block the main ML milestone
- Appending DFT outputs to training data requires explicit validation and provenance tracking

## Testing Rules
- If behavior changes, tests must be added or updated
- If a bug is fixed, add a regression test when practical
- If code paths branch on edge cases, add at least one edge-case test
- Never remove failing tests without explaining why in the review summary
- If schema or model inputs change, update validation tests
- If plotting or reporting outputs change, verify file generation paths

## Review Expectations

Every implementation summary must include:
- what changed
- which files changed
- what commands were run
- what tests passed
- remaining risks or follow-up items

Every review summary must include:
- blocking issues
- non-blocking issues
- risk assessment
- suggested minimal fixes

## Artifact Expectations
For UI, browser, API journey, workflow, or report-output changes, Antigravity should attach at least one of:
- screenshot
- browser walkthrough
- implementation summary artifact

For research runs, artifact summaries should include:
- dataset version
- model name
- config used
- output paths for metrics and figures

## Directory Guidance
- `src/` contains production code
- `tests/` contains test code only
- `docs/` contains workflow, architecture, review, and decision records
- `.agents/skills/` contains reusable Codex skills
- `.claude/commands/` contains reusable Claude Code review commands
- `scripts/hooks/` contains hook scripts used by Claude Code or local workflows
- `reports/tables/` stores metrics and tabular outputs
- `reports/figures/` stores plots and visual outputs

## Do Not
- Do not skip review for medium or high risk changes
- Do not mix planning notes into production code files
- Do not silently change stable interfaces
- Do not silently change schema, target definitions, or split logic
- Do not auto-merge without review and checks
- Do not bypass hooks for convenience unless explicitly approved and documented
- Do not let DFT-side complexity derail the main ML workflow milestone