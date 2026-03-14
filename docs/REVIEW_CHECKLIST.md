# Review Checklist

Use this checklist during Claude Code review, human review, or final merge checks.

## Functional Correctness
- Does the change solve the requested problem?
- Do happy-path flows still work?
- Are edge cases handled explicitly?
- Are errors surfaced clearly and safely?

## Scope Control
- Does the branch stay within the approved task boundary?
- Were unrelated files or behaviors changed?
- Were stable interfaces preserved where expected?

## Architecture and Maintainability
- Does the change fit existing patterns?
- Is the design simpler or at least no more confusing than before?
- Are names, boundaries, and responsibilities clear?
- Has temporary debugging code been removed?

## Tests
- Were tests added or updated when behavior changed?
- Do tests cover the main path and at least one edge case?
- Is there a regression test for a bug fix when practical?
- Are tests stable rather than flaky or timing-dependent?

## Tooling and Verification
- Was formatting run?
- Was linting run?
- Was type checking run when relevant?
- Were affected tests run?
- Were commands documented in the implementation summary?

## Security and Safety
- Are secrets kept out of code and logs?
- Are permissions, auth, or security-sensitive paths handled carefully?
- Are shell commands, file operations, and external calls validated?
- Were new dependencies checked for necessity and risk?

## Data and State
- Are schema or storage changes backward-aware?
- Are migrations safe and reversible where needed?
- Is failure behavior acceptable if part of the workflow breaks?
- Are retries, idempotency, and duplication handled where relevant?

## Observability
- Are logs useful without leaking sensitive data?
- Are important failures discoverable?
- Is there enough information to debug production issues?

## User Experience
- For UI or API behavior changes, is the resulting flow understandable?
- Were screenshots or verification artifacts attached when relevant?
- Are error messages clear to the user?

## Release Readiness
- Are there unresolved blocking issues?
- Are non-blocking follow-ups documented?
- Is the branch safe to merge today?
