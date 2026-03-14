---
name: plan-feature
description: Produce a bounded implementation plan before coding medium or high complexity work.
---

# When to use
Use this skill before implementing a feature, refactor, migration, or any task that touches multiple files or carries nontrivial risk.

# Inputs
- task request
- repository guidance from `AGENTS.md`
- related code and documentation

# Required output
Produce a plan with the following sections:
1. goal
2. affected files
3. invariants to preserve
4. step-by-step implementation plan
5. test plan
6. risks
7. rollback strategy

# Rules
- Read before proposing changes
- Keep the plan implementation-oriented
- Prefer the smallest viable solution
- Explicitly call out risky interfaces and state transitions
- Do not edit files while using this skill unless the user explicitly asks for code changes
