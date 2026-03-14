---
name: update-tests
description: Add or update tests whenever behavior changes or a bug is fixed.
---

# When to use
Use this skill after implementing a behavior change, bug fix, refactor with behavior impact, or edge-case handling update.

# Inputs
- changed production code
- existing tests
- repository guidance from `AGENTS.md`

# Required output
1. identify the affected behavior
2. locate the correct test area
3. add or update tests for the main path
4. add at least one edge-case or regression test when practical
5. list commands to run for verification

# Rules
- Match the existing test style unless there is a good reason not to
- Prefer focused tests over broad end-to-end tests for local behavior changes
- Do not remove failing tests without explanation
- If a bug was fixed, aim to encode the original failure mode in a regression test
