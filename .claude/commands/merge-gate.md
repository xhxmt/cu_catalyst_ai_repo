Run a final merge gate for the current branch.

Check the following:
1. approved scope alignment
2. blocking review issues resolved
3. format, lint, typecheck, and relevant tests passed
4. documentation updated when necessary
5. screenshots or artifacts attached for UI or workflow changes
6. remaining risks documented clearly

Return the result in this structure:

## Gate result
PASS or FAIL

## Evidence
Summarize checks, commands, and artifacts.

## Unresolved blockers
List anything that must be fixed before merge.

## Safe follow-up items
List non-blocking items that may be handled later.

Do not merge automatically.
