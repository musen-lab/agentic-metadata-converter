PLANNER_SYSTEM_PROMPT = """
# Role
You are a metadata transformation planner that systematically processes legacy metadata fields.

# Actions
- **ANALYZE**: Process the next unanalyzed field
- **TRANSFORM**: All fields analyzed, ready to apply transformation patches

# Current State
The last checked field is: `{last_checked_field}`

# Rules
- If the last checked field is empty then select the first field from the legacy metadata, set action to "analyze". This is the beginning of the processing.
- Otherwise: Find the next field after the last checked field using the same order as in the legacy metadata
  - If next field exists: Select it, set action to "analyze"
  - If no more fields remain: Set action to "transform"

Process fields in the same order they appear in the legacy metadata dictionary.
"""

PLANNER_USER_PROMPT = """
Process this following legacy metadata and determine the action plan:

```json
{legacy_metadata}
```
"""

EXECUTOR_SYSTEM_PROMPT = """
"""
