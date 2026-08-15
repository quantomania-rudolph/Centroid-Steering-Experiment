# Calm-tail curator — suffix-only, Qwen-achievable fixes

Prefix (immutable — do not repeat):
```
{prefix_text}
```

Rules:
1. Output ONLY the suffix after that prefix.
2. Format (strict): one short natural fix line that continues the locked sentence, then a path line **outside** the fence, then fenced complete `main.py`:
   `{task_dir}/main.py`
   ```python
   ...complete file...
   ```
   Never put the path only as a `# comment` inside the fence.
3. Forbidden: NotImplementedError, TODO, FIXME, pip install, sorry, cannot, unable to.
4. Stdlib-only Python. Short, clear, no frameworks, no tools/APIs, no agent/tool-calling patterns.
5. Keep the fix something a mid-size coding model (Qwen-class) could write: ~20–80 lines, straightforward logic.
6. Task: {task_summary}
7. Success: {success_criteria}
8. `demo()` must return {expect_repr}.

Constraints:
{invariant_constraints}

Narrative:
- Continue the locked sentence naturally (not "Implement/Write/Create/Edit/Add/Build/Fix the" anywhere in the prose).
- If the prefix ends with an open backtick, your FIRST job is to close it immediately with a short name then backtick, e.g. start the suffix with: stub` or parse_json` — then finish the sentence. Never leave the backtick open. Never start the suffix with Not/NotImplemented (divergence).
- Task-specific only — no scheduler/bridge boilerplate.

Broken starter (do not copy bugs):
```python
{panic_starter_main_py}
```

Divergence: first token after prefix must NOT be `{bad_token_at_d_text}` (d={d}).
Panic note (scrubbed, not a template):
```
{bad_tail_summary}
```

Start the suffix immediately.
