# Native declaration reply review

The final `declaration_fix_report.json` uses the deployed prompt, parser and
sampling options with local `qwen3:8b`. No Memory API/database writes, response
classification or semantic rewriting occur in this runner. JSON validity is
not a semantic success score.

Observed in the final eight-turn run:

- Cousin Ayşe, neighbor Mehmet and sugar-free coffee statements were acknowledged
  without confirmation questions. Their subsequent recall preserved the values.
- Residence İzmir was recognized and recalled, but its initial response still
  asked an unnecessary location question. Some Turkish morphology remains poor.
- This is a targeted improvement, not a claim that every conversational error is fixed.

Earlier diagnostic probes are under `diagnostics/native_reply_20260914/`.
Rejected role-based fictional examples and placeholders leaked into personal
recall, even with an additional system boundary. The deployed implementation
does not insert any fake user/assistant turns. The system-only unknown-relative
probe did not emit the placeholder, but answered with an irrelevant address
preference: unknown personal recall therefore also remains a model quality issue.

End-to-end smoke runs wrote only to the synthetic user
`reply-fix-smoke-20260914`. The final deployed prompt was verified in the new
session `reply-fix-smoke-session-final-20260914`: the model itself answered
"Anladım, kuzeninizin adı Ayşe." and then "Kuzeninizin adı Ayşe'di."
Raw and displayed answers were equal; no semantic rewrite occurred. Repeating
the existing cousin fact was `auto_applied` with consolidation `unchanged`;
the subsequent recall question was `discard` / `ignored` as a new memory.
No existing real-user records, extractor settings or database schema changed.
