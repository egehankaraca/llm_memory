# Conversation baseline v1: manual review

Report: `conversation_baseline_v1_report.json`.

This is one run of qwen3:8b with the short conversation prompt, thinking disabled,
temperature 0, and synthetic fixed memory. All 21 completions parsed successfully;
none were rewritten. This is **not** 100% conversational accuracy.

Observed issues in the raw model output:

- `tennis_advice_recall/1`: the advice answer assumes the recorded 7 o'clock time
  is appropriate now, rather than establishing what advice the user needs.
- `topic_followup/1`: an isolated car topic is tied to tennis without conversational
  evidence. A relevant profile fact is not proof of intent.
- `negation_and_today/2`: the model mixes today's negative intention with the
  standing Saturday schedule.
- `address_request_value/1`: the assistant introduces itself as Ahmet Bey. This
  confuses the user's address preference with the assistant's identity.
- `spelling_and_mixed_message`: the model implies weather capability that the demo
  does not have, and does not clearly answer the tea-preference follow-up.
- `fragment_after_choice`: topic continuity works, but the sports explanations are
  inaccurate; the scoring answer does not meet the dataset's review expectation.
- `reported_advice_followup/2-3`: the false "memory not found" substitution is gone,
  but the model asks the same schedule question twice instead of advancing the dialogue.

Positive observations are limited to these cases: explicit Saturday/time recall,
recognizing the newly stated 8 o'clock time in recent history, acknowledging an
unknown contact rather than inventing a number, and answering the subsequent math
question. These observations do not establish coverage of unseen conversations.

Conclusion: the answer-rewriting architecture error is removed, while raw model
quality remains a separate issue. Do not add sentence-specific production rules
to turn these cases into a misleading success score. Compare documented inference
settings or a candidate model on the same review cases, retain the original report,
and use additional held-out conversations before choosing a deployment setting.

The memory extractor, durable facts, consent controls, and API/database behavior
are not evaluated by this no-write baseline. State and consent integration tests
are separate.

## Small sampling comparison

`conversation_sampling_comparison.json` contains one additional run of 7 turns
from 3 scenarios using the baseline-only `qwen3-non-thinking` preset. The
temperature/top-p/top-k/min-p values follow the official Qwen3-8B model card.
The same model, conversation prompt, and response parser were used; service
defaults were not changed. No database calls were made.

All 7 responses parsed and none were rewritten, but the assistant still introduced
itself as Ahmet Bey. The advice answer still leaned on the stored schedule. The
tennis scoring answer included the expected 15/30/40 sequence, but this one
stochastic run does not establish a general quality improvement or justify
selecting the preset for deployment. Keep both reports, and evaluate model
configuration/capability separately from memory storage.
