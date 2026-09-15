# LLM Memory Service

Structured long-term facts, TTL-based short-term session state, a bounded
conversation window, and append-only interaction events for an LLM-based
assistant.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   venv/bin/pip install -r requirements.txt
   ```

2. For local Homebrew PostgreSQL, use the current macOS user through its Unix
   socket. No placeholder password is needed:

   ```bash
   export DATABASE_URL='postgresql+psycopg2:///memory_db'
   ```

   For an isolated test database, replace `memory_db` with `memory_db_test`.
   Hosted/production PostgreSQL should use its real username, password, host,
   and TLS configuration instead.

3. Start Ollama and select an installed extraction model:

   ```bash
   ollama serve
   ollama pull qwen3:8b
   export MEMORY_ANALYZER_PROVIDER='ollama'
   export OLLAMA_MODEL='qwen3:8b'
   ```

   If the Ollama desktop application is already running, do not start a second
   `ollama serve` process. The service uses `http://127.0.0.1:11434` by default.

4. Apply the schema migration:

   ```bash
   venv/bin/alembic upgrade head
   ```

5. Start the API:

   ```bash
   venv/bin/uvicorn main:app --reload --port 8001
   ```

The application no longer creates or modifies tables during import. Schema changes
must be applied through Alembic.

If the old prototype already created these tables, back up any required data and
either recreate the development database before running the initial migration or
write a one-off migration for that existing schema. Do not stamp the migration as
complete without adding the new columns and constraints.

## Tests

The test suite uses an isolated in-memory SQLite database and does not access the
configured PostgreSQL database:

```bash
venv/bin/python -m unittest discover -s tests -v
```

The suite mocks Ollama. It validates the structured-output parser, automatic
fallback, storage routing, confirmation policy, and idempotency without loading
a real model.

The curated Turkish model-quality cases are in `evals/memory_cases.jsonl`. Each
line contains one independent input and only the stable behavioral expectations:
memory type, sensitivity, confirmation requirement, and storage destination.
Exact model-generated keys, reasons, and confidence scores are intentionally not
treated as ground truth.

Run a four-case real-model smoke evaluation without writing anything to
PostgreSQL:

```bash
venv/bin/python scripts/run_memory_evals.py \
  --case long_001 \
  --case short_001 \
  --case sensitive_001 \
  --case discard_001
```

Run all 50 cases and save the detailed report:

```bash
venv/bin/python scripts/run_memory_evals.py \
  --json-report evals/latest_report.json
```

The full evaluation is deliberately sequential to avoid loading concurrent model
contexts into limited RAM. It can take several minutes. Use `--tag health`,
`--limit 5`, or repeated `--case CASE_ID` options for smaller runs. Any mismatch
or hidden rules fallback produces a non-zero exit code.

## Memory-only end-to-end checks

These checks exercise the Memory API without generating conversational answers.
The default runner bypasses extraction, so Ollama is not needed. Use a dedicated
development/test database: the runner writes to **the database selected by the
running API**, not a database selected by the runner's environment.

In terminal 1, apply the new per-item temporary-memory migration and restart the
API (stop an older server first so it loads the new endpoints and context policy):

```bash
export DATABASE_URL='postgresql+psycopg2:///memory_db_test'
export MEMORY_ANALYZER_PROVIDER='rules'
export MEMORY_WINDOW_MAX_MESSAGES=10
export MEMORY_WINDOW_MAX_TOKENS=2000
export MEMORY_TEMPORARY_MAX_ITEMS=20
export MEMORY_TEMPORARY_MAX_TOKENS=1500
venv/bin/alembic upgrade head
venv/bin/uvicorn main:app --reload --port 8001
```

In terminal 2, from the project directory:

```bash
venv/bin/python scripts/run_memory_scenarios.py \
  --json-report evals/memory_scenarios_report.json
```

Expect seven `PASS [storage]` checks and exit code `0`: profile duplicate/update,
two unrelated temporary goals, same-slot update, independent item expiry, the
newest 10 of 15 raw messages, message-ID replay, cross-session/user isolation,
and HTTP `403` for another user accessing the session. The expiry probe uses
`--ttl-seconds 3` by default and applies the remaining 10-second polling deadline
to each HTTP request. These are storage
checks, not proof that a model classifies every sentence correctly.

Each run prints fresh synthetic user/session IDs and leaves those isolated test
records in the API's database; it does not delete existing data or clean up its
own profile/event records. Do not run it against production. Reports can contain
candidate values and should not be published with real-user traces.

To test actual extraction as well, restart terminal 1 with
`MEMORY_ANALYZER_PROVIDER=ollama`, keep Ollama running, then run:

```bash
venv/bin/python scripts/run_memory_scenarios.py --with-analyzer \
  --json-report evals/memory_scenarios_analyzer_report.json
```

This additionally ingests the original 15 Turkish messages sequentially under a
separate fresh user/session. It checks the rest intention, discard of questions
and greeting, coffee deduplication, tennis correction, and pending health data:
the current headache must have finite expiry, while the persistent medication
routine must not. It also verifies that unrelated temporary intentions still
coexist even if a small model emits the same semantic key, plus event-ID replay
and the sensitive-history policy. Ordinary profile sentences that differ only by
terminal sentence punctuation deduplicate; protected literal values remain exact.

Any mismatch or `rules`/`rules_fallback` extraction fails explicitly with exit
code `1`. A `rules_guard` result is different: Ollama completed, then the
deterministic protected-data guard intervened. That is a passing system-safety
result, recorded separately in `analyzer_summary.guard_intervention_count` and
with `analysis.upstream_analyzer_source=ollama`. A clean model-only run has
`pure_ollama_count=15`, `guard_intervention_count=0`, and `fallback_count=0`.
Passing these finite cases is not a universal semantic-quality score.
`--memory-url` selects a different API; `--timeout-seconds` defaults to `120`.

### Temporary items and sensitive context

`POST /v1/temporary-memories` accepts an explicit `user_id`, `session_id`, `key`,
`value`, and timezone-aware future `expires_at`; `category` defaults to `session`.
The trusted backend can also supply provenance and verification fields. Different
category/key slots coexist; updating the same slot preserves unrelated items.
For analyzer-originated writes, an occupied key with a different value is
disambiguated instead of silently overwriting the existing item; model-provided
`matched_memory_id`/relation hints alone never authorize deletion. A late event
also cannot roll an active slot back to an older `occurred_at`.
`GET /v1/sessions/SESSION/temporary-memories?user_id=USER` returns the bounded live
`temporary_memories` array, count, and limits. `POST /v1/context:build` includes
that array, while legacy `session` exposes only the newest compatible goal.
Each item expires independently; raw conversation history has its own TTL.

For a manual expiry check, use fresh IDs and create two items (on macOS):

```bash
probe_id=$(uuidgen)
short_expiry=$(date -u -v+15S '+%Y-%m-%dT%H:%M:%SZ')
long_expiry=$(date -u -v+10M '+%Y-%m-%dT%H:%M:%SZ')
curl -sS http://127.0.0.1:8001/v1/temporary-memories \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"probe-$probe_id\",\"session_id\":\"probe-session-$probe_id\",\"key\":\"rest\",\"value\":\"Biraz dinlenmek istiyorum.\",\"expires_at\":\"$short_expiry\"}"
curl -sS http://127.0.0.1:8001/v1/temporary-memories \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"probe-$probe_id\",\"session_id\":\"probe-session-$probe_id\",\"key\":\"news\",\"value\":\"Bu akşam haberleri izlemek istiyorum.\",\"expires_at\":\"$long_expiry\"}"
curl -sS "http://127.0.0.1:8001/v1/sessions/probe-session-$probe_id/temporary-memories?user_id=probe-$probe_id"
```

Run the final GET again after 15 seconds: `rest` should disappear while `news`
remains. Refresh the timestamps if you pause before submitting the requests.
Unverified protected data must go through `interactions:process` and explicit
candidate confirmation, not this deterministic temporary-item endpoint.

By default, `context:build` excludes raw user messages linked to pending/rejected
sensitive candidates and confirmed finite sensitive states after their expiry.
Assistant messages written with that user turn's `parent_message_id` are excluded
with it, so a reply cannot echo protected content back into default context.
A trusted backend can explicitly opt in with
`"include_unconfirmed_sensitive_history": true`; `history_policy` records the
choice. The raw session/messages debug endpoint and event log are not deleted or
redacted by this context filter. The host still sees the current input, and this
is not comprehensive sensitive-data detection or a full consent/deletion system.

Runner-only mock tests (no database or real model access):

```bash
venv/bin/python -m unittest tests/test_memory_scenarios.py -v
```

## Ollama configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMORY_ANALYZER_PROVIDER` | `ollama` | Use `ollama` or force deterministic `rules` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama API |
| `OLLAMA_MODEL` | `qwen3:8b` | Installed extraction model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Per-analysis timeout |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long the model remains loaded |
| `OLLAMA_NUM_CTX` | `4096` | Context window; reduce this first when RAM is tight |
| `MEMORY_TIMEZONE` | `Europe/Istanbul` | Computes today/tomorrow expiry |
| `MEMORY_WINDOW_MAX_MESSAGES` | `10` | Maximum live conversation messages returned per session |
| `MEMORY_WINDOW_MAX_TOKENS` | `2000` | Approximate token budget for the live conversation suffix |
| `MEMORY_CONVERSATION_TTL_HOURS` | `24` | Default retention for raw conversation messages |
| `MEMORY_TEMPORARY_MAX_ITEMS` | `20` | Maximum live temporary items selected per session |
| `MEMORY_TEMPORARY_MAX_TOKENS` | `1500` | Approximate token budget for live temporary item envelopes |
| `MEMORY_TEMPORARY_TTL_MINUTES` | `60` | Default item TTL for current-session/unknown transient states |
| `MEMORY_CONSOLIDATION_MAX_FACTS` | `50` | Maximum active profile facts shown to conflict analysis |
| `MEMORY_PROFILE_MAX_FACTS` | `20` | Maximum long-term profile facts selected for one context |
| `MEMORY_PROFILE_MAX_TOKENS` | `1500` | Approximate token budget for selected profile facts |
| `MEMORY_PROFILE_PINNED_CATEGORIES` | `communication,accessibility,emergency_contact` | Categories considered even without query overlap |
| `MEMORY_OUTBOX_MAX_ATTEMPTS` | `5` | Maximum asynchronous extraction attempts before terminal failure |
| `MEMORY_ASYNC_INGESTION` | `false` | Let the terminal chat enqueue extraction instead of waiting for it |

For a low-RAM development profile, use `qwen3:4b-instruct`, reduce
`OLLAMA_NUM_CTX` to `2048`, and set `OLLAMA_KEEP_ALIVE=0` so the model unloads
after every request. This saves memory at the cost of latency and classification
quality. The 4B model must pass the project's Turkish health/preference evaluation
set before it is used in production.

The service sends the model a strict JSON schema with temperature `0`. Ollama is
only the semantic extractor: it returns `should_store`, `claim_kind`,
`evidence_text`, `subject`, `speech_act`, `temporal_scope`, and
`sensitivity_domain`. Deterministic Python policy version 3
then decides `short_term`, `long_term`, `sensitive`, or `discard` and the storage
destination. Health, emergency-contact, exact-location, financial, and credential
data always remain pending until confirmed. If the model is unavailable or
returns invalid JSON, the small rule classifier takes over and the response shows
`analyzer_source: rules_fallback`.

Each candidate response also includes an `analysis` object so the policy decision
is auditable without treating the model's free-text reason as authoritative.
Stored candidates require a literal quote from the current user message and an
explicit assertion (or a contextual reply for temporary state). Question
presuppositions, inferences, and bare topic fragments must not create profile
facts, even when the model incorrectly labels their speech act. Independent
checks reject unsupported quotes, digits, weekdays, and daily/weekly changes.
The `analysis.evidence_guard` field explains an evidence-check rejection.
Protected categories cannot be downgraded by a conflicting `personal`/`none`
model label. Bounded literal-data checks on the validated evidence also flag
explicit emergency contacts, addresses, IBAN/account and credential markers;
`effective_sensitivity_domain` and `sensitivity_overridden_by_policy` record the
override. These checks are not comprehensive sensitive-data recognition.

## Asynchronous outbox and memory worker

`POST /v1/interactions:enqueue` is the latency-safe alternative to the existing
synchronous `POST /v1/interactions:process`. It stores the event, raw user
message, bounded pre-event analysis context, and outbox job in one PostgreSQL
transaction, returns HTTP `202`, and never calls Ollama in the API request.

Run a separate worker with the same database and analyzer environment:

```bash
venv/bin/python scripts/run_memory_worker.py
```

The worker claims jobs with a lease, preserves order within each user/session,
runs the existing extractor/policy/consolidator, retries transient failures with
bounded exponential backoff, and marks successful jobs `completed`. PostgreSQL
workers use `FOR UPDATE SKIP LOCKED`, so different eligible sessions can be
processed without claiming the same job. A delayed worker uses the bounded
context snapshot captured at enqueue time; messages created later cannot leak
backward into an earlier extraction.

Inspect the durable job and its eventual decisions with:

```text
GET /v1/interactions/{event_id}/status?user_id={user_id}
```

Default context hides unfinished/failed async events and their linked assistant
replies because they have not passed sensitive-data policy. The synchronous
endpoint remains available for tests and callers that explicitly require the
memory result in the same response.

### Manual async latency test

Terminal 1:

```bash
export DATABASE_URL='postgresql+psycopg2:///memory_db_test'
export MEMORY_ANALYZER_PROVIDER='ollama'
export OLLAMA_MODEL='qwen3:8b'
venv/bin/alembic upgrade head
venv/bin/uvicorn main:app --reload --port 8001
```

Terminal 2 (the environment must point to the same database):

```bash
export DATABASE_URL='postgresql+psycopg2:///memory_db_test'
export MEMORY_ANALYZER_PROVIDER='ollama'
export OLLAMA_MODEL='qwen3:8b'
venv/bin/python scripts/run_memory_worker.py
```

Terminal 3:

```bash
ASYNC_EVENT_ID="async-$(uuidgen | tr '[:upper:]' '[:lower:]')"
curl -sS -w '\nHTTP %{http_code}; enqueue süresi %{time_total} saniye\n' \
  -X POST http://127.0.0.1:8001/v1/interactions:enqueue \
  -H 'Content-Type: application/json' \
  -d "{\"event_id\":\"$ASYNC_EVENT_ID\",\"user_id\":\"async-user-001\",\"session_id\":\"async-session-001\",\"text\":\"Her sabah bol köpüklü Türk kahvesi içerim\"}"
```

Expect HTTP `202`, `status: queued`, `job.status: pending`, and an enqueue time
that does not include model inference. After the worker prints that event, inspect
the durable result:

```bash
curl -sS \
  "http://127.0.0.1:8001/v1/interactions/$ASYNC_EVENT_ID/status?user_id=async-user-001" \
  | python3 -m json.tool --no-ensure-ascii

curl -sS \
  http://127.0.0.1:8001/v1/users/async-user-001/memories \
  | python3 -m json.tool --no-ensure-ascii
```

The first result should become `job.status: completed` and contain a
`long_term/auto_applied` decision. The second should contain the coffee routine.
To test the terminal chat without waiting for extraction:

```bash
venv/bin/python scripts/run_chat_demo.py \
  --async-memory \
  --user-id chat-ahmet-001 \
  --session-id chat-session-async-001 \
  --debug
```

The debug block shows the outbox event ID. `/status EVENT_ID` displays its
eventual decision; `/pending` displays sensitive candidates after the worker has
processed them.

## Conversation orchestrator / terminal chat demo

The HTTP-only coordinator is in `conversation_orchestrator.py`; its terminal
interface is `scripts/run_chat_demo.py`. It does not import database models,
replace the Memory Service, or touch STT/TTS. The existing FastAPI service stays
independent. The default mode remains sequential for compatibility:

```text
user text -> context:build(query=text) -> Ollama /api/chat -> natural-language reply
          -> interactions:process(user text) -> save assistant session message
```

With `--async-memory` (or `MEMORY_ASYNC_INGESTION=true`), the second line uses
`interactions:enqueue`; the API returns after its short PostgreSQL transaction
and the separate worker performs extraction later. This removes extraction time
from the chat's response path.

The reply model receives selected, attributed `profile_facts`, active session
state, unexpired observations, and the newest conversation suffix. Each fact
includes its owner, record ID, unchanged value, and minimal provenance. Retrieval
scores, duplicate legacy `profile` data, full source-event payloads, and debug
metadata are not sent. Ingestion stores the user message; the coordinator
stores only the assistant message, so user turns are not duplicated. New memories
are available to subsequent turns. Only user messages are extracted into facts;
assistant answers are conversation history, not authoritative profile facts.

Reply generation uses Ollama's native final `message.content` text. The
request does not include `format` or require an `{"answer":"..."}` envelope.
The coordinator validates text type/size and completion status; it does not classify intent, require
literal quotations, compare number/day tokens, or replace a valid answer with
"memory not found". Advice, recall, and contextual follow-ups are interpreted by
the LLM. This removes misleading answer rewrites, **not** model hallucinations:
Transport validity does not establish correctness or clinical safety. Even
JSON-looking final text is delivered literally, not decoded as a reply template.
The memory extractor uses its own strict JSON schema and remains independent of
the reply model. Its semantic-review and per-item TTL policy are described above.

Every non-command message, including a bare topic such as `tenis`, goes to the
reply model with memory and recent conversation context. The model decides
whether to clarify or answer; there is no one-word canned-reply shortcut. Address
preferences are passed as context, never prepended to the answer by code. Restart
the CLI after changing its code; the Memory API's reload does not reload the CLI.
Ingestion still conservatively rejects one-word assertions: for this MVP, say
`Ben emekliyim` rather than `Emekliyim`. This is a known false-negative tradeoff,
not a complete Turkish morphological classifier.

Memory extraction and its conservative write-evidence checks are unchanged.
PostgreSQL, session TTL/window bounds, ownership, idempotency, conflict handling,
explicit protected-information consent, and audit remain in the Memory Service.
A discarded memory does not discard the conversational answer. A memory-write
failure preserves the generated answer for `/retry`; it is not a missing-memory
reply. Do not deploy this local baseline as a clinical or emergency assistant.

### Database facts versus the final sentence

`POST /v1/context:build` returns data, not a prewritten assistant answer. The
additive `profile_facts` contract makes ownership explicit:

```json
{
  "user_id": "demo-user",
  "profile_facts": [
    {
      "fact_id": "fact-123",
      "owner_user_id": "demo-user",
      "category": "routine",
      "key": "weekly_walk",
      "value": "Pazar günü yürüyüş yaparım",
      "provenance": {
        "source_event_id": "event-123",
        "verification_status": "unverified",
        "confidence": 0.95
      }
    }
  ]
}
```

`owner_user_id` identifies the profile owner, not necessarily the subject of
every proposition: a relative's name still belongs to this user's profile.
`source_event_id` is returned only when it references an existing event belonging
to that same user; otherwise it is `null`. Event payloads are not expanded into
the context, since one event can mix ordinary and pending protected information.

The legacy `profile` response remains available. New demo code sends only the
attributed facts to the reply model and instructs it to compose its own sentence
addressed to the user. Older API responses are wrapped locally without invented
source IDs or rewritten values. Wrong-owner or malformed attributed facts fail
before generation or persistence. The full fact envelope counts toward both the
service's profile budget and the demo's total prompt budget; an oversized fact
is omitted whole rather than silently truncated.

This change does **not** normalize old first-person values or guarantee Turkish
grammar. In a read-only local `qwen3:8b` check, the correct saved morning-coffee
fact arrived in a new session with no history, but the model still replied with
`içerim` rather than addressing the user. The coordinator does not hide that
failure with a string replacement. Automated SQLite/mock tests establish data
attribution, isolation, budgeting, and unchanged model-answer delivery—not that
the model always writes the right sentence.

The earlier attributed-profile envelope required no database migration. The
per-item temporary-memory store is migration `20260914_07`, and linked assistant
privacy is `20260915_08`; run `alembic upgrade head` and restart the API. To test
cross-session recall using existing records:

```bash
venv/bin/python scripts/run_chat_demo.py --user-id chat-ahmet-001 --debug
```

Ask `sabahları ne içerdim unuttum`, then use `/context sabahları ne içerdim unuttum`
to inspect the selected data. A generated session ID starts with empty history
while retaining that user's long-term facts. The answer appears once after
`Asistan:`; the final wording comes from the LLM.

Start the Memory API in one terminal (if not already running):

```bash
export DATABASE_URL='postgresql+psycopg2:///memory_db_test'
export MEMORY_ANALYZER_PROVIDER='ollama'
export OLLAMA_MODEL='qwen3:8b'
venv/bin/alembic upgrade head
venv/bin/uvicorn main:app --reload --port 8001
```

Keep Ollama running. In a second terminal, from the project directory, run:

```bash
venv/bin/python scripts/run_chat_demo.py \
  --user-id chat-ahmet-001 \
  --session-id chat-session-001 \
  --debug
```

Try these messages, one at a time:

```text
Bana Ahmet Bey diye hitap et.
Her cumartesi sabah saat 7'de tenis oynarım.
Cumartesi tenis saatim kaçtı?
Bugün parka gitmek istiyorum.
Bugün ne yapmak istiyorum?
Tansiyon hastasıyım.
/pending
```

After a sensitive statement or an ambiguous conflict, the terminal displays the
exact candidate value and its ID. The model does **not** decide consent. Use
`/confirm CANDIDATE_ID` to approve or `/reject CANDIDATE_ID` to decline. A plain
`evet` is a normal chat message, not an automatic confirmation. Commands bypass
the response model and memory extractor; the Memory API enforces candidate owner
and status. `/reject` prevents profile promotion; it does not erase raw text from
the existing TTL conversation window or event log. This is not a full consent,
deletion, or clinical-safety system.

Developer commands:

| Command | Purpose |
| --- | --- |
| `/context [question]` | Inspect the exact Memory API context, optionally query-filtered |
| `/memories` | Inspect active long-term facts |
| `/pending` | Inspect the newest 200 pending candidates for this user |
| `/confirm ID`, `/reject ID` | Explicitly resolve a displayed candidate |
| `/retry` | Complete a failed memory write using the same event/message IDs |
| `/help`, `/exit` | Show help / leave the demo |

To demonstrate long-term recall independently of the session window, exit and
restart with the **same user** and a **different session**:

```bash
venv/bin/python scripts/run_chat_demo.py \
  --user-id chat-ahmet-001 \
  --session-id chat-session-002 \
  --debug
```

Ask `Cumartesi tenis saatim kaçtı?` again. The first prompt in the new session
should have zero history messages and retrieve the existing tennis fact. A new
user ID must not retrieve Ahmet's facts. Use fresh IDs for a clean demo; without
ID flags, the CLI creates and prints unique demo user/session IDs.

For a non-interactive smoke run (writes only to the database selected by the API):

```bash
venv/bin/python scripts/run_chat_demo.py \
  --user-id chat-smoke-001 \
  --session-id chat-smoke-session-001 \
  --text 'Her cumartesi sabah saat 7’de tenis oynarım.' \
  --text 'Cumartesi tenis saatim kaçtı?' \
  --debug
```

CLI flags `--memory-url`, `--ollama-url`, and `--model` override environment values.
The response model is configured separately from the Memory Service extractor:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMORY_SERVICE_URL` | `http://127.0.0.1:8001` | Existing Memory Service API |
| `MEMORY_SERVICE_TIMEOUT_SECONDS` | `120` | Memory requests, including synchronous extraction |
| `CHAT_OLLAMA_BASE_URL` | `OLLAMA_BASE_URL` or local port `11434` | Reply-generation Ollama |
| `CHAT_OLLAMA_MODEL` | `OLLAMA_MODEL` or `qwen3:8b` | Reply model; does not change extraction model |
| `CHAT_OLLAMA_TIMEOUT_SECONDS` | `120` | Reply-generation timeout |
| `CHAT_OLLAMA_NUM_CTX` | `OLLAMA_NUM_CTX` or `4096` | Total model context window |
| `CHAT_OLLAMA_NUM_PREDICT` | `512` | Maximum reply tokens |
| `CHAT_OLLAMA_KEEP_ALIVE` | `OLLAMA_KEEP_ALIVE` or `5m` | Reply model retention |

Using the same installed model for replies and extraction avoids needing two
different model weights in RAM. Calls are sequential, not concurrent. The
coordinator reserves reply tokens plus a safety margin and bounds the **combined**
memory/history/current-input prompt with the same conservative character estimate
used by the service. If necessary it drops observations, session data, then
low-ranked unpinned profile facts; history uses the remaining budget as a newest
suffix. Pinned facts have priority but still obey the global cap. Current user
text is never silently truncated: an oversized input is rejected. `--debug`
reports selected facts, included history count, budget/trimming, actual Ollama
token counts, extraction sources, and memory-write guard reasons. It does not
repeat the assistant answer in the debug JSON. Debug output and reports may contain personal information;
do not publish real-user traces.

Replies use a concise generic system instruction, without synthetic style
demonstrations, response lookup tables, or topic-specific branches. The model
receives actual chronological conversation history and selected memory data;
its final text is displayed unchanged apart from outer whitespace. Qwen3 reply sampling
remains temperature 0.7, top-p 0.8, top-k 20, min-p 0. The `gemma4` model family
uses demo temperature 0.3, top-p 0.95, top-k 64, min-p 0. The
[official Gemma 4 instructions](https://ollama.com/library/gemma4) recommend
temperature 1.0; the lower temperature is a deliberate local demo choice tested
with `evals/chat_model_acceptance.jsonl`, not an official or universal recommendation.
Both use `think: false` and the existing token limits; extractor settings are unchanged. Debug output identifies
this transport with `response_mode: native_ollama_chat`.

### Local Gemma 4 reply trial

`gemma4:12b` was downloaded and tested locally with 10 scenarios / 14 turns.
The final settings use the generic system instruction above, conservative demo
sampling, and native text output; there are no per-message reply templates or
Python sentence replacements. The local run used a 4096-token context and Ollama
reported about 8 GB of model memory. Thinking mode was not retained: one simple
declaration took about 44 seconds, and the advice probe timed out at 60 seconds.

Start a fresh session while retaining your existing profile:

```bash
CHAT_OLLAMA_KEEP_ALIVE=0 venv/bin/python scripts/run_chat_demo.py \
  --model gemma4:12b \
  --user-id chat-ahmet-001 \
  --debug
```

The `--model` flag overrides the old `OLLAMA_MODEL` fallback for replies only.
The Memory Service extractor remains `qwen3:8b`; this change does not fix its
misclassification of every symptom report. `keep_alive=0` releases the reply
weights after generation to limit retention, at the cost of reloading latency.
Neither existing profile records nor old sessions are deleted.

```bash
venv/bin/python scripts/run_conversation_baseline.py \
  --dataset evals/chat_model_acceptance.jsonl \
  --model gemma4:12b \
  --json-report evals/gemma4_manual_report.json
```

The recorded final run is `evals/gemma4_12b_final_report.json`. Compared with the
earlier runs, recalled coffee, news, schedule corrections and negative intentions
were addressed to the correct human, and gratuitous follow-up/weather questions
were absent. This is a finite conversational check, not a medical or universal
quality certification. Advice can still be too routine-focused, and phrases
such as "not ettim" do not prove a durable memory write: inspect memory decisions,
not conversational wording. The baseline writes no Memory Service records.

### Earlier Qwen3 diagnostic

The reported unsolicited weather questions were already present in the raw
`qwen3:8b` output; they were not inserted by the coordinator or retrieved weather
observations. A read-only native test also produced unnecessary questions, and a
seven-turn check of this simplified integration still showed poor conversational
replies and recall. Removing format/style constraints is an integration cleanup,
not a verified semantic fix. The SQLite/mock tests below cannot prove model
quality. Neither stored history nor model weights were deleted or replaced.

For a new-session terminal check, without the old questioning history:

```bash
venv/bin/python scripts/run_chat_demo.py \
  --user-id window-chat-user-001 \
  --session-id native-chat-session-001 \
  --debug
```

Try `Pencereyi açmak istiyorum.`, `Şimdi salona geçiyorum.`,
`Önce haberleri izlemek istiyorum.`, then `Önce neyi izlemek istiyorum?`.
Review the actual `Asistan:` response; successful delivery is not a correctness score. Numbered test
messages are accepted unchanged; there is no production prefix-stripping rule.

To review declarations and subsequent recall, including old questioning history:

```bash
venv/bin/python scripts/run_conversation_baseline.py \
  --dataset evals/declaration_conversations.jsonl \
  --json-report evals/declaration_fix_report.json
```

Restart the chat CLI to load the new prompt. A fresh session avoids replaying old
poor replies while the same user ID retains stored profile facts. No API restart,
migration, model download, or memory deletion is required for this reply change.

### Conversation baseline: no database writes

`evals/conversation_scenarios.jsonl` contains 10 scenarios / 21 turns: advice vs
recall, the reported advice/follow-up failure, corrections, negation, short topic
answers in context, unknown personal facts, general questions, spelling, and mixed
messages. They are review cases, not model-training data or proof of universal coverage.

```bash
venv/bin/python scripts/run_conversation_baseline.py \
  --json-report evals/conversation_baseline_report.json
```

To review only the reported conversation:

```bash
venv/bin/python scripts/run_conversation_baseline.py \
  --scenario reported_advice_followup \
  --json-report evals/conversation_reported_case.json
```

Only Ollama is needed. The runner uses synthetic, fixed memory context and the
same native prompt/options/text validator as the demo; it calls no Memory API and writes
no profile/session records. Each completion's single `answer`, included
history, token counts, and review expectations are recorded. `valid` means the
text/transport worked; **there is no automatic semantic-success percentage**.
Review whether answers address the request and preserve dates, negation, and
corrections. This baseline intentionally does not consolidate new profile facts.
Use `scripts/run_chat_regressions.py` separately for real API/database state checks.

The existing explicitly named sampling preset remains available:

```bash
venv/bin/python scripts/run_conversation_baseline.py \
  --preset qwen3-non-thinking \
  --scenario reported_advice_followup \
  --scenario address_request_value \
  --scenario fragment_after_choice \
  --json-report evals/conversation_sampling_comparison.json
```

This preset now matches the reply defaults, using Qwen's documented non-thinking
sampling parameters (temperature 0.7, top-p 0.8, top-k 20, min-p 0).
[Official Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B).
Different sampling can change answers between runs; review quality and latency,
not just transport validity. `conversation_baseline_v1_review.md` records observed
issues in the original temperature-0 run, without hiding them behind a score.

Failure behavior: a failed context/generation request does not ingest the new
turn. If persistence fails after generation, interactive mode keeps that one turn
in RAM and blocks new turns until `/retry`; retries reuse IDs and do not generate
another answer. In async mode, once enqueue succeeds the extraction job itself is
durable even if the chat process exits; only a transport failure before enqueue
still needs `/retry`. The non-interactive CLI exits non-zero on unresolved
persistence failure. Extractor rule fallback is reported through the eventual
candidate rather than presented as a successful Ollama analysis.

Reply-generation changes require no extra package. The memory lifecycle changes
require migrations through `20260915_09`. Automated tests mock Ollama and exercise
turn order, prompt budgets, explicit confirmation, retry behavior, and integration
with isolated SQLite Memory Service state. Deliberately wrong model labels,
quotes, hours, and weekdays are included; tests check persisted facts as well as
the reply:

```bash
venv/bin/python -m unittest discover -s tests -v
```

After updating code, leave an already-running chat CLI with `/exit` and restart
it: API `--reload` does not reload the separate CLI process. The current API
version is `0.12.0`; `/v1/analyzer/status` should show `policy_version: "3"`.
The new policy does not retroactively repair previously stored facts or erase
old assistant history. Trace and revoke an invalid fact explicitly; use a new
session to test recall without the old conversation window.

Manual reproduction after seeding the weekly tennis routine: start a new
session with the same user, run `/memories`, then ask `her sabah kaçta tenis
oynarım`. The expected response quotes the Saturday routine and asks whether you
mean every day. Send `tenis`: expect a topic clarification rather than a repeated
routine. `/memories` should show the same active fact IDs/values afterward; neither
message should create a profile fact or temporary goal.

### Real-model regressions

The original 50 classification cases remain unchanged. Twelve additional
bug-derived cases are in `evals/memory_regressions.jsonl`; they are regressions,
not a blind holdout or training data. With Ollama running:

```bash
export MEMORY_ANALYZER_PROVIDER='ollama'
export OLLAMA_MODEL='qwen3:8b'
venv/bin/python scripts/run_memory_evals.py \
  --dataset evals/memory_regressions.jsonl \
  --json-report evals/memory_regressions_report.json
```

The JSON report includes candidate analysis, values, and policy rejection reasons
for diagnosis. These synthetic reports are for development; do not publicly
export a similar report containing real sensitive messages. Scores describe the
**model plus application policy**, not model-only understanding.
The classifier score compares type, sensitivity, confirmation and destination
signatures, including item count. It does not independently judge complete
category/key/value semantics. Two valid preferences extracted from one message
can fail a one-item golden expectation; normal/personal labeling can also differ
without changing storage destination. Review candidate details, not only scores.

`policy_v3_initial_report.json` is the first 50-case run **before** protected-data
hardening (44/50). `policy_v3_targeted_report.json` retests its six failures after
the fixes; a subset rerun is not a fresh full-dataset score. To obtain that score:

```bash
venv/bin/python scripts/run_memory_evals.py \
  --json-report evals/policy_v3_report.json
```

With the API and Ollama running, test the entire conversation and PostgreSQL
persistence path:

```bash
venv/bin/python scripts/run_chat_regressions.py
```

This creates a unique demo user, seeds a form-of-address preference and a weekly
tennis routine, then opens a new
session and tests a daily question, `tenis`, actual routine recall, and general
arithmetic. After every question it asserts that active fact IDs/values are
unchanged, and no active session goal was created. Analyzer fallbacks fail the
test rather than counting as model passes. It exits non-zero on failure and leaves its isolated demo records for
inspection in the API-selected database. Calls are sequential to avoid loading
multiple models concurrently. These live tests depend on model output and can
fail; do not treat one successful run as universal reliability.

This is a local development demo, not a production elder-care assistant. It has
no authentication boundary, live weather/device/calendar tools, emergency-call
integration, or production safety evaluation. Prompt instructions are not a
security or clinical guarantee.

## Manual end-to-end test

Start the API on port 8001, then verify model readiness:

```bash
curl -s http://127.0.0.1:8001/v1/analyzer/status | python -m json.tool
```

The response should contain `"provider": "ollama"`, `"available": true`, and
`"model": "qwen3:8b"`.

Send a preference that the old keyword list could not recognize:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/interactions:process \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "demo-preference-001",
    "user_id": "ahmet-001",
    "session_id": "demo-session-001",
    "text": "Çayımı ince belli bardakta ve açık içerim"
  }' | python -m json.tool
```

Look for `memory_type: long_term`, `status: auto_applied`, and
`analyzer_source: ollama`. Use a new `event_id` each time; repeating the same ID
intentionally returns the original idempotent result.

Test a temporary intention:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/interactions:process \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "demo-short-001",
    "user_id": "ahmet-001",
    "session_id": "demo-session-001",
    "text": "Bu öğleden sonra parkta biraz hava almak istiyorum"
  }' | python -m json.tool
```

Test a health statement that is not in the deterministic keyword guard:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/interactions:process \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "demo-sensitive-001",
    "user_id": "ahmet-001",
    "session_id": "demo-session-001",
    "text": "Sabah kalkınca başım dönüyor ve birkaç dakika oturmam gerekiyor"
  }' | python -m json.tool
```

This result must be `memory_type: sensitive`, `status: pending`, and
`requires_confirmation: true`. Confirm it only after an explicit user decision:

```bash
curl -s -X POST \
  http://127.0.0.1:8001/v1/candidates/CANDIDATE_ID:confirm \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "ahmet-001"}' | python -m json.tool
```

Then inspect the exact bounded context that a response-generating LLM would
receive:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/context:build \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "ahmet-001",
    "session_id": "demo-session-001"
  }' | python -m json.tool
```

The context response now contains `recent_messages`, `conversation_window`, and
`message_refs` in addition to profile and structured session memory.

## Short-term conversation window

`POST /v1/interactions:process` automatically stores the current user message.
The orchestrator stores each assistant reply through the session message
endpoint:

```bash
curl -s -X POST \
  http://127.0.0.1:8001/v1/sessions/demo-window-001/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "message_id": "demo-assistant-001",
    "user_id": "ahmet-001",
    "role": "assistant",
    "content": "Yürüyüşe ne zaman çıkmak istersiniz?"
  }' | python -m json.tool --no-ensure-ascii
```

Now process an answer that cannot be resolved without the previous turn. Do not
send `recent_messages`; the service loads its own bounded PostgreSQL window:

```bash
curl -s -X POST http://127.0.0.1:8001/v1/interactions:process \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "demo-window-user-001",
    "user_id": "ahmet-001",
    "session_id": "demo-window-001",
    "text": "Evet, bugün yapalım"
  }' | python -m json.tool --no-ensure-ascii
```

The decision should be `short_term` and `auto_applied`. Inspect the active
sliding window directly:

```bash
curl -s \
  'http://127.0.0.1:8001/v1/sessions/demo-window-001/messages?user_id=ahmet-001' \
  | python -m json.tool --no-ensure-ascii
```

It should return the assistant question followed by the user answer. Repeating
either request with the same `message_id` or `event_id` is idempotent. Use new
IDs when testing a new turn.

Messages are read in chronological order but only the newest suffix within both
configured limits is returned. Expired messages are excluded immediately and
physically cleaned on later message writes. PostgreSQL stores the rows; no Redis
or in-process RAM conversation cache is required for this MVP.

## Conflict and consolidation

Before a long-term or sensitive candidate is stored, the service compares its
slot with the user's active profile facts. Ollama receives a bounded list of
existing slots and may reference one, but the deterministic consolidator
validates that reference and also compares canonical category/key forms.
Protected fact values are redacted from the extractor input.

The candidate response contains `consolidation_action`:

- `create`: no related active fact exists;
- `unchanged`: the same value already exists and no duplicate fact is created;
- `supersede`: an explicit correction such as `artık` replaces the active fact;
- `conflict_requires_confirmation`: a different value for the same slot remains
  pending until the user confirms or rejects it.

Run the complete live API demonstration with a unique test user:

```bash
venv/bin/python scripts/run_consolidation_demo.py
```

The script processes `7`, explicit update `Artık 8`, ambiguous conflict `9`,
confirms the pending candidate, and verifies one active plus two superseded facts.
It exits non-zero when any policy invariant fails. The API, PostgreSQL, and Ollama
must already be running.

## Query-aware profile retrieval

`POST /v1/context:build` no longer sends every active long-term fact. It selects
profile facts with deterministic lexical scoring, recency, confidence, and
verification boosts, then enforces both fact-count and approximate-token limits.
Pinned categories remain candidates even when they do not overlap the query.

Send the current user utterance in `query`:

```bash
curl -sS -X POST http://127.0.0.1:8001/v1/context:build \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "ahmet-001",
    "session_id": "demo-window-002",
    "query": "Bu sabah saat kaçta kalkarım?"
  }' | python3 -m json.tool --no-ensure-ascii
```

With a wake-time routine, tea preference, and form-of-address fact, this query
returns the wake-time routine plus the pinned communication preference. The tea
fact is omitted. `profile_retrieval` explains the result without repeating fact
values:

```json
{
  "strategy": "deterministic_lexical_v1",
  "query_used": true,
  "eligible_fact_count": 3,
  "selected_fact_count": 2,
  "omitted_fact_count": 1,
  "estimated_tokens": 38,
  "max_facts": 20,
  "max_tokens": 1500
}
```

When `query` is omitted, the service returns the highest-ranked recent facts
within the same budgets. The selector does not invoke another LLM or embedding
model. Semantic vector retrieval can be added later if lexical matching becomes
insufficient for a larger profile corpus.

For a no-command demo, open `http://127.0.0.1:8001/docs` and run the same three
endpoints with **Try it out**.

## Automatic memory routing

`POST /v1/interactions:process` stores the interaction event and applies the
configured analyzer:

- explicit stable preferences are written as long-term facts;
- temporary intentions/states are written as independent session items with expiry;
- health, medication, and emergency-contact statements become pending
  candidates;
- messages without reusable information are recorded but ignored for memory.

The analyzer accepts optional `recent_messages` for backward compatibility. When
that field is omitted, the service automatically loads the session's bounded
conversation window. For extraction it keeps only the suffix beginning with the
last assistant message: this preserves contextual replies such as “Evet, bugün
yapalım” while preventing unrelated user-only ingestion events from contaminating
the next classification. The chosen strategy and message count are recorded in
each candidate's `analysis.analysis_context`. The full bounded window is still
stored and returned by `context:build`.

Policy v3 intentionally separates evidence-checked extraction from routing:

```text
message -> Ollama signals + literal evidence -> validation/policy -> memory destination
```

The Ollama extraction contract accepts only a literal string `value`; structured
objects remain available to trusted direct storage endpoints but cannot bypass
the model evidence check. A `contextual_reply` is accepted only when the bounded
history ends with a directly preceding assistant question. If a complete
standalone statement is mislabeled as contextual, one bounded semantic review
may correct the label; a still-contextual result without that prompt is discarded.
Accepted contextual temporary items retain the verified question as
`in_reply_to`, and bounded protected-domain guards also inspect that question so
an affirmative medication/health reply cannot silently become normal memory.

If the extractor returns internally contradictory semantic labels, or copies an
ungrounded value/number/day from recent or existing memory, the service may
request one narrower re-extraction from the original user evidence. It never
rewrites a model-declared question into an assertion by itself and never accepts
the old value as evidence; the reviewed result must still pass question grammar,
literal quote/value, subject, temporal, protected-domain, and policy checks. A
second invalid result is discarded. The retry count/reason remains in candidate
audit metadata (for example `extraction_retry_reasons=["unsupported_value"]`).
For multi-proposition messages, already validated clauses remain immutable during
repair; only evidence belonging to an initially rejected clause can add a new
decision.

Questions, one-off device commands, prompt-injection attempts, and transient
third-party statements are discarded. Stable preferences/profile facts become
long-term memory, temporary user intents become session memory, and protected
domains become pending candidates.

Pending candidates can be inspected and resolved with:

```text
GET  /v1/candidates?user_id=ahmet-001&status=pending
POST /v1/candidates/{candidate_id}:confirm
POST /v1/candidates/{candidate_id}:reject
```

Active memories can be listed or forgotten with:

```text
GET    /v1/users/{user_id}/memories
DELETE /v1/memories/{memory_id}?user_id={user_id}
```

## Local troubleshooting

If PostgreSQL reports `password authentication failed`, remove a previously
exported placeholder URL and use the Homebrew Unix-socket connection:

```bash
export DATABASE_URL='postgresql+psycopg2:///memory_db_test'
```

If Uvicorn reports `Address already in use`, stop the older server with `Ctrl+C`
in its terminal or start this instance on another port:

```bash
venv/bin/uvicorn main:app --reload --port 8001
```
