# Support Studio

A local demo of a client-facing support workflow: import approved policies, receive a support request, retrieve relevant passages, create a cited draft, and mark it approved after human review. Optional live inference uses **Nebius Token Factory**. The app never sends a reply to a customer.

This is a portfolio prototype built around the application-layer topics in the [Nebius Agentic AI Builder exam guide](https://nebius.com/certification/agentic-ai): model inference, structured output, retrieval, evaluation and operational controls. **It is not a Nebius certification, a production deployment or evidence of paid client work.**

## What a client can see

- Inbox, knowledge base and response workspace in one local dashboard.
- Full-text retrieval via SQLite FTS5; each draft stores the IDs of the passages used.
- Local demo mode requiring no account, key or credit. Its response is a deterministic excerpt, not an LLM answer.
- Optional Nebius Token Factory chat completion with JSON response, source-ID validation, retry on transient errors, daily call limit, token counts and latency.
- Human review can edit the generated answer before saving an approved version. The original draft remains visible for comparison. Approval **does not send** any email or message.

The bundled Northstar Shop policies and customer are fictional. The sample demonstrates return, shipping and damaged-item questions without exposing personal information.

## Run locally

Requires Python 3.11+. On Rubén's Mac, `python3` currently points to 3.9, so use the installed `uv` launcher to create a 3.12 environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787`, click **Load fictional example**, select the new ticket and click **Create demo draft**. Review its source text before marking it approved.

### Enable Nebius inference

Create a Token Factory API key and choose a model with chat completions and JSON mode from your own [model catalog](https://api.tokenfactory.nebius.com/docs). Set variables in your shell, not in committed files:

```bash
export NEBIUS_API_KEY='…'
export NEBIUS_MODEL='exact/model-id-from-your-catalog'
export ENABLE_LIVE=1
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8787
```

The base endpoint is the documented [OpenAI-compatible `/v1/chat/completions` API](https://api.tokenfactory.nebius.com/docs). Live calls may incur charges. The key is never returned to the browser or saved in SQLite. `MAX_LIVE_CALLS_PER_DAY` defaults to 20 saved calls per UTC day. This prototype does not calculate currency cost; model pricing must be checked separately. The call limit is local and not a hard billing safeguard if a network call succeeds but the app fails before saving its draft.

## Scope and limits

This prototype does not authenticate users, isolate customer tenants, ingest PDFs, send messages, execute tools, or fine-tune a model. Keep it bound to `127.0.0.1` and use only fictional or consented data. The retrieval is lexical and can miss synonyms; every draft needs source and factual review. A production version would add authentication, per-tenant storage, document lifecycle, PII controls, evaluation cases, observability and a deployment plan.

## Freelance positioning

This is a concrete example of work Rubén could adapt for a small e-commerce or service business: “I can build a support assistant that grounds draft replies in your policies and keeps a human in charge.” It is a demonstration of implementation skill, not a claim of client results.
