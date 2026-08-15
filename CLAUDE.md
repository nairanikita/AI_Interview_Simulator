# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app that runs a 10-question AI mock interview: it reads a candidate's resume PDF, compares it against a pasted job description, asks adaptive questions in text or voice mode, then produces a scored report and an answer guide. Single OpenAI-backed Python app, no separate backend/frontend split.

## Commands

```bash
# setup
python -m venv virenv
source virenv/bin/activate
pip install -r requirements.txt
# requires OPENAI_API_KEY in a .env file (see .env.example)

# run
streamlit run app.py                 # http://localhost:8501, opens ws server on :3002

# docker
OPENAI_API_KEY=sk-... docker compose up --build

# CI import check (what .github/workflows/ci.yml actually runs — there is no
# pytest/ruff step wired into CI despite the .pytest_cache/.ruff_cache dirs)
python -c "from document_processor import extract_text_from_pdf, chunk_text, extract_skills"
python -c "from vector_store import build_vector_store"
python -c "from scorer import generate_score_report, generate_answer_guide"
python -c "from interviewer_agent import InterviewerAgent"
python -c "from voice_pipeline import text_to_speech"
```

There is no pytest suite currently checked in (`tests/` is untracked and empty aside from a stale `__pycache__`) and no ruff config file — treat both `.pytest_cache` and `.ruff_cache` as leftovers from local runs, not as live tooling.

## Architecture

Pipeline: resume PDF + job description → `document_processor.py` (extract/chunk/summarize) → `vector_store.py` (ChromaDB, in-memory, rebuilt fresh per session) → `interviewer_agent.py` (LangChain chat loop, RAG-grounded) → `app.py` (Streamlit UI, session-state driven) → `scorer.py` (post-interview report + answer guide).

- **`app.py`** is the orchestrator: Streamlit session state holds the `InterviewerAgent` instance, message list, and question count. It also owns the WebSocket voice server (see below) and the retry-countdown UI for LLM rate limits.
- **`interviewer_agent.py`**: `InterviewerAgent` holds `chat_history` as a list of LangChain messages and rebuilds the system prompt (persona "Alex") on every turn using freshly retrieved resume context (`vectorstore.similarity_search`, k=3). The interview ends when the LLM reply contains the literal phrase "concludes our interview" — `app.py` checks for that string, so changing the system prompt's sign-off phrase requires updating both places in sync.
- **`vector_store.py`**: builds a fresh Chroma store per interview from resume chunks — no persistence across sessions is intended for the RAG index itself (the `chroma_db/` dir / docker volume is incidental Chroma state, not curated data).
- **`scorer.py`**: two independent LLM calls over the full transcript — `generate_score_report` (5 dimensions + hiring recommendation) and `generate_answer_guide` (per-question ideal answer + missed points). Both parse a strict `LABEL: value` text format out of the LLM response by hand (no JSON mode/structured output) — if you change a prompt's output format, update the matching parser (`parse_score_report`, the loop in `generate_answer_guide`) in the same edit.
- **Voice mode** is the most involved piece — see the "Voice mode architecture" section of `README.md` for the full state machine. In short: `app.py` starts a `websockets` server on port 3002 in a background thread (`@st.cache_resource`-cached across reruns), the browser component (`voice_component/index.html`) runs the Web Speech API state machine (SPEAKING → WAITING → RESPONDING/timeout → PROCESSING) and talks to that socket directly, and `voice_pipeline.py` generates gTTS audio that gets base64-pushed to connected clients. STT is browser-native; `voice_pipeline.transcribe_audio` (Whisper) exists only as an unused fallback path.
- Rate-limit handling: `document_processor.extract_skills` and `InterviewerAgent._invoke_with_retry` both retry up to 3 times on 429/quota errors with `on_retry(wait_seconds)` callbacks wired to `app.py`'s countdown UI — keep new LLM call sites consistent with this pattern if they need the same UX.

### Stale/dead code to be aware of

- The `__main__` blocks in `document_processor.py` and `interviewer_agent.py` still reference `ChatGoogleGenerativeAI`, which isn't imported anywhere in the current (OpenAI-based) codebase — leftovers from a Gemini→OpenAI migration (see git history). They will raise `NameError` if run directly; don't treat them as working examples.
- The top of `interviewer_agent.py` is a large commented-out block (an older `ConversationalRetrievalChain`-based implementation) kept for reference above the live `InterviewerAgent` class.
