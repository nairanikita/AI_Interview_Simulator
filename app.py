import asyncio
import base64
import json
import os
import queue as _queue
import tempfile
import threading
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from document_processor import extract_text_from_pdf, chunk_text, extract_skills
from interviewer_agent import InterviewerAgent
from scorer import generate_score_report, generate_answer_guide
from vector_store import build_vector_store
from voice_pipeline import text_to_speech

load_dotenv()

st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🤖",
    layout="wide"
)

# ── WebSocket voice server (port 3002) ────────────────────────────────────────
# The voice iframe connects via WebSocket — no HTTP polling, no CORS/PNA issues.
# Python pushes TTS audio to the browser instantly; the browser sends transcripts back.
_VOICE_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_component", "index.html")
_WS_PORT = 3002

@st.cache_resource
def _start_ws_server():
    import websockets  # top-level serve() is stable across all versions

    events_q    = _queue.Queue()
    clients     = set()
    audio_store = {"b64": "", "id": 0, "text": ""}
    lock        = threading.Lock()
    loop        = asyncio.new_event_loop()

    async def _handler(websocket):
        clients.add(websocket)
        # Do NOT send pending audio on reconnect — the browser would replay the
        # last question every time it reconnects (e.g. after Streamlit restart).
        # Audio is only sent when _queue_tts() explicitly pushes it.
        try:
            async for raw in websocket:
                try:
                    events_q.put(json.loads(raw))
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            clients.discard(websocket)

    async def _main():
        # Use 127.0.0.1 (IPv4) — "localhost" resolves to ::1 on macOS and
        # causes Errno 48 when the port is briefly in TIME_WAIT after a restart.
        # reuse_address=True sets SO_REUSEADDR so we can rebind immediately.
        async with websockets.serve(_handler, "127.0.0.1", _WS_PORT,
                                    reuse_address=True):
            await asyncio.Future()   # run forever

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_main())

    threading.Thread(target=_run, daemon=True).start()
    return events_q, clients, audio_store, lock, loop


_voice_events, _ws_clients, _audio_store, _audio_lock, _ws_loop = _start_ws_server()


def _queue_tts(text: str):
    """Generate TTS, store it, and push immediately to all connected WebSocket clients."""
    tts_bytes = text_to_speech(text)
    b64 = base64.b64encode(tts_bytes).decode()

    with _audio_lock:
        _audio_store["id"]  += 1
        _audio_store["b64"]  = b64
        _audio_store["text"] = text
        audio_id = _audio_store["id"]

    if _ws_clients:
        msg = json.dumps({"type": "audio", "b64": b64, "text": text, "audio_id": audio_id})

        async def _push():
            for ws in list(_ws_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    pass

        asyncio.run_coroutine_threadsafe(_push(), _ws_loop)


def _voice_html() -> str:
    with open(_VOICE_HTML_PATH, encoding="utf-8") as f:
        return f.read()


# ── session state defaults ────────────────────────────────────────────────────
_defaults = {
    "agent":              None,
    "messages":           [],
    "interview_active":   False,
    "interview_complete": False,
    "score_report":       None,
    "answer_guide":       None,
    "question_count":     0,
    "last_question":      "",
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Interview Setup")
    st.divider()

    resume_file = st.file_uploader(
        "Upload your Resume Here (PDF format)",
        type=["pdf"],
        help="Please upload your resume in PDF format only."
    )
    job_description = st.text_area(
        "Paste the Job Description here",
        height=200,
        help="Please paste the full job description for the role you're applying to."
    )
    role = st.text_input(
        "Role you are applying for",
        placeholder="e.g. Data Scientist"
    )

    st.divider()
    voice_mode = st.toggle(
        "🎤 Voice Mode",
        value=False,
        help="Continuous voice conversation — Chrome or Edge required."
    )
    st.divider()

    start_button = st.button(
        "Start Interview",
        type="primary",
        disabled=st.session_state.interview_active
    )
    if st.session_state.interview_active:
        if st.button("Reset Interview"):
            for key, val in _defaults.items():
                st.session_state[key] = val
            st.rerun()

# ── retry countdown ───────────────────────────────────────────────────────────
_retry_placeholder = st.empty()

def show_retry_countdown(wait_seconds: int):
    for remaining in range(wait_seconds, 0, -1):
        _retry_placeholder.warning(f"⏳ Rate limit hit. Retrying in {remaining}s…")
        time.sleep(1)
    _retry_placeholder.empty()

# ── start interview ───────────────────────────────────────────────────────────
if start_button:
    if not resume_file:
        st.sidebar.error("Please upload your resume PDF.")
    elif not job_description:
        st.sidebar.error("Please paste a job description.")
    elif not role:
        st.sidebar.error("Please enter the role you are applying for.")
    else:
        with st.spinner("Reading your resume and setting up the interview..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            raw_text = extract_text_from_pdf(tmp_path)
            chunks   = chunk_text(raw_text)
            resume_summary, candidate_name = extract_skills(
                raw_text, llm, on_retry=show_retry_countdown
            )
            vectorstore = build_vector_store(chunks)
            os.unlink(tmp_path)

            agent = InterviewerAgent(
                llm=llm,
                vectorstore=vectorstore,
                candidate_name=candidate_name,
                role=role,
                resume_summary=resume_summary,
                job_description=job_description,
                on_retry=show_retry_countdown,
            )
            first_question = agent.start_interview()

            st.session_state.agent            = agent
            st.session_state.interview_active = True
            st.session_state.question_count   = 1
            st.session_state.last_question    = first_question
            st.session_state.messages.append({"role": "assistant", "content": first_question})

            if voice_mode:
                _queue_tts(first_question)

            st.rerun()

# ── main area ─────────────────────────────────────────────────────────────────
st.title("AI Interview Simulator")
st.caption("Powered by OpenAI · Your resume is analyzed in real time")

if not st.session_state.interview_active:
    st.info("Upload your resume and paste a job description in the sidebar to begin.")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**Step 1**\nUpload your resume PDF")
    with col2: st.markdown("**Step 2**\nPaste the job description")
    with col3: st.markdown("**Step 3**\nClick Start Interview")

else:
    MAX_QUESTIONS = 10
    st.progress(min(st.session_state.question_count / MAX_QUESTIONS, 1.0))
    st.caption(f"Question {st.session_state.question_count} of {MAX_QUESTIONS}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # ── interview complete ────────────────────────────────────────────────────
    if st.session_state.interview_complete:
        st.success("Interview complete! Your results are below.")
        st.divider()

        if st.session_state.score_report is None:
            chat_history = st.session_state.agent.get_history()
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            with st.spinner("Generating your score report…"):
                st.session_state.score_report = generate_score_report(chat_history, role, llm)
            with st.spinner("Building your answer guide…"):
                st.session_state.answer_guide = generate_answer_guide(chat_history, role, llm)

        tab_score, tab_guide = st.tabs(["Score Report", "Answer Guide"])

        with tab_score:
            report  = st.session_state.score_report
            overall = report["overall_score"]
            if overall >= 8:   st.success(f"Overall Score: {overall}/10")
            elif overall >= 6: st.warning(f"Overall Score: {overall}/10")
            else:              st.error(f"Overall Score: {overall}/10")

            st.markdown(f"**Hiring Recommendation:** {report['hiring_recommendation']}")
            st.caption(report["recommendation_reason"])
            st.divider()

            st.subheader("Scores by dimension")
            for label, data in [
                ("Technical Knowledge",  report["technical"]),
                ("Communication",        report["communication"]),
                ("Problem Solving",      report["problem_solving"]),
                ("Experience Relevance", report["experience"]),
                ("Confidence",           report["confidence"]),
            ]:
                c1, c2 = st.columns([1, 3])
                with c1: st.metric(label, f"{data['score']}/10")
                with c2: st.caption(data["feedback"])

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Strengths")
                for s in report["strengths"]: st.markdown(f"- {s}")
            with c2:
                st.subheader("Areas to improve")
                for a in report["areas_to_improve"]: st.markdown(f"- {a}")

        with tab_guide:
            st.subheader("How you could have answered better")
            st.caption("Each question shows your answer, an ideal answer, and key points to focus on.")
            st.divider()
            for i, item in enumerate(st.session_state.answer_guide or [], 1):
                q_preview = item["question"][:90] + ("…" if len(item["question"]) > 90 else "")
                with st.expander(f"Q{i}: {q_preview}", expanded=(i == 1)):
                    st.markdown("**Your answer**")
                    st.info(item["candidate_answer"] or "_No answer recorded_")
                    st.markdown("**Ideal answer**")
                    st.success(item["ideal_answer"] or "_N/A_")
                    if item.get("did_well") and item["did_well"].lower() != "n/a":
                        st.markdown(f"**What you did well:** {item['did_well']}")
                    if item.get("missed_points"):
                        st.markdown("**Key points to include next time**")
                        for pt in item["missed_points"]: st.markdown(f"- {pt}")

    # ── active interview ──────────────────────────────────────────────────────
    else:
        def handle_agent_response(user_text: str):
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.spinner("Alex is thinking…"):
                response = st.session_state.agent.get_response(user_text)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.question_count += 1
            st.session_state.last_question  = response
            if voice_mode:
                _queue_tts(response)
            if "concludes our interview" in response.lower():
                st.session_state.interview_complete = True
            st.rerun()

        # ── voice mode ────────────────────────────────────────────────────────
        if voice_mode:
            components.html(_voice_html(), height=160)

            @st.fragment(run_every=0.5)
            def _voice_poller():
                if not _voice_events.empty():
                    event = _voice_events.get_nowait()
                    st.session_state["_pending_voice"] = event
                    st.rerun(scope="app")

            _voice_poller()

            event = st.session_state.pop("_pending_voice", None)
            if event:
                etype = event.get("type")
                if etype == "transcript":
                    handle_agent_response(event["text"])
                elif etype == "timeout":
                    handle_agent_response(
                        "[The candidate did not respond after two attempts. "
                        "As Alex, acknowledge this kindly and move to the next question.]"
                    )
                elif etype == "repeat":
                    _queue_tts(st.session_state.last_question)
                    st.rerun()

        # ── text mode ─────────────────────────────────────────────────────────
        else:
            user_input = st.chat_input("Type your answer here...")
            if user_input:
                handle_agent_response(user_input)
