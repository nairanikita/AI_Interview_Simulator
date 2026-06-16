import base64
import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from document_processor import extract_text_from_pdf, chunk_text, extract_skills
from interviewer_agent import InterviewerAgent
from scorer import generate_score_report
from vector_store import build_vector_store
from voice_pipeline import text_to_speech

load_dotenv()

# declare the custom voice component once at module level
_voice_component = components.declare_component(
    "voice_component",
    path="voice_component"
)

st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🤖",
    layout="wide"
)

# ── session state defaults ────────────────────────────────────────────────────
_defaults = {
    "agent": None,
    "messages": [],
    "interview_active": False,
    "interview_complete": False,
    "score_report": None,
    "question_count": 0,
    "last_question": "",
    "pending_audio_b64": "",   # base64 MP3 waiting to be played by the component
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

            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
            raw_text = extract_text_from_pdf(tmp_path)
            chunks = chunk_text(raw_text)
            resume_summary = extract_skills(raw_text, llm)

            name_response = llm.invoke(
                f"Extract only the candidate's full name from this resume text. "
                f"Return just the name, nothing else.\n\n{raw_text[:500]}"
            )
            candidate_name = name_response.content.strip()
            vectorstore = build_vector_store(chunks)
            os.unlink(tmp_path)

            agent = InterviewerAgent(
                llm=llm,
                vectorstore=vectorstore,
                candidate_name=candidate_name,
                role=role,
                resume_summary=resume_summary,
                job_description=job_description,
            )
            first_question = agent.start_interview()

            st.session_state.agent = agent
            st.session_state.interview_active = True
            st.session_state.question_count = 1
            st.session_state.last_question = first_question
            st.session_state.messages.append(
                {"role": "assistant", "content": first_question}
            )

            # in voice mode, immediately queue TTS for the opening question
            if voice_mode:
                tts_bytes = text_to_speech(first_question)
                st.session_state.pending_audio_b64 = base64.b64encode(tts_bytes).decode()

            st.rerun()

# ── main area ─────────────────────────────────────────────────────────────────
st.title("AI Interview Simulator")
st.caption("Powered by Gemini · Your resume is analyzed in real time")

if not st.session_state.interview_active:
    st.info(
        "Upload your resume and paste a job description "
        "in the sidebar to begin your interview."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1**\nUpload your resume PDF")
    with col2:
        st.markdown("**Step 2**\nPaste the job description")
    with col3:
        st.markdown("**Step 3**\nClick Start Interview")

else:
    # progress bar
    MAX_QUESTIONS = 10
    st.progress(min(st.session_state.question_count / MAX_QUESTIONS, 1.0))
    st.caption(f"Question {st.session_state.question_count} of {MAX_QUESTIONS}")

    # chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # ── score report ──────────────────────────────────────────────────────────
    if st.session_state.interview_complete:
        st.success("Interview complete! Your score report is below.")
        st.divider()

        if st.session_state.score_report is None:
            with st.spinner("Generating your score report..."):
                chat_history = st.session_state.agent.get_history()
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
                st.session_state.score_report = generate_score_report(
                    chat_history, role, llm
                )

        report = st.session_state.score_report
        overall = report["overall_score"]

        if overall >= 8:
            st.success(f"Overall Score: {overall}/10")
        elif overall >= 6:
            st.warning(f"Overall Score: {overall}/10")
        else:
            st.error(f"Overall Score: {overall}/10")

        st.markdown(f"**Hiring Recommendation:** {report['hiring_recommendation']}")
        st.caption(report["recommendation_reason"])
        st.divider()

        st.subheader("Scores by dimension")
        dimensions = [
            ("Technical Knowledge",  report["technical"]),
            ("Communication",        report["communication"]),
            ("Problem Solving",      report["problem_solving"]),
            ("Experience Relevance", report["experience"]),
            ("Confidence",           report["confidence"]),
        ]
        for label, data in dimensions:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(label, f"{data['score']}/10")
            with col2:
                st.caption(data["feedback"])

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Strengths")
            for s in report["strengths"]:
                st.markdown(f"- {s}")
        with col2:
            st.subheader("Areas to improve")
            for a in report["areas_to_improve"]:
                st.markdown(f"- {a}")

    # ── active interview ──────────────────────────────────────────────────────
    else:

        def handle_agent_response(user_text: str):
            """Get agent reply, update state, queue TTS if in voice mode."""
            st.session_state.messages.append({"role": "user", "content": user_text})

            with st.spinner("Alex is thinking..."):
                response = st.session_state.agent.get_response(user_text)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.question_count += 1
            st.session_state.last_question = response

            if voice_mode:
                tts_bytes = text_to_speech(response)
                st.session_state.pending_audio_b64 = base64.b64encode(tts_bytes).decode()

            if "concludes our interview" in response.lower():
                st.session_state.interview_complete = True

            st.rerun()

        # ── voice mode ────────────────────────────────────────────────────────
        if voice_mode:
            result = _voice_component(
                audio_b64=st.session_state.pending_audio_b64,
                last_question=st.session_state.last_question,
                key="voice_session",
                default=None,
            )

            # clear audio after handing it to the component
            if st.session_state.pending_audio_b64:
                st.session_state.pending_audio_b64 = ""

            if result is not None:
                event_type = result.get("type")

                if event_type == "transcript":
                    handle_agent_response(result["text"])

                elif event_type == "timeout":
                    # 15 seconds of silence — Alex moves on
                    handle_agent_response(
                        "[The candidate did not respond for 15 seconds. "
                        "As Alex, acknowledge this briefly and move to the next question.]"
                    )

                elif event_type == "repeat":
                    # candidate asked to repeat — replay last question via TTS
                    tts_bytes = text_to_speech(st.session_state.last_question)
                    st.session_state.pending_audio_b64 = base64.b64encode(tts_bytes).decode()
                    st.rerun()

                elif event_type == "pause":
                    # candidate asked for a moment — component already extended its timer
                    tts_bytes = text_to_speech("Of course, take your time.")
                    st.session_state.pending_audio_b64 = base64.b64encode(tts_bytes).decode()
                    st.rerun()

        # ── text mode ─────────────────────────────────────────────────────────
        else:
            user_input = st.chat_input("Type your answer here...")
            if user_input:
                handle_agent_response(user_input)
