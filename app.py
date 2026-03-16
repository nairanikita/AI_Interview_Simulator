from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
import tempfile
from document_processor import extract_text_from_pdf,chunk_text,extract_skills
from vector_store import build_vector_store
from interviewer_agent import InterviewerAgent

load_dotenv()
# llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)
# response=llm.invoke("Say hello as a strict technical interviewer in one sentence.")
# print(response.content)
st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🤖",
    layout="wide"
)
if "agent"  not in st.session_state:
    st.session_state.agent=None
if "messages"  not in st.session_state:
    st.session_state.messages=[]
if "interview_active"  not in st.session_state:
    st.session_state.interview_active=False


if "interview_complete"  not in st.session_state:
    st.session_state.interview_complete=False

with st.sidebar:
    st.title("Interview Setup")
    st.divider()
    resume_file=st.file_uploader(
        "Upload your Resume Here (PDF format)",
        type=["pdf"],
        help="Please upload your resume in PDF format only."
    )
    job_description=st.text_area(
        "Paste the Job Description here",
        height=200,
        help="Please paste the full job description for the role you're applying to."
    )
    role=st.text_input(
        "Role you are applying for",
        placeholder="e.g. Data Scientist"
    )

    st.divider()
    start_button=st.button(
        "Start Interview",
        type="primary",
        disabled=st.session_state.interview_active
    )
    if st.session_state.interview_active:
        if st.button("Reset_Interview"):
            st.session_state.agent=None
            st.session_state.messages=[]
            st.session_state.interview_active=False
            st.session_state.interview_complete=False
            st.rerun()
if start_button:
    if not resume_file:
        st.sidebar.error("Please upload your resume PDF.")
    elif not job_description:
        st.sidebar.error("Please paste a job description.")
    elif not role:
        st.sidebar.error("Please enter the role you are applying for.")
    else:
        with st.spinner("Reading your resume and setting up the interview..."):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:
                tmp.write(resume_file.read())
                tmp_path=tmp.name
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)
            raw_text=extract_text_from_pdf(tmp_path)
            chunks=chunk_text(raw_text)
            resume_summary=extract_skills(raw_text,llm)
            name_response = llm.invoke(
                f"Extract only the candidate's full name from this resume text. "
                f"Return just the name, nothing else.\n\n{raw_text[:500]}"
            )
            candidate_name = name_response.content.strip()
            vectorstore=build_vector_store(chunks)
            os.unlink(tmp_path)
            agent=InterviewerAgent(
                llm=llm,
                vectorstore=vectorstore,
                candidate_name=candidate_name,
                role=role,
                resume_summary=resume_summary,
                job_description=job_description

            )
            first_question=agent.start_interview()

            st.session_state.agent=agent
            st.session_state.interview_active=True
            st.session_state.messages.append({
                "role":"assistant",
                "content":first_question
            })
            st.rerun()
# ── Main chat area ───────────────────────────────────────────
st.title("AI Interview Simulator")
st.caption("Powered by Gemini · Your resume is analyzed in real time")
if not st.session_state.interview_active:
    # Show instructions before interview starts
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
    # Display all messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Check if interview is complete
    if st.session_state.interview_complete:
        st.success(
            "Interview complete! Go to the sidebar and click "
            "Reset to start a new interview."
        )

    else:
        # Chat input — where candidate types their answer
        user_input = st.chat_input("Type your answer here...")

        if user_input:
            # Add candidate message to display
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            # Get Alex's response
            with st.spinner("Alex is thinking..."):
                response = st.session_state.agent.get_response(user_input)

            # Add Alex's response to display
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            # Check if interview is over
            if "concludes our interview" in response.lower():
                st.session_state.interview_complete = True

            st.rerun()
          


