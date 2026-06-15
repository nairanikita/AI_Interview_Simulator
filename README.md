# AI Interview Simulator

An AI-powered mock interview platform. Upload your resume, paste a job description, and get interviewed by an AI interviewer that asks questions tailored to your background and the role.

---

## How it works

1. Upload your resume (PDF)
2. Paste the job description
3. Enter the role you're applying for
4. Chat with Alex, your AI interviewer
5. Get a scored report at the end covering technical skills, communication, problem solving, experience, and confidence

---

## Tech stack

- **Streamlit** — UI
- **Gemini 2.0 Flash** — LLM for question generation and scoring
- **LangChain** — LLM orchestration and chat history
- **ChromaDB** — in-memory vector store for resume retrieval
- **pdfplumber** — PDF text extraction

---

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/nairanikita/AI_Interview_Simulator.git
cd AI_Interview_Simulator
```

**2. Create a virtual environment**
```bash
python -m venv virenv
source virenv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```
Open `.env` and fill in your `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey).

**5. Run the app**
```bash
streamlit run app.py
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key from Google AI Studio |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key (only needed if tracing is on) |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Under **Advanced settings → Secrets**, add your `GOOGLE_API_KEY`
5. Click **Deploy**
