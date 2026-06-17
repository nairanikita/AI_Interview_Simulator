from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()


def format_transcript(chat_history:list)->str:
    transcript=""
    for message in chat_history:
        if isinstance(message,HumanMessage):
            if message.content=="[Interview started]":
                continue
            transcript+=f'CANDIDATE:{message.content}\n\n'
        elif isinstance(message,AIMessage):
            transcript+=f'INTERVIEWER:{message.content}\n\n'

    return transcript.strip()

def generate_score_report(chat_history:list, role:str,llm)->dict:
    transcript=format_transcript(chat_history)
    prompt=f"""
    You are an expert hiring manager reviewing an interview transcript.
    The candidate was interviewing for the role of: {role}

    INTERVIEW TRANSCRIPT:
    {transcript}

    Evaluate the candidate and return your assessment in EXACTLY this format,
    with no extra text before or after:

    TECHNICAL_SCORE: [0-10]
    TECHNICAL_FEEDBACK: [one sentence]

    COMMUNICATION_SCORE: [0-10]
    COMMUNICATION_FEEDBACK: [one sentence]

    PROBLEM_SOLVING_SCORE: [0-10]
    PROBLEM_SOLVING_FEEDBACK: [one sentence]

    EXPERIENCE_SCORE: [0-10]
    EXPERIENCE_FEEDBACK: [one sentence]

    CONFIDENCE_SCORE: [0-10]
    CONFIDENCE_FEEDBACK: [one sentence]

    OVERALL_SCORE: [0-10]

    STRENGTHS:
    - [strength 1]
    - [strength 2]

    AREAS_TO_IMPROVE:
    - [area 1]
    - [area 2]

    HIRING_RECOMMENDATION: [Strong Yes / Yes / Maybe / No]
    RECOMMENDATION_REASON: [two sentences explaining the recommendation]
    """
    response=llm.invoke([HumanMessage(content=prompt)])
    raw_text=response.content
    return parse_score_report(raw_text)

def parse_score_report(raw_text:str)->dict:
    report = {
        "technical": {"score": 0, "feedback": ""},
        "communication": {"score": 0, "feedback": ""},
        "problem_solving": {"score": 0, "feedback": ""},
        "experience": {"score": 0, "feedback": ""},
        "confidence": {"score": 0, "feedback": ""},
        "overall_score": 0,
        "strengths": [],
        "areas_to_improve": [],
        "hiring_recommendation": "",
        "recommendation_reason": ""
    }
    lines=raw_text.strip().split("\n")
    current_section=None
    for line in lines:
        line=line.strip()
        if not line:
            continue

        if line.startswith("TECHNICAL_SCORE:"):
            report["technical"]["score"] = int(
                line.split(":")[1].strip()
            )
        elif line.startswith("TECHNICAL_FEEDBACK:"):
            report["technical"]["feedback"] = line.split(":", 1)[1].strip()

        elif line.startswith("COMMUNICATION_SCORE:"):
            report["communication"]["score"] = int(
                line.split(":")[1].strip()
            )
        elif line.startswith("COMMUNICATION_FEEDBACK:"):
            report["communication"]["feedback"] = line.split(":", 1)[1].strip()

        elif line.startswith("PROBLEM_SOLVING_SCORE:"):
            report["problem_solving"]["score"] = int(
                line.split(":")[1].strip()
            )
        elif line.startswith("PROBLEM_SOLVING_FEEDBACK:"):
            report["problem_solving"]["feedback"] = line.split(":", 1)[1].strip()

        elif line.startswith("EXPERIENCE_SCORE:"):
            report["experience"]["score"] = int(
                line.split(":")[1].strip()
            )
        elif line.startswith("EXPERIENCE_FEEDBACK:"):
            report["experience"]["feedback"] = line.split(":", 1)[1].strip()

        elif line.startswith("CONFIDENCE_SCORE:"):
            report["confidence"]["score"] = int(
                line.split(":")[1].strip()
            )
        elif line.startswith("CONFIDENCE_FEEDBACK:"):
            report["confidence"]["feedback"] = line.split(":", 1)[1].strip()

        elif line.startswith("OVERALL_SCORE:"):
            report["overall_score"] = int(line.split(":")[1].strip())

        elif line.startswith("STRENGTHS:"):
            current_section = "strengths"
        elif line.startswith("AREAS_TO_IMPROVE:"):
            current_section = "areas"
        elif line.startswith("HIRING_RECOMMENDATION:"):
            current_section = None
            report["hiring_recommendation"] = line.split(":", 1)[1].strip()
        elif line.startswith("RECOMMENDATION_REASON:"):
            report["recommendation_reason"] = line.split(":", 1)[1].strip()

        elif line.startswith("- "):
            if current_section == "strengths":
                report["strengths"].append(line[2:])
            elif current_section == "areas":
                report["areas_to_improve"].append(line[2:])

    return report


def extract_qa_pairs(chat_history: list) -> list[dict]:
    """Return list of {question, answer} dicts from the chat history."""
    pairs = []
    last_question = None
    for msg in chat_history:
        if isinstance(msg, AIMessage):
            last_question = msg.content.strip()
        elif isinstance(msg, HumanMessage):
            if msg.content == "[Interview started]":
                continue
            if last_question:
                pairs.append({"question": last_question, "answer": msg.content.strip()})
                last_question = None
    return pairs


def generate_answer_guide(chat_history: list, role: str, llm) -> list[dict]:
    """
    For each Q&A exchange, ask the LLM for an ideal answer and the key points
    the candidate missed. Returns a list of dicts with keys:
      question, candidate_answer, ideal_answer, missed_points
    """
    pairs = extract_qa_pairs(chat_history)
    if not pairs:
        return []

    results = []
    for pair in pairs:
        prompt = f"""You are a career coach reviewing a mock interview for the role of {role}.

INTERVIEWER QUESTION:
{pair['question']}

CANDIDATE'S ACTUAL ANSWER:
{pair['answer']}

Respond in EXACTLY this format with no extra text:

IDEAL_ANSWER: [2-4 sentences showing what a strong answer would sound like, using specific technical terms relevant to the role]
MISSED_POINTS:
- [key point or keyword they should have mentioned]
- [key point or keyword they should have mentioned]
- [key point or keyword they should have mentioned]
WHAT_THEY_DID_WELL: [one sentence, or "N/A" if nothing stood out]"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        ideal = ""
        missed = []
        did_well = ""
        section = None

        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("IDEAL_ANSWER:"):
                ideal = line.split(":", 1)[1].strip()
                section = None
            elif line.startswith("MISSED_POINTS:"):
                section = "missed"
            elif line.startswith("WHAT_THEY_DID_WELL:"):
                did_well = line.split(":", 1)[1].strip()
                section = None
            elif line.startswith("- ") and section == "missed":
                missed.append(line[2:])

        results.append({
            "question": pair["question"],
            "candidate_answer": pair["answer"],
            "ideal_answer": ideal,
            "missed_points": missed,
            "did_well": did_well,
        })

    return results

