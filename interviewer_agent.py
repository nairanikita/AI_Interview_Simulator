# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate,HumanMessagePromptTemplate,HumanMessagePromptTemplate, SystemMessagePromptTemplate
# from dotenv import load_dotenv
# SYSTEM_PROMPT = """
# You are Alex, a senior technical interviewer at a top tech company.
# You are interviewing a candidate for the role of {role}.

# CANDIDATE PROFILE:
# {resume_summary}

# JOB DESCRIPTION:
# {job_description}

# YOUR RULES — follow these strictly:
# 1. Ask ONE question at a time. Never ask two questions together.
# 2. If the answer is vague or shallow, probe deeper with a follow-up:
#    - "Can you be more specific about that?"
#    - "Walk me through the technical details."
#    - "What was YOUR specific contribution to that?"
# 3. Reference specific items from the candidate's resume when probing.
# 4. Cover these areas across the interview:
#    - Technical skills mentioned in both resume and job description
#    - Past projects — go deep on implementation details
#    - System design decisions they made
#    - Behavioral questions using STAR format
# 5. If the candidate gives a strong answer, move to the next topic.
# 6. After 8-10 exchanges, say exactly: "Thank you, that concludes our interview."
# 7. Never reveal you are an AI. Stay in character as Alex.
# 8. Be professional but direct. This is a real technical interview.

# Use the retrieved resume context below to ask informed questions.
# """

# def build_interview_chain(llm,vectorstore,role:str,resume_summary:str,job_description:str):
#     memory=ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer"
#     )
#     filled_system_prompt=SYSTEM_PROMPT.format(
#         role=role,
#         resume_summary=resume_summary,
#         job_description=job_description
#     )
#     retriever=vectorstore.as_retriever(search_kwargs={"k":3})
#     chain=ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         return_source_documents=True,
#         verbose=False

#     )
#     return chain,filled_system_prompt

# def get_interviewer_reponse(chain,system_prompt:str,user_message:str)->str:
#     full_query=f"{system_prompt}\n\nCandidate just said:{user_message}"
#     result=chain.invoke({"question":full_query})
#     return result["answer"]

# def start_interview(chain,system_prompt:str)->str:
#     opening=(
#         f"{system_prompt}\n\n"
#         "Start the inteview now with a warm but professional greeting"
#         "and your first question.Remember to ask only one question at a time."
#     )
#     result=chain.invoke({"question":opening})
#     return result["answer"]

# if __name__=="__main__":
#     from document_processor import extract_text_from_pdf,chunk_text,extract_skills
#     from vector_store import build_vector_store
#     pdf_path="test_files/resume.pdf"

#     JOB_DESCRIPTION = """
#     Senior Data Engineer at a fintech company.
#     Requirements:
#     - 3+ years experience with Python and SQL
#     - Experience with cloud platforms (AWS or GCP)
#     - Built and maintained data pipelines at scale
#     - Experience with ML model deployment
#     - Strong communication skills
#     """

#     print("=" * 50)
#     print("Setting up interview...")
#     print("=" * 50)
#     llm=ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temprature=0.7
#     )
#     raw_text=extract_text_from_pdf(pdf_path)
#     chunks=chunk_text(raw_text)
#     resume_summary=extract_skills(raw_text,llm)
#     vectorstore=build_vector_store(chunks)

#     chain,system_prompt=build_interview_chain(
#         llm=llm,
#         vectorstore=vectorstore,
#         role="Senior DataEngineer",
#         resume_summary=resume_summary,
#         job_description=JOB_DESCRIPTION
#     )

#     print("\nInterview starting...\n")
#     print("=" * 50)

#     # Start the interview
#     first_question = start_interview(chain, system_prompt)
#     print(f"Interviewer: {first_question}")
#     print("=" * 50)

#     # Simulate 3 candidate responses
#     test_responses = [
#         "I have 3 years of experience at Deloitte working on data pipelines using PySpark and BigQuery",
#         "For the TransitFlow project I built real-time pipelines processing 30 million daily transit events",
#         "I used XGBoost and Prophet for time series forecasting on ridership data"
#     ]

#     for response in test_responses:
#         print(f"\nCandidate: {response}")
#         print("-" * 50)
#         interviewer_reply=get_interviewer_reponse(chain,system_prompt,response)
#         print(f"Interviewer: {interviewer_reply}")
#         print("=" * 50)

# interviewer_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are Alex, a senior technical interviewer at a top tech company.
You are interviewing a candidate named {candidate_name} for the role of {role}.

CANDIDATE PROFILE:
{resume_summary}

JOB DESCRIPTION:
{job_description}

RETRIEVED RESUME CONTEXT:
{context}

YOUR RULES — follow these strictly:
1. Ask ONE question at a time. Never ask two questions together.
2. If the answer is vague or shallow, probe deeper:
   - "Can you be more specific about that?"
   - "Walk me through the technical details."
   - "What was YOUR specific contribution?"
3. Reference specific items from the candidate's resume when probing.
4. Cover: technical skills, past projects, system design, behavioral (STAR format).
5. If the candidate gives a strong answer, move to the next topic.
6. After 8-10 exchanges say exactly: "Thank you, that concludes our interview."
7. Never reveal you are an AI. Stay in character as Alex.
8. Be professional but direct. This is a real technical interview."""


class InterviewerAgent:
    

    def __init__(self, llm, vectorstore,candidate_name:str,role: str,
                 resume_summary: str, job_description: str):

        self.llm = llm
        self.vectorstore = vectorstore
        self.candidate_name = candidate_name
        self.role = role
        self.resume_summary = resume_summary
        self.job_description = job_description

        
        self.chat_history = []

        self.output_parser = StrOutputParser()

    def _get_context(self, query: str) -> str:
       
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    def _build_system_message(self, context: str) -> str:
        
        return SYSTEM_PROMPT.format(
            candidate_name=self.candidate_name,
            role=self.role,
            resume_summary=self.resume_summary,
            job_description=self.job_description,
            context=context
        )

    def start_interview(self) -> str:
        
        context = self._get_context(
            f"introduction opening question for {self.role}"
        )
        system_message = self._build_system_message(context)

       
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=(
                "Start the interview with a professional greeting "
                "and your first question. ONE question only."
            ))
        ]

        response = self.llm.invoke(messages)
        reply = self.output_parser.invoke(response)

        
        self.chat_history.append(
            HumanMessage(content="[Interview started]")
        )
        self.chat_history.append(AIMessage(content=reply))

        return reply

    def get_response(self, candidate_answer: str) -> str:
       

        context = self._get_context(candidate_answer)
        system_message = self._build_system_message(context)

      
        messages = [SystemMessage(content=system_message)]
        messages.extend(self.chat_history)
        messages.append(HumanMessage(content=candidate_answer))

        response = self.llm.invoke(messages)
        reply = self.output_parser.invoke(response)

        # save this exchange to history for next turn
        self.chat_history.append(HumanMessage(content=candidate_answer))
        self.chat_history.append(AIMessage(content=reply))

        return reply

    def get_history(self) -> list:
      
        
        return self.chat_history


# ── runs only when you execute this file directly ──
if __name__ == "__main__":
    from document_processor import extract_text_from_pdf, chunk_text, extract_skills
    from vector_store import build_vector_store

    PDF_PATH = "testfiles/resume.pdf"

    JOB_DESCRIPTION = """
    Senior Data Engineer at a fintech company.
    Requirements:
    - 3+ years experience with Python and SQL
    - Experience with cloud platforms (AWS or GCP)
    - Built and maintained data pipelines at scale
    - Experience with ML model deployment
    - Strong communication skills
    """

    print("=" * 50)
    print("Setting up interview...")
    print("=" * 50)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7
    )
    raw_text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(raw_text)
    resume_summary = extract_skills(raw_text, llm)
    vectorstore = build_vector_store(chunks)

    
    agent = InterviewerAgent(
        llm=llm,
        vectorstore=vectorstore,
        role="Senior Data Engineer",
        resume_summary=resume_summary,
        job_description=JOB_DESCRIPTION
    )

    print("\nInterview starting...\n")
    print("=" * 50)

    first_question = agent.start_interview()
    print(f"Alex: {first_question}")
    print("=" * 50)

    
    test_responses = [
        "I have 3 years of experience at Deloitte working on data pipelines using PySpark and BigQuery",
        "For the TransitFlow project I built real-time pipelines processing 30 million daily transit events",
        "I used XGBoost and Prophet for time series forecasting on ridership data"
    ]

    for response in test_responses:
        print(f"\nCandidate: {response}")
        print("-" * 50)
        reply = agent.get_response(response)
        print(f"Alex: {reply}")
        print("=" * 50)