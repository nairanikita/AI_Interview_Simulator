from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=1.0)
response=llm.invoke("Say hello as a strict technical interviewer in one sentence.")
print(response.content)