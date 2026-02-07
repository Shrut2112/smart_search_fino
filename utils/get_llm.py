from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
def get_llm():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    try:
        llm = ChatGroq(model="openai/gpt-oss-120b")
    except:
        print("Error Loading Model")
        return
    return llm
