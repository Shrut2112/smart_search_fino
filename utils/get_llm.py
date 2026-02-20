from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from utils.logger import get_logger

log = get_logger("utils.llm")

load_dotenv()
def get_llm():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    try:
        llm = ChatGroq(model="qwen/qwen3-32b",temperature=0)
    except:
        log.error("Error loading LLM model")
        return
    return llm

def get_gpt():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    try:
        llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0)
    except:
        log.error("Error loading GPT model")
        return
    return llm