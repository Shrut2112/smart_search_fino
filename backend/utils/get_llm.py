from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from utils.logger import get_logger
from langchain_openai import ChatOpenAI

log = get_logger("utils.llm")

load_dotenv()
def get_llm():
    api_key = os.getenv("SARVAM_API_KEY")
    model_name = "sarvam-30b"
    if not api_key:
        log.error("SARVAM_API_KEY not found in environment")
        return None

    try:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.sarvam.ai/v1", 
            max_tokens=6000,
            max_completion_tokens=6000,
            temperature=0,
            streaming=True
        )
        return llm
    except Exception as e:
        log.error(f"Error loading Sarvam model {model_name}: {e}")
        return None

def get_refiner_model():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    try:
        llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0)
    except:
        log.error("Error loading GPT model")
        return
    return llm