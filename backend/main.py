from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",  # Your React/Vue local dev port
    "http://localhost:8501",  # Default Streamlit port
    "http://localhost:5173",  # Vite default port
    "http://127.0.0.1:5173",  # Vite default IP
    "https://your-fino-app.com", # Your production domain
]


try:
    from utils.logger import get_logger
    log = get_logger("backend")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("backend")

from answering_agent.answering_graph import create_graph

# Global dictionary to store models/graphs
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the LangGraph graph at startup
    log.info("Starting up FastAPI application...")
    log.info("Loading LangGraph agent...")

    try:
        ml_models["graph"] = create_graph()
        log.info("LangGraph agent loaded successfully.")

    except Exception as e:
        log.error(f"Failed to load LangGraph agent: {e}")
        # Not raising here to allow startup, but health check could fail if critical.
        # Alternatively, could raise e to prevent startup.
    yield
    # Clean up on shutdown
    log.info("Shutting down FastAPI application...")
    ml_models.clear()

app = FastAPI(
    title="Fino Smart Search API", 
    description="Backend API for Fino Smart Search Agent",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure service is up and graph is loaded."""
    if "graph" in ml_models and ml_models["graph"] is not None:
        return {"status": "healthy", "graph_loaded": True}
    return {"status": "degraded", "graph_loaded": False}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint to interact with the LangGraph answering agent."""
    graph = ml_models.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Answering agent graph is not loaded or available.")
    
    log.info(f"Received query: {request.query}")
    try:
        # Invoke the graph using the same format as the original app
        config = {"configurable": {"thread_id": request.thread_id}}
        result = await graph.ainvoke({"query": request.query}, config)
        answer = result.get("answer", "I'm sorry, I couldn't process that request.")
        
        log.info("Successfully processed query.")
        return ChatResponse(answer=answer)
    except Exception as e:
        log.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
