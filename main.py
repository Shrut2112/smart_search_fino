from answering_agent.retrival_class import RetrievalPipeline
from utils.get_embedd_model import embedding_model
from utils.get_llm import get_llm,get_gpt
from utils.get_cross_encoder import get_crossencoder

def get_answer(retrieval_pipeline,user_query):
    refined_query = retrieval_pipeline.generate_response(user_query)
    print(refined_query)
if __name__ == "__main__":
    embedding_model = embedding_model()
    llm = get_llm()
    ans_llm = get_gpt()
    reranker = get_crossencoder()
    retrieval_pipeline = RetrievalPipeline(embedding_model, llm,ans_llm,reranker)
    
    query =  input("query: ")
    while query != "end":
        get_answer(retrieval_pipeline, query)
        query =  input("query: ")