from answering_agent.answering_graph import create_graph
from utils.get_embedd_model import embedding_model
from utils.get_llm import get_llm,get_gpt
from utils.get_cross_encoder import get_crossencoder
from utils.logger import get_logger

log = get_logger("main")


def get_answer(graph,user_query):
    log.info(f"User query: {user_query}")
    refined_query = graph.invoke({"query":user_query})
    log.info(f"Answer generated")
    print(refined_query['answer'])
    
if __name__ == "__main__":
    
    graph = create_graph()
    
    query =  input("query: ")
    while query != "end":
        get_answer(graph, query)
        query =  input("query: ")