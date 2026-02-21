from typing import List, Literal
from langgraph.graph import StateGraph, END, START
from utils.schema import AnswerState, RefinedQuery
from answering_agent.retrival_class import RetrievalPipeline
from utils.get_embedd_model import embedding_model
from answering_agent.db_ret import retrieve_similar_chunks, retrieve_similar_chunks_key
from utils.prompts.retrival_prompts import refine_query_prompt,answering_prompt
from utils.get_llm import get_llm, get_gpt
from utils.get_cross_encoder import get_crossencoder
from langchain_core.prompts import ChatPromptTemplate
from answering_agent.caching_logic import cache_check_node, push_cache
import tiktoken
import json
# from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod

def get_models():
    llm = get_llm()
    emb_model = embedding_model()
    ans_llm = get_gpt()
    reranker = get_crossencoder()
    return llm, emb_model, ans_llm, reranker

llm, emb_model, ans_llm, reranker = get_models()
retriever = RetrievalPipeline(emb_model, llm, ans_llm, reranker)
encoding = tiktoken.get_encoding("cl100k_base")
top_k = 30

    
def limit_context_by_tokens(chunks,prompt,query,max_limit=6000):
        if not chunks: return ""
        try:
            static_text = prompt.format(clean_entry="",user_query=query)
            static_tokens = len(encoding.encode(static_text))
            available_tokens = max_limit - static_tokens - 1000
        
            final_chunks = []
            current_token = 0
            for c in chunks:
                text_to_encode = str(c)
                chunk_token_count = len(encoding.encode(text_to_encode))
            
                if current_token + chunk_token_count > available_tokens:
                    #print(f"Token limit reached. Skipping remaining {len(chunks) - len(final_chunks)} chunks.")
                    break
                
                final_chunks.append(c)
                current_token += chunk_token_count
            return "".join(final_chunks)
        except Exception as e:
            print(f"Tokenization Error: {e}")
            return "".join(chunks[:2]) # Aggressive fallback
        
def refiner_agent_node(state: AnswerState) -> AnswerState:
    #print(f"\n--- [1] REFINER NODE ---")
    user_query = state['query']
    current_attempts = state.get("attempt_count", 0) + 1
    #print(f"User Query: {user_query}")
    #print(f"Attempt: {current_attempts}/3")
    
    prompt = refine_query_prompt.substitute(user_query=user_query)
    
    try:    
        refined = llm.with_structured_output(RefinedQuery).invoke(prompt)
        #print(f"Refined - Semantic: {refined.semantic_query}")
        #print(f"Refined - Keyword: {refined.keyword_query}")
        return {
            "keyword_query": refined.keyword_query,
            "semantic_query": refined.semantic_query,
            "attempt_count": current_attempts
        }
    except Exception as e:
        #print(f"Refiner Error: {e}. Using raw query as fallback.")
        return {
            "keyword_query": user_query,
            "semantic_query": user_query,
            "attempt_count": current_attempts
        }

def semantic_search_node(state: AnswerState):
    #print(f"--- [2A] SEMANTIC SEARCH START ---")
    query = state.get('semantic_query') or state['query']
    
    if not query: return {"retrived_sem_doc": []}
    
    try:
        query_embedding = emb_model.embed_query(query)
        docs = retrieve_similar_chunks(query_embedding, top_k)
        #print(f"Semantic Result: {len(docs)} chunks found.")
        return {"retrived_sem_doc": docs or []}
    except Exception as e:
        print(f"Semantic Search Error: {e}")
        return {"retrived_sem_doc": []}

def keyword_search_node(state: AnswerState):
    #print(f"--- [2B] KEYWORD SEARCH START ---")
    query = state.get('keyword_query') or state['query']
    try:
        clean_query = query.replace("(","").replace(")","").replace("|","")
        words = [w for w in clean_query.split()]
        lenient_query = " | ".join(words)
        docs = retrieve_similar_chunks_key(lenient_query, top_k) 
        #print(f"Keyword Result: {len(docs)} chunks found.")
        return {"retrived_key_doc": docs}
    except Exception as e:
        #print(f"Keyword Search Error: {e}")
        return {"retrived_key_doc": []}

def rerank_doc_node(state: AnswerState):
    #print(f"--- [3] RRF MERGING NODE ---")
    k = 60
    rrf_scores = {} 
    semantic_doc = state.get('retrived_sem_doc', [])
    keyword_doc = state.get('retrived_key_doc', [])
    
    for rank, chunk in enumerate(semantic_doc):
        chunk_id = chunk[0] 
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        
    for rank, chunk in enumerate(keyword_doc):
        chunk_id = chunk[0]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    
    all_chunks = {c[0]: c for c in semantic_doc + keyword_doc}
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_chunks = [all_chunks[chunk_id] for chunk_id, score in sorted_ids]
    
    #print(f"Merged {len(semantic_doc)} semantic and {len(keyword_doc)} keyword docs into {len(final_chunks)} unique chunks.")
    return {"reranked_docs": final_chunks}

def get_final_context_node(state: AnswerState):
    #print(f"--- [4] CROSS-ENCODER RERANKING NODE ---")
    docs = state.get('reranked_docs', [])
    if not docs:
        # print("No documents found to rerank.")
        return {"final_doc": []}

    unique_docs = {doc[0]: doc for doc in docs}
    deduplicated_docs = list(unique_docs.values())
    query = state['query']
    
    pairs = [[query, doc[2]] for doc in deduplicated_docs]
    scores = reranker.predict(pairs)
    
    reranked_list = []
    for i, doc in enumerate(deduplicated_docs):
        reranked_list.append({
            'id': doc[0],
            'doc_id': doc[1],
            'text': doc[2],
            'page_no': doc[3].get('page_number',""),
            'chunk_type': "Has Table Data" if doc[3].get('has_tables') else "No table Data", 
            'rerank_score': float(scores[i])
        })
    
    reranked_list.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_docs = reranked_list[:5]
    #print(f"Top Score: {top_docs[0]['rerank_score'] if top_docs else 'N/A'}")
    return {"final_doc": top_docs}

def answer_agent_node(state: AnswerState):
    #print(f"--- [5] GENERATING ANSWER ---")
    try:
        # Check if we failed the 3-try limit
        if not state.get('final_doc') and state.get('attempt_count', 0) >= 3:
            #print("FAILED: No docs found after 3 attempts.")
            return {"answer": "I couldn't find specific info to answer that question after 3 tries.","skip_cache":True}
        
        context_text = state.get('final_doc', [])
        clean_entry = []
        for c in context_text:
                src = c.get('doc_id',"")
                text = c.get('text',"")
                pageno = c.get('page_no',"")
                chunk_type = c.get('chunk_type',"")
                scoren = c.get('rerank_score',"")
                
                #print(f"{scoren}\n")
                entry = f"\n# Source: {src} PAGE NO: {pageno} TABLE CONTENT: {chunk_type} TEXT: {text} RERANK SCORE: {scoren}\n"
                clean_entry.append(entry)
                    
        if not clean_entry:
            #print("No high-confidence docs available.")
            return {"answer": "I couldn't find specific info to answer that question.","skip_cache":True}

        prompt = ChatPromptTemplate.from_messages([
            ("system", answering_prompt),
            ("human", "### CONTEXT:\n{clean_entry}\n ### USER QUESTION:\n{user_query}")
        ])

        user_query = state.get('query', "")
        # Fixed the function call (passing clean_entry directly)
        final_chunk = limit_context_by_tokens(clean_entry, prompt, user_query)
        final_prompt = prompt.invoke({"clean_entry": final_chunk, "user_query": user_query})
        
        #print("Sending to LLM...")
        raw_response = ans_llm.bind(response_format={"type":"json_object"}).invoke(final_prompt)
        
        try:
            content = raw_response.content
            parsed_json = json.loads(content)
            answer = parsed_json.get("final_answer")
            if not answer:
                 return {"answer": "I'm sorry, I ran into an error while drafting your answer.","skip_cache":True}
            #print("Answer generation complete.")
            
            return {"answer": answer,"skip_cache":False}
        except (json.JSONDecodeError, ValueError) as e:
            #print(f"JSON Parsing Error: {e}. Falling back to raw content.")
            return {"answer": "I'm sorry, I ran into an error while drafting your answer.","skip_cache":True}
        
    except Exception as e:
        #print(f"Answering Error: {e}")
        return {"answer": "I'm sorry, I ran into an error while drafting your answer.", "skip_cache":True}

def route_after_retrieval(state: AnswerState):
    doc_count = len(state.get("final_doc", []))
    attempt = state.get("attempt_count", 0)
    
    if attempt >= 3:
        #print(f"Routing: TERMINATE (Failed after {attempt} attempts)")
        return "fail"
    
    if doc_count > 0:
        #print(f"Routing: PROCEED (Found {doc_count} docs)")
        return "generate"
    
    print(f"Routing: RETRY (Attempt {attempt} yielded no results)")
    return "retry"

def route_cached(state:AnswerState):
    cache_hit = state['cache_hit']
    #print(cache_hit)
    if cache_hit == True:
        return "done"
    return "process_graph"

def route_to_cache(state: AnswerState):
    if state.get("skip_cache", False):
        return "skip"
    failure_phrases = [
        "i'm sorry", 
        "i don't know", 
        "couldn't find", 
        "error",
        "failed after 3 tries"
    ]
    answer = state.get("answer", "").lower()
    if any(phrase in answer for phrase in failure_phrases):
        # print("--- [ROUTER] Answer quality low. Skipping Cache. ---")
        return "skip"
    return "push"

def create_graph():
    builder = StateGraph(AnswerState)

    builder.add_node("cache_check", cache_check_node)
    builder.add_node("refiner", refiner_agent_node)
    builder.add_node("semantic_search", semantic_search_node)
    builder.add_node("keyword_search", keyword_search_node)
    builder.add_node("rerank_rrf", rerank_doc_node)
    builder.add_node("cross_encode", get_final_context_node)
    builder.add_node("answer_node", answer_agent_node)
    builder.add_node("push_cache_node", push_cache)

    builder.add_edge(START, "cache_check")
    builder.add_conditional_edges(
        "cache_check",
        route_cached,
        {
            "done": END,
            "process_graph": "refiner"
        }
    )
    
    # Parallelize the searches
    builder.add_edge("refiner", "semantic_search")
    builder.add_edge("refiner", "keyword_search")
    
    # Join searches into RRF
    builder.add_edge("semantic_search", "rerank_rrf")
    builder.add_edge("keyword_search", "rerank_rrf")
    builder.add_edge("rerank_rrf", "cross_encode")

    # The Logic Diamond: Check if we have docs or need to retry
    builder.add_conditional_edges(
        "cross_encode",
        route_after_retrieval,
        {
            "generate": "answer_node",
            "retry": "refiner",
            "fail": "answer_node" # Will trigger the error message in the node
        }
    )
    
    builder.add_conditional_edges(
        "answer_node",
        route_to_cache,
        {
            "push": "push_cache_node",
            "skip": END
        }
    )
    builder.add_edge("push_cache_node", END)

    graph = builder.compile()
    # img_path = "langgraph_diagram.png"
    # graph.get_graph().draw_mermaid_png(output_file_path=img_path, draw_method=MermaidDrawMethod.API)

    return graph