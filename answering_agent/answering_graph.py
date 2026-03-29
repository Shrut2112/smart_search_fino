from typing import List, Literal
from langgraph.graph import StateGraph, END, START
from utils.schema import AnswerState, RefinedQuery
from answering_agent.retrival_class import RetrievalPipeline
from utils.get_embedd_model import embedding_model
from answering_agent.db_ret import retrieve_similar_chunks, retrieve_similar_chunks_key
from utils.prompts.retrival_prompts import refine_query_prompt,answering_prompt
from utils.get_llm import get_llm
from utils.get_cross_encoder import get_crossencoder
from langchain_core.prompts import ChatPromptTemplate
from answering_agent.caching_logic import cache_check_node, push_cache
import tiktoken
import json
# from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod
from utils.logger import get_logger

log = get_logger("answering_agent.main")


def get_models():
    llm = get_llm()
    emb_model = embedding_model()
    ans_llm = get_llm()
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
                    log.info(f"Token limit reached. Stopping context addition. Current tokens: {current_token}, Chunk tokens: {chunk_token_count}, Available: {available_tokens}")
                    break
                
                final_chunks.append(c)
                current_token += chunk_token_count
            return "".join(final_chunks)
        except Exception as e:
            log.error(f"Error during tokenization: {e}")
            return "".join(chunks[:2]) # Aggressive fallback
        
def refiner_agent_node(state: AnswerState) -> AnswerState:
    user_query = state['query']
    current_attempts = state.get("attempt_count", 0) + 1
    
    log.info(f"Refiner Agent Invoked - Attempt {current_attempts} for query: {user_query}")
    
    # Use the same prompt template
    prompt = refine_query_prompt.substitute(user_query=user_query)
    content = ""
    try:    
        response = llm.invoke(prompt)
        content = response.content
        
        refined_dict = extract_json_from_text(content)
        
        if refined_dict:
            keyword = refined_dict.get("keyword_query") or refined_dict.get("keyword") or user_query
            semantic = refined_dict.get("semantic_query") or refined_dict.get("semantic") or user_query
            # The refiner prompt returns the detected input language — use it to
            # make the final answer respond in the user's own language.
            detected_lang = refined_dict.get("detected_language", "English")
            
            log.info(f"Refiner Output - Attempt {current_attempts}: Success. Language: {detected_lang}")
            return {
                "keyword_query": keyword,
                "semantic_query": semantic,
                "detected_language": detected_lang,
                "attempt_count": current_attempts
            }

    except Exception as e:
        log.error(f"Refiner Error on Attempt {current_attempts}: {e} {content}. Falling back to raw query.")
        return {
            "keyword_query": user_query,
            "semantic_query": user_query,
            "detected_language": "English",
            "attempt_count": current_attempts
        }

def semantic_search_node(state: AnswerState):
    log.info(f"Semantic Search Invoked")
    query = state.get('semantic_query') or state['query']
    
    if not query: return {"retrived_sem_doc": []}
    
    try:
        query_embedding = emb_model.embed_query(query)
        docs = retrieve_similar_chunks(query_embedding, top_k)
        log.info(f"Semantic Search found {len(docs)} chunks.")
        return {"retrived_sem_doc": docs or []}
    except Exception as e:
        log.error(f"Semantic Search Error for query: {query} - {e}")
        return {"retrived_sem_doc": []}

def keyword_search_node(state: AnswerState):
    log.info(f"Keyword Search Invoked.")
    query = state.get('keyword_query') or state['query']
    try:
        clean_query = query.replace("(","").replace(")","").replace("|","")
        words = [w for w in clean_query.split()]
        lenient_query = " | ".join(words)
        docs = retrieve_similar_chunks_key(lenient_query, top_k) 
        log.info(f"Keyword Search found {len(docs)} chunks.")
        return {"retrived_key_doc": docs}
    except Exception as e:
        log.error(f"Keyword Search Error for query: {query} - {e}")
        return {"retrived_key_doc": []}

def rerank_doc_node(state: AnswerState):
    #print(f"--- [3] RRF MERGING NODE ---")
    log.info(f"RRF Merging Invoked.")
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
    
    log.info(f"RRF Merging completed. Total unique chunks after merging: {len(final_chunks)}")
    return {"reranked_docs": final_chunks}

def get_final_context_node(state: AnswerState):
    log.info(f"Cross-Encoder Reranking Invoked.")
    docs = state.get('reranked_docs', [])
    if not docs:
        log.info("No documents to rerank. Skipping to answer generation.")
        return {"final_doc": []}

    unique_docs = {doc[0]: doc for doc in docs}
    deduplicated_docs = list(unique_docs.values())
    # Use the refined English semantic_query for cross-encoding.
    # The raw query may be in Hindi/Hinglish, but docs are in English —
    # using the translated semantic_query gives accurate reranking scores.
    query = state.get('semantic_query') or state['query']
    
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
    log.info(f"Cross-Encoder reranking completed. Top doc score: {top_docs[0]['rerank_score'] if top_docs else 'N/A'}")
    return {"final_doc": top_docs}

import re

def extract_json_from_text(text):
    """
    Finds and extracts the first valid JSON object from a string.
    Handles cases where the LLM adds conversational text before/after.
    """
    try:
        # Look for the first '{' and the last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            return None
            
        json_str = text[start_idx:end_idx + 1]
        return json.loads(json_str)
    except Exception:
        return None

def answer_agent_node(state: AnswerState):
    log.info(f"Answer Generation Invoked.")
    try:
        # Check if we failed the 3-try limit
        if not state.get('final_doc') and state.get('attempt_count', 0) >= 3:
            log.info("Answer Generation Failed: No documents found after 3 attempts.")
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
            log.info("No high-confidence documents available for answer generation.")
            return {"answer": "I couldn't find specific info to answer that question.","skip_cache":True}

        user_query = state.get('query', "")
        detected_language = state.get('detected_language', 'English')

        prompt = ChatPromptTemplate.from_messages([
            ("system", answering_prompt),
            ("human", "### CONTEXT:\n{clean_entry}\n\n### USER QUESTION:\n{user_query}\n\n### LANGUAGE INSTRUCTION:\nThe user's query was written in **{detected_language}**. You MUST write your `final_answer` entirely in **{detected_language}**. Do not switch to English unless the user's language is English.")
        ])

        final_chunk = limit_context_by_tokens(clean_entry, prompt, user_query)
        final_prompt = prompt.invoke({"clean_entry": final_chunk, "user_query": user_query, "detected_language": detected_language})
        
        raw_response = ans_llm.bind(response_format={"type":"json_object"}).invoke(final_prompt)
        
        try:
            content = raw_response.content
            parsed_json = extract_json_from_text(content)

            if parsed_json:
                answer = parsed_json.get("final_answer")
                if answer:
                    log.info(f"Answer Generation Successful. Answer length: {len(answer)} characters.")
                    return {"answer": answer, "skip_cache": False}

            if len(content) > 50:
                log.warning("JSON failed but content is substantial. Using raw content.")
                return {"answer": content.strip(), "skip_cache": False}
            
            log.error(f"JSON failed and content is not substantial. Returning error message. {content}")
            return {"answer": f"I'm sorry, I ran into an error while drafting your answer.","skip_cache":True}
        except (json.JSONDecodeError, ValueError) as e:
            log.error(f"Answer Generation JSON Parsing Error: {e}. Raw response: {raw_response.content}")
            return {"answer": "I'm sorry, I ran into an error while drafting your answer.","skip_cache":True}
        
    except Exception as e:
        log.error(f"Answer Generation Error: {e}")
        return {"answer": "I'm sorry, I ran into an error while drafting your answer.", "skip_cache":True}

def route_after_retrieval(state: AnswerState):
    doc_count = len(state.get("final_doc", []))
    attempt = state.get("attempt_count", 0)
    
    if attempt >= 3:
        #print(f"Routing: TERMINATE (Failed after {attempt} attempts)")
        log.error(f"Routing Decision: TERMINATE - No documents found after {attempt} attempts for query: {state.get('query')}")
        return "fail"
    
    if doc_count > 0:
        #print(f"Routing: PROCEED (Found {doc_count} docs)")
        log.info(f"Routing Decision: PROCEED - Found {doc_count} documents.")
        return "generate"
    
    print(f"Routing: RETRY (Attempt {attempt} yielded no results)")
    return "retry"

def route_cached(state:AnswerState):
    cache_hit = state['cache_hit']
    #print(cache_hit)
    if cache_hit == True:
        log.info("Cache hit detected. Directly Answering from Cache.")
        return "done"
    log.info("No cache hit. Routing to retrieval and answer generation.")
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
        log.info("Answer quality indicates failure. Skipping cache push.")
        return "skip"
    
    log.info("Answer deemed suitable for caching. Executed cache push.")
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