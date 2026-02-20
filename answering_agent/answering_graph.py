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
import tiktoken
import json
from utils.logger import get_logger

log = get_logger("answering.graph")

def get_models():
    llm = get_llm()
    emb_model = embedding_model()
    ans_llm = get_gpt()
    reranker = get_crossencoder()
    return llm, emb_model, ans_llm, reranker

llm, emb_model, ans_llm, reranker = get_models()
retriever = RetrievalPipeline(emb_model, llm, ans_llm, reranker)
encoding = tiktoken.get_encoding("cl100k_base")
top_k = 10

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
                    log.info(f"Token limit reached. Skipping remaining {len(chunks) - len(final_chunks)} chunks.")
                    break
                
                final_chunks.append(c)
                current_token += chunk_token_count
            return "".join(final_chunks)
        except Exception as e:
            log.warning(f"Tokenization Error: {e}")
            return "".join(chunks[:2]) # Aggressive fallback
        
def refiner_agent_node(state: AnswerState) -> AnswerState:
    log.info("--- [1] REFINER NODE ---")
    user_query = state['query']
    current_attempts = state.get("attempt_count", 0) + 1
    log.info(f"User Query: {user_query}")
    log.info(f"Attempt: {current_attempts}/3")
    
    prompt = refine_query_prompt.substitute(user_query=user_query)
    
    try:    
        refined = llm.with_structured_output(RefinedQuery).invoke(prompt)
        log.info(f"Refined - Semantic: {refined.semantic_query}")
        log.info(f"Refined - Keyword: {refined.keyword_query}")
        return {
            "keyword_query": refined.keyword_query,
            "semantic_query": refined.semantic_query,
            "attempt_count": current_attempts
        }
    except Exception as e:
        log.warning(f"Refiner Error: {e}. Using raw query as fallback.")
        return {
            "keyword_query": user_query,
            "semantic_query": user_query,
            "attempt_count": current_attempts
        }

def semantic_search_node(state: AnswerState):
    log.info("--- [2A] SEMANTIC SEARCH START ---")
    query = state.get('semantic_query') or state['query']
    
    if not query: return {"retrived_sem_doc": []}
    
    try:
        query_embedding = emb_model.embed_query(query)
        docs = retrieve_similar_chunks(query_embedding, top_k)
        log.info(f"Semantic Result: {len(docs)} chunks found.")
        return {"retrived_sem_doc": docs or []}
    except Exception as e:
        log.error(f"Semantic Search Error: {e}")
        return {"retrived_sem_doc": []}

def keyword_search_node(state: AnswerState):
    log.info("--- [2B] KEYWORD SEARCH START ---")
    query = state.get('keyword_query') or state['query']
    try:
        clean_query = query.replace("(","").replace(")","").replace("|","")
        words = [w for w in clean_query.split()]
        lenient_query = " | ".join(words)
        docs = retrieve_similar_chunks_key(lenient_query, top_k) 
        log.info(f"Keyword Result: {len(docs)} chunks found.")
        return {"retrived_key_doc": docs}
    except Exception as e:
        log.error(f"Keyword Search Error: {e}")
        return {"retrived_key_doc": []}

def rerank_doc_node(state: AnswerState):
    log.info("--- [3] RRF MERGING NODE ---")
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
    
    log.info(f"Merged {len(semantic_doc)} semantic and {len(keyword_doc)} keyword docs into {len(final_chunks)} unique chunks.")
    return {"reranked_docs": final_chunks}

def get_final_context_node(state: AnswerState):
    log.info("--- [4] CROSS-ENCODER RERANKING NODE ---")
    docs = state.get('reranked_docs', [])
    if not docs:
        log.warning("No documents found to rerank.")
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
    log.info(f"Top Score: {top_docs[0]['rerank_score'] if top_docs else 'N/A'}")
    return {"final_doc": top_docs}

def answer_agent_node(state: AnswerState):
    log.info("--- [5] GENERATING ANSWER ---")
    try:
        # Check if we failed the 3-try limit
        if not state.get('final_doc') and state.get('attempt_count', 0) >= 3:
            log.warning("FAILED: No docs found after 3 attempts.")
            return {"answer": "I couldn't find specific info to answer that question."}
        
        context_text = state.get('final_doc', [])
        clean_entry = []
        for c in context_text:
                src = c.get('doc_id',"")
                text = c.get('text',"")
                pageno = c.get('page_no',"")
                chunk_type = c.get('chunk_type',"")
                scoren = c.get('rerank_score',"")
                if scoren > 0:
                    log.debug(f"Rerank score: {scoren}")
                    entry = f"\n# Source: {src} PAGE NO: {pageno} TABLE CONTENT: {chunk_type} TEXT: {text} RERANK SCORE: {scoren}\n"
                    clean_entry.append(entry)
                    
        if not clean_entry:
            log.warning("No high-confidence docs available.")
            return {"answer": "I couldn't find specific info to answer that question."}

        prompt = ChatPromptTemplate.from_messages([
            ("system", answering_prompt),
            ("human", "### CONTEXT:\n{clean_entry}\n ### USER QUESTION:\n{user_query}")
        ])

        user_query = state.get('query', "")
        # Fixed the function call (passing clean_entry directly)
        final_chunk = limit_context_by_tokens(clean_entry, prompt, user_query)
        final_prompt = prompt.invoke({"clean_entry": final_chunk, "user_query": user_query})
        
        log.info("Sending to LLM...")
        raw_response = ans_llm.bind(response_format={"type":"json_object"}).invoke(final_prompt)
        
        try:
            content = raw_response.content
            parsed_json = json.loads(content)
            answer = parsed_json.get("final_answer")
            if not answer:
                 return {"answer": "I'm sorry, I ran into an error while drafting your answer."}
            log.info("Answer generation complete.")
            return {"answer": answer}
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"JSON Parsing Error: {e}. Falling back to raw content.")
            return {"answer": "I'm sorry, I ran into an error while drafting your answer."}
        
    except Exception as e:
        log.error(f"Answering Error: {e}")
        return {"answer": "I'm sorry, I ran into an error while drafting your answer."}

def route_after_retrieval(state: AnswerState):
    doc_count = len(state.get("final_doc", []))
    attempt = state.get("attempt_count", 0)
    
    if doc_count > 0:
        log.info(f"Routing: PROCEED (Found {doc_count} docs)")
        return "generate"
    
    if attempt >= 3:
        log.info(f"Routing: TERMINATE (Failed after {attempt} attempts)")
        return "fail"
    
    log.info(f"Routing: RETRY (Attempt {attempt} yielded no results)")
    return "retry"

def create_graph():
    builder = StateGraph(AnswerState)

    builder.add_node("refiner", refiner_agent_node)
    builder.add_node("semantic_search", semantic_search_node)
    builder.add_node("keyword_search", keyword_search_node)
    builder.add_node("rerank_rrf", rerank_doc_node)
    builder.add_node("cross_encode", get_final_context_node)
    builder.add_node("answer_node", answer_agent_node)

    builder.add_edge(START, "refiner")
    
    # Parallelize the searches
    builder.add_edge("refiner", "semantic_search")
    builder.add_edge("refiner", "keyword_search")
    
    # Join searches into RRF
    builder.add_edge(["semantic_search", "keyword_search"], "rerank_rrf")
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
    
    builder.add_edge("answer_node", END)

    return builder.compile()