from tenacity import asyncio
from typing import List, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from utils.schema import AnswerState, RefinedQuery
from answering_agent.retrival_class import RetrievalPipeline
from utils.get_embedd_model import embedding_model
from database.db_ret import retrieve_similar_chunks, retrieve_similar_chunks_key
from utils.prompts.retrival_prompts import refine_query_prompt,answering_prompt
from utils.get_llm import get_llm,get_refiner_model
from utils.get_cross_encoder import get_crossencoder
from langchain_core.prompts import ChatPromptTemplate
from answering_agent.caching_logic import cache_check_node, push_cache
import tiktoken
import json
import asyncio
import os
# from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod
from utils.logger import get_logger

log = get_logger("answering_agent.main")

def get_models():
    llm = get_llm()
    emb_model = embedding_model()
    ans_llm = get_llm()
    refiner_llm = get_refiner_model()
    reranker = get_crossencoder()
    return llm, emb_model, ans_llm, reranker,refiner_llm

llm, emb_model, ans_llm, reranker,refiner_llm = get_models()
# retriever = RetrievalPipeline(emb_model, llm, ans_llm, reranker,refiner_llm)
encoding = tiktoken.get_encoding("cl100k_base")
top_k = 15

    
def limit_context_by_tokens(chunks,prompt,query,language,max_limit=6000):
        if not chunks: return ""
        try:
            static_text = prompt.format(
            clean_entry="", 
            user_query=query, 
            language=language
        )
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
        
async def refiner_agent_node(state: AnswerState) -> AnswerState:
    user_query = state['query']
    full_history = state.get('messages', [])
    current_attempts = state.get("attempt_count", 0) + 1

    recent_history = get_recent_history(full_history, k=4)
    
    log.info(f"Refiner Agent Invoked - Attempt {current_attempts}")

    # Fast-path: only skip refiner if there is NO history AND the query is very short (≤3 words).
    # ≤4 was too aggressive — short Hinglish follow-ups (e.g. "aur 20000 nikalne ho to?") were
    # bypassing the refiner even when conversation context was available.
    if not recent_history and len(user_query.split()) <= 3:
        log.info("Refiner: Fast-Path triggered (No history + Short query ≤3 words)")
        return {
            "keyword_query": user_query,
            "semantic_query": user_query,
            "attempt_count": current_attempts
        }

    # Format history as clean text so the refiner LLM can actually parse it
    formatted_history = format_history_for_refiner(recent_history)

    try:
        prompt = refine_query_prompt.substitute(user_query=user_query, recent_history=formatted_history)

        response = await refiner_llm.ainvoke(prompt)
        
        content = response.content
        refined_dict = extract_json_from_text(content)
        
        if refined_dict:
            keyword = refined_dict.get("keyword") or user_query
            semantic = refined_dict.get("semantic") or user_query
            
            # Validate the refiner actually refined — if it echoed the raw query back, warn and proceed
            if keyword == user_query or semantic == user_query:
                log.warning("Refiner returned raw query unchanged. Quality may be degraded.")
            
            log.info(f"Refiner Output: Query de-referenced successfully {refined_dict}. Semantic - {semantic}, Keyword - {keyword}")
            return {
                "keyword_query": keyword,
                "semantic_query": semantic,
                "attempt_count": current_attempts
            }
    except Exception as e:
        log.error(f"Refiner Error: {e}. Falling back to raw query.")
        return {
            "keyword_query": user_query,
            "semantic_query": user_query,
            "attempt_count": current_attempts
        }


async def semantic_search_node(state: AnswerState):
    log.info(f"Semantic Search Invoked")
    semantic_query = state.get('semantic_query') or state['query']
    original_query = state['query']
    
    if not semantic_query: return {"retrived_sem_doc": []}
    
    try:
        loop = asyncio.get_event_loop()
        
        # Use the refined English semantic_query for embedding if it differs from the raw query.
        # The cache_check_node embeds the raw user query (Hinglish/Hindi), which is a poor
        # match against our English document corpus. The refiner produces a clean English
        # policy-heading — embedding THAT gives significantly better vector search results.
        if semantic_query.strip().lower() != original_query.strip().lower():
            log.info(f"Semantic Search: re-embedding refined query → '{semantic_query}'")
            query_embedding = await emb_model.aembed_query(semantic_query)
        else:
            # Fast-path or refiner echoed raw query — reuse the pre-computed embedding
            query_embedding = state.get('query_embedding')
        
        docs = await loop.run_in_executor(None, retrieve_similar_chunks, query_embedding, top_k)
        log.info(f"Semantic Search found {len(docs)} chunks.")
        return {"retrived_sem_doc": docs or []}
    except Exception as e:
        log.error(f"Semantic Search Error for query: {semantic_query} - {e}")
        return {"retrived_sem_doc": []}


async def keyword_search_node(state: AnswerState):
    log.info(f"Keyword Search Invoked.")
    query = state.get('keyword_query') or state['query']
    try:
        loop = asyncio.get_event_loop()
        clean_query = query.replace("(","").replace(")","").replace("|","")
        words = [w for w in clean_query.split()]
        lenient_query = " | ".join(words)
        docs = await loop.run_in_executor(None, retrieve_similar_chunks_key,lenient_query, top_k) 
        log.info(f"Keyword Search found {len(docs)} chunks.")
        return {"retrived_key_doc": docs}
    except Exception as e:
        log.error(f"Keyword Search Error for query: {query} - {e}")
        return {"retrived_key_doc": []}

async def rerank_doc_node(state: AnswerState):
    """
    RRF Merging is mostly dictionary operations, but for consistency 
    and to ensure no micro-blocking, we convert it to async.
    """
    log.info(f"RRF Merging Invoked.")
    
    # We define the logic in a small inner function to pass to the thread pool
    def sync_rrf_logic():
        k = 15
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
        return [all_chunks[chunk_id] for chunk_id, score in sorted_ids]

    # Offload the computation
    final_chunks = await asyncio.to_thread(sync_rrf_logic)
    
    log.info(f"RRF Merging completed. Total unique chunks: {len(final_chunks)}")
    return {"reranked_docs": final_chunks[:10]}

async def get_final_context_node(state: AnswerState):
    """
    CRITICAL: This node contains the Cross-Encoder .predict() call.
    Moving this to a thread prevents it from blocking the FastAPI event loop.
    """
    log.info("Cross-Encoder Reranking Invoked (Async Path).")
    docs = state.get('reranked_docs', [])
    
    if not docs:
        log.info("No documents to rerank. Skipping.")
        return {"final_doc": []}

    query = state.get('semantic_query') or state['query']

    # Define the CPU-heavy work
    def cpu_bound_reranking():
        # 1. Deduplicate
        unique_docs = {doc[0]: doc for doc in docs}
        deduplicated_docs = list(unique_docs.values())
        docs_to_score = deduplicated_docs[:12] 
        
        # 2. Prepare pairs
        pairs = [[query, doc[2]] for doc in docs_to_score]
        
        # 3. Model Prediction (The actual bottleneck)
        scores = reranker.predict(pairs, batch_size=12, show_progress_bar=False)
        
        # 4. Construct Results
        reranked_list = []
        for i, doc in enumerate(docs_to_score):
            meta = doc[3]
            reranked_list.append({
                'id': doc[0],
                'doc_id': doc[1],
                'text': doc[2],
                'page_no': meta.get('page_number', ""),
                'chunk_type': "Has Table Data" if meta.get('has_tables') else "No table Data", 
                'rerank_score': float(scores[i])
            })
        
        reranked_list.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top 3. Note: ms-marco-MiniLM scores for banking/policy docs can legitimately
        # range from -10 to +8. Withdrawal queries regularly score -8 to -9 even when
        # the retrieved docs are correct. Hallucination is prevented by prompts, not here.
        top3 = reranked_list[:3]
        log.info(
            f"Cross-encoder top scores: "
            + ", ".join(f"{d['rerank_score']:.3f}" for d in top3)
        )
        return top3

    # Await the CPU work in a separate thread
    top_docs = await asyncio.to_thread(cpu_bound_reranking)
    
    log.info(f"Cross-Encoder completed via ThreadPool. Top score: {top_docs[0]['rerank_score'] if top_docs else 'N/A'}")
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

def get_recent_history(messages: list, k: int = 6):
    """Returns the last k messages from the history."""
    if not messages:
        return []
    return messages[-k:]

def format_history_for_refiner(messages: list) -> str:
    """
    Converts LangChain HumanMessage/AIMessage objects into clean readable text
    that the refiner LLM can parse. Without this, the model receives raw
    Python repr like [HumanMessage(content='...')] which it struggles to use.
    """
    if not messages:
        return "No prior conversation."
    lines = []
    for msg in messages:
        msg_type = getattr(msg, 'type', None)
        content = getattr(msg, 'content', str(msg))
        if msg_type == 'human':
            lines.append(f"User: {content}")
        elif msg_type == 'ai':
            lines.append(f"Assistant: {content}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)

async def answer_agent_node(state: AnswerState):
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
        language = state.get('language', 'English')

        full_history = state.get('messages', [])
        recent_history = get_recent_history(full_history, k=6)
        prompt = ChatPromptTemplate.from_messages([
            ("system", answering_prompt),
            *recent_history,
            ("human", """
            ### CONTEXT:\n{clean_entry}\n\n
            ### USER QUESTION:\n{user_query}\n\n
            ### LANGUAGE INSTRUCTION:\n
            1. The user's query was in **{language}**. 
            2. You MUST write the 'final_answer' entirely in **{language}**.
            3. Even though the CONTEXT is in English, do not use English sentences. 
            4. Translate technical banking terms to the common **{language}** equivalent used by Fino Bank customers.
            """)
        ])

        final_chunk = limit_context_by_tokens(clean_entry, prompt, user_query, language)
        final_prompt = prompt.invoke({"clean_entry": final_chunk, "user_query": user_query, "language": language})
        
        raw_response = await ans_llm.ainvoke(final_prompt)
        answer = raw_response.content.strip()
        
        # Internal retry: Sarvam occasionally returns empty/very short on complex follow-ups.
        # One retry is cheaper than showing the user a "trouble formatting" dead-end.
        if not answer or len(answer) < 10:
            log.warning("Answer < 10 chars on first attempt. Retrying once...")
            raw_response = await ans_llm.ainvoke(final_prompt)
            answer = raw_response.content.strip()
        
        if not answer or len(answer) < 10:
            log.error("Answer still empty after retry. Returning FALLBACK.")
            return {
                "answer": "No information found regarding this query in our current records.",
                "skip_cache": True
            }

        log.info(f"Answer Generation Successful. Length: {len(answer)}")
        return {"answer": answer, 
                "skip_cache": False, 
                "attempt_count": 0,
                "messages": [HumanMessage(content=user_query), AIMessage(content=answer)]}

    except Exception as e:
        log.error(f"Answer Generation Error: {e}")
        return {"answer": "I'm sorry, I ran into an error while drafting your answer.", "skip_cache":True}

def route_after_retrieval(state: AnswerState):
    final_doc = state.get("final_doc", [])
    doc_count = len(final_doc)
    attempt = state.get("attempt_count", 0)
    
    if attempt >= 3:
        log.error(f"Routing Decision: TERMINATE - No relevant documents after {attempt} attempts for query: {state.get('query')}")
        return "fail"
    
    if doc_count > 0:
        top_score = final_doc[0].get('rerank_score', -999)
        log.info(f"Routing Decision: PROCEED - Found {doc_count} documents. Top rerank score: {top_score:.3f}")
        return "generate"
    
    # doc_count == 0 means quality gate filtered everything out
    log.warning(f"Routing Decision: RETRY - Attempt {attempt} returned no qualifying documents.")
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
        "no information found regarding this query",
        "No information found regarding this query in our current records.",
        "I'm sorry, but I couldn't find any information regarding this query in our current records.",
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

def create_graph(checkpointer):
    """Build and compile the LangGraph RAG pipeline.
    The checkpointer is passed in from the lifespan context (after the
    event loop is running) to avoid the module-level AsyncConnectionPool bug.
    """
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

    graph = builder.compile(checkpointer=checkpointer)
    # img_path = "langgraph_diagram.png"
    # graph.get_graph().draw_mermaid_png(output_file_path=img_path, draw_method=MermaidDrawMethod.API)

    return graph