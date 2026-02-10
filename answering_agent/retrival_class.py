from answering_agent.db_ret import retrieve_similar_chunks, retrieve_similar_chunks_key
from utils.schema import Answer,RefinedQuery
import tiktoken
import json
from utils.prompts.retrival_prompts import refine_query_prompt, answering_prompt
from langchain_core.prompts import ChatPromptTemplate

class RetrievalPipeline:
    def __init__(self, embedding_model, llm,ans_llm, reranker, top_k=10):
        self.embedding_model = embedding_model
        self.llm = llm
        self.ans_llm = ans_llm
        self.reranker = reranker
        self.top_k = top_k
        self.encoding = tiktoken.get_encoding("cl100k_base")
    def query_refiner_agent(self,user_query: str) -> RefinedQuery:
        """
        Refines the user query for better Vector and Keyword search performance.
        """
        prompt = refine_query_prompt.substitute(
            user_query=user_query
        )
        
        refined_query = self.llm.with_structured_output(RefinedQuery).invoke(prompt)
        return refined_query

    def semantic_search_retrieve(self, query):
        query_embedding = self.embedding_model.embed_query(query)
        return retrieve_similar_chunks(query_embedding, self.top_k)
    
    def keyword_search_retrival(self, query):
        clean_query = query.replace("(","").replace(")","").replace("|","")
    
        words = [w for w in clean_query.split()]
        lenient_query = " | ".join(words)
        return retrieve_similar_chunks_key(lenient_query, self.top_k) 
    
    def rerank_doc(self, semantic_doc, keyword_doc):
        """
        Reciprocal Rank Fusion (RRF) to merge results from two different search methods.
        """
        k = 60
        rrf_scores = {} 
        
        for rank, chunk in enumerate(semantic_doc):
            chunk_id = chunk[0] 
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            
        for rank, chunk in enumerate(keyword_doc):
            chunk_id = chunk[0]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        
        all_chunks = {c[0]: c for c in semantic_doc + keyword_doc}
        
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [all_chunks[chunk_id] for chunk_id, score in sorted_ids]
     
    def get_final_context(self,query,docs):
        if not docs:
            return []

        unique_docs = {}
        for doc in docs:
            if doc[0] not in unique_docs:
                unique_docs[doc[0]] = doc
        
        deduplicated_docs = list(unique_docs.values())

        pairs = [[query, doc[2]] for doc in deduplicated_docs]
        
        scores = self.reranker.predict(pairs)
        
        reranked_list = []
        for i, doc in enumerate(deduplicated_docs):
            reranked_list.append({
                'id': doc[0],
                'doc_id': doc[1],  # Useful for metadata tracking
                'text': doc[2],
                'page_no': doc[3].get('page_number',""),
                'chunk_type': ["Has Table Data" if doc[3].get('has_tables',"") else "No table Data"], 
                'rerank_score': float(scores[i]) # Convert to float for JSON compatibility
            })
        
        reranked_list.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # 6. Return top 5
        return reranked_list[:5]
    
    def answer_agent(self,context_text,user_query):
        try:
            clean_entry = []
            for c in context_text:
                src = c.get('doc_id',"")
                text = c.get('text',"")
                pageno = c.get('page_no',"")
                chunk_type = c.get('chunk_type',"")
                scoren = c.get('rerank_score',"")
                if scoren > 0:
                    print(f"{scoren}\n")
                    entry = f"\n# Source: {src} PAGE NO: {pageno} TABLE CONTENT: {chunk_type} TEXT: {text} RERANK SCORE: {scoren}\n"
                    clean_entry.append(entry)
                    
            if not clean_entry:
                return "I couldn't find any specific information in our documents to answer that question."
            
            prompt = ChatPromptTemplate.from_messages(
                [("system",answering_prompt),
                ("human","### CONTEXT:\n{clean_entry}\n ### USER QUESTION:\n{user_query}")]
            )

            final_chunk = self.limit_context_by_tokens(clean_entry,prompt,user_query)
            final_prompt = prompt.invoke({"clean_entry":final_chunk,"user_query":user_query})
           
            raw_response = self.ans_llm.bind(response_format={"type":"json_object"}).invoke(final_prompt)
            
            try:
                parsed_json = json.loads(raw_response.content)
                return parsed_json.get("final_answer")
            
            except json.JSONDecodeError:
                return raw_response.content if raw_response.content else "The assistant provided an invalid response format."
        
        except Exception as e:
            return "I'm sorry, I ran into an error while drafting your answer."
            
    def limit_context_by_tokens(self,chunks,prompt,query,max_limit=6000):
        
        static_text = prompt.format(clean_entry="",user_query=query)
        static_tokens = len(self.encoding.encode(static_text))
        available_tokens = max_limit - static_tokens - 1000
        
        final_chunks = []
        current_token = 0
        for c in chunks:
            chunk_token_count = len(self.encoding.encode(c))
            
            if current_token + chunk_token_count > available_tokens:
                print(f"Token limit reached. Skipping remaining {len(chunks) - len(final_chunks)} chunks.")
                break
            final_chunks.append(c)
            current_token += chunk_token_count

        return "".join(final_chunks)
            
    def generate_response(self, query):
        try:
            try:
                refined = self.query_refiner_agent(query)
                sem_q = refined.semantic_query
                key_q = refined.keyword_query
            except Exception as e:
                sem_q = key_q = query # Safe fallback
            
            try:
                retrieved_chunks_sem = self.semantic_search_retrieve(sem_q)
                retrieved_chunks_key = self.keyword_search_retrival(key_q)
            except Exception as e:
                retrieved_chunks_sem = []
                retrieved_chunks_key = []
            
            if not retrieved_chunks_sem and not retrieved_chunks_key:
                return "I'm sorry, I couldn't access the database right now. Please try again later."

            # Merge and Rerank
            final_chunks = self.rerank_doc(retrieved_chunks_sem, retrieved_chunks_key)
            reranked_data = self.get_final_context(refined.semantic_query,final_chunks)
            
            return self.answer_agent(reranked_data, sem_q)
        except:
            return "Something went wrong while processing your request. Please try a different query."
        