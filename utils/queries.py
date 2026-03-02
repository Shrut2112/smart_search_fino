sql_query = f"""
SELECT 
    chunk_id, 
    doc_id, 
    text, 
    metadata, 
    -- 1 minus distance gives us a similarity score (closer to 1.0 is better)
    (1 - (embedding <=> %s::vector))::FLOAT AS similarity_score
FROM chunks
WHERE status = 'active'
-- Order by similarity (closest vectors first)
ORDER BY embedding <=> %s::vector
-- Limit to top matches
LIMIT %s;
"""

keyword_query = f"""
  SELECT 
      chunk_id, 
      doc_id, 
      text, 
      metadata,
      -- ts_rank_cd calculates a relevancy score based on word frequency and proximity
      ts_rank_cd(body_search, to_tsquery('english', %s)) AS keyword_score
  FROM chunks
  WHERE body_search @@ to_tsquery('english', %s)
    AND status = 'active'
  ORDER BY keyword_score DESC
  LIMIT %s;
"""

hash_check_query = f"""
  UPDATE cache
  SET ask_count = ask_count + 1, last_asked = NOW()
  WHERE query_hash = %s
  RETURNING answer, created_at;
"""

semantic_cache_query = f"""
  UPDATE cache
  SET ask_count = ask_count + 1, last_asked = NOW()
  WHERE id = (
    SELECT id 
    FROM cache 
    WHERE embedding_vector <=> %s::vector < 0.10
    ORDER BY embedding_vector <=> %s::vector ASC
    LIMIT 1
  )
  RETURNING answer, created_at; 
"""
insert_cache = f"""
  INSERT INTO cache (query_hash, user_query, answer, embedding_vector, ask_count, last_asked, created_at)
    VALUES (%s, %s, %s, %s, 1, NOW(), NOW())
    ON CONFLICT (query_hash) 
    DO UPDATE SET 
        answer = EXCLUDED.answer,
        created_at = NOW(),
        ask_count = cache.ask_count,
        last_asked = NOW();
"""