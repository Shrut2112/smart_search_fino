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