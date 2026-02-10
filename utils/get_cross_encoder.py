from sentence_transformers import CrossEncoder

def get_crossencoder():
    reranker = CrossEncoder('D:\hf_cache\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8')
    return reranker