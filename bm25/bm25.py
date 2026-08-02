import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    if not docs:
        return np.array([], dtype=float)
        
    N = len(docs)
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lengths)
    
    # Avoid division by zero in case all documents are completely empty
    if avgdl == 0:
        avgdl = 1.0
        
    scores = np.zeros(N, dtype=float)
    
    # Use a Counter to handle duplicate terms in the query efficiently
    query_counts = Counter(query_tokens)
    
    for term, q_count in query_counts.items():
        # Get term frequency for this specific term across all docs
        # list.count() is implemented in C and is extremely fast for this
        tfs = np.array([doc.count(term) for doc in docs], dtype=float)
        
        # Document frequency: number of docs where the term appears at least once
        df = np.sum(tfs > 0)
        
        # If the term is not in the corpus, it contributes 0 to the score
        if df == 0:
            continue
            
        # IDF Calculation
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        # BM25 Term Calculation
        numerator = tfs * (k1 + 1)
        denominator = tfs + k1 * (1.0 - b + b * (doc_lengths / avgdl))
        
        # Accumulate score for this term across all documents
        # We multiply by q_count in case the query token was repeated (e.g. ["machine", "learning", "machine"])
        scores += q_count * idf * (numerator / denominator)
        
    return scores
        