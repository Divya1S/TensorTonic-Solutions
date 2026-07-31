import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    n_docs = len(documents)
    
    # Step 1: Tokenize documents (convert to lowercase and split by whitespace)
    doc_tokens = [doc.lower().split() for doc in documents]
    
    # Step 2: Build sorted vocabulary
    unique_terms = set(term for tokens in doc_tokens for term in tokens)
    vocabulary = sorted(list(unique_terms))
    n_vocab = len(vocabulary)
    
    # Handle empty vocabulary edge case
    if n_vocab == 0:
        return np.zeros((n_docs, 0), dtype=np.float64), vocabulary
        
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    
    # Step 3: Compute Document Frequency (df)
    df = np.zeros(n_vocab, dtype=np.float64)
    for tokens in doc_tokens:
        unique_in_doc = set(tokens)
        for term in unique_in_doc:
            df[term_to_idx[term]] += 1.0
            
    # Step 4: Compute IDF array: log(N / df(t))
    idf = np.log(n_docs / df)
    
    # Step 5: Compute TF and construct TF-IDF matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab), dtype=np.float64)
    
    for doc_idx, tokens in enumerate(doc_tokens):
        doc_len = len(tokens)
        if doc_len == 0:
            continue  # Empty documents remain all zeros
            
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            col_idx = term_to_idx[term]
            tf = count / doc_len
            tfidf_matrix[doc_idx, col_idx] = tf * idf[col_idx]
            
    return tfidf_matrix, vocabulary