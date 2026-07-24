def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    #Slice the recommended list to get only the top-k items 
    top_k = recommended[:k]

    #Convert relevant items to a set for O(1) lookups 
    relevant_set = set(relevant)

    #Count relevant items to a set for O(1) lookups 
    hits = sum(1 for item in top_k if item in relevant_set)

    #Calculate precision and recall
    precision = hits / k
    recall = hits / len(relevant)

    return [float(precision), float(recall)]

    

    

    