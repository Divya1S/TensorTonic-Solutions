def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if not y_true:
        return 0.0

    #Count how many predictions exactly match the true labels 
    correct_predictions = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) 

    #Micro F1 for single-label multi-class is equivalent to accuracy 
    return float(correct_predictions / len(y_true))
