import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Accuracy is a global metric and calculated exactly the same across all modes
    accuracy = np.mean(y_true == y_pred)
    
    # Micro-averaging treats the entire dataset as one aggregate calculation.
    # In single-label multiclass, every False Positive is also a False Negative for another class. 
    # Therefore, micro Precision, Recall, and F1 mathematically simplify to equal global Accuracy.
    if average == "micro":
        return {
            "accuracy": float(accuracy),
            "precision": float(accuracy),
            "recall": float(accuracy),
            "f1": float(accuracy)
        }
        
    # Binary-averaging isolates a single positive class
    elif average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
    # Macro and Weighted averaging require calculating metrics for every unique class
    elif average in ["macro", "weighted"]:
        # Find all unique classes present in either true labels or predictions
        classes = np.unique(np.concatenate((y_true, y_pred)))
        
        precisions = []
        recalls = []
        f1s = []
        supports = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f_score = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f_score)
            supports.append(np.sum(y_true == c))
            
        if average == "macro":
            # Macro averages all class metrics equally, regardless of class size
            return {
                "accuracy": float(accuracy),
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "f1": float(np.mean(f1s))
            }
            
        else: # average == "weighted"
            # Weighted averages class metrics by the true number of instances (support) in each class
            total_support = np.sum(supports)
            if total_support > 0:
                prec_w = np.average(precisions, weights=supports)
                rec_w = np.average(recalls, weights=supports)
                f1_w = np.average(f1s, weights=supports)
            else:
                prec_w = rec_w = f1_w = 0.0
                
            return {
                "accuracy": float(accuracy),
                "precision": float(prec_w),
                "recall": float(rec_w),
                "f1": float(f1_w)
            }