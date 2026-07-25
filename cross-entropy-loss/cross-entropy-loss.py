import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    #Ensure inputs are numpy arrays 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #N is the number of samples (eg number of questions on the test)
    N = len(y_true)

    #Advanced indexing: Pluck out only the probabilities for the correct classes 
    #np.arrange(N) creates a list of row indices: [0, 1, 2,.......N-1]
    #y_true provides the column indices 
    correct_class_probs = y_pred[np.arange(N), y_true]

    #Calculate the negative log for each sample then find the average (mean)
    loss = -np.mean(np.log(correct_class_probs))

    return float(loss)