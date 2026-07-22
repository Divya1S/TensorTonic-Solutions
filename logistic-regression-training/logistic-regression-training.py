import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, D = X.shape

    #Initialize weights and bias 
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        #Compute predictions
        p = _sigmoid(X @ w + b)

        #Compute gradients 
        dw = (X.T @ (p-y)) / N
        db = np.mean(p-y)

        #Update parameters 
        w -= lr * dw
        b -= lr * db 
    return w, b

        