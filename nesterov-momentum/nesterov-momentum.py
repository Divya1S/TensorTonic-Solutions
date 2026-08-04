import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    w : current parameter 
    v : current velocity 
    grad: gradients at look ahead position
    lr: learning rate 
    momentum: Momentum factor 
    """

    #Ensure inputs are numpy arrays (useful if lists are passed directly)
    w = np.array(w)
    v = np.array(v)
    grad = np.array(grad)

    #Step 2: Update the velocity 
    #v <- momentum * v + lr * grad(w_look)
    new_v = momentum * v + lr * grad

    #Step 3: Update Weights 
    new_w = w - new_v

    return new_w, new_v
    
    