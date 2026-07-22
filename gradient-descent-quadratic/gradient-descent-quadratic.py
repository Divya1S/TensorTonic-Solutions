def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0

    for _ in range(steps):
        #Calculate the gradient (derivative) at current x
        gradient = 2 * a * x + b

        #Update x by moving in the opposite direction of the gradient 
        x = x  - lr * gradient

    return x