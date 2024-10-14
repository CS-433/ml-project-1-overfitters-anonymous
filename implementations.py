# Contains all functions needed for project 1

def least_squares(y, tx):
    return "okok"








# Max area ##########################################################
import numpy as np


# General Functions 

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    
    return np.e**t/(1+np.e**t)


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    if y.shape[0] != tx.shape[0]:
        raise ValueError("Mismatch: The number of samples in y and tx must be the same.")
    if tx.shape[1] != w.shape[0]:
        raise ValueError("Mismatch: The number of features in tx must match the number of weights in w.")
    
    # Calculate log-loss L        
    sig_pred = sigmoid(tx @ w)
    loss = -1 / len(y) * (y.T @ np.log(sig_pred) + (1 - y).T @ np.log(1 - sig_pred)).item()
    
    return loss








# Normal Logistic Regression #######################

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    
    return 1 / len(y) * tx.T @ (sigmoid(tx @ w) - y)





def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y, tx, w)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """
    # Creating S matrix with sigma values in diagonale
    sigmas = []
    for xi in tx:
        sig_xi = sigmoid(xi.T @ w)
        temp = sig_xi * (1 - sig_xi)
        sigmas.append(temp.item())
    
    # Matrix with sig (1- sig) in diagonale
    S = np.diag(sigmas)
    
    return 1 / len(y) * tx.T @ S @ tx




def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of Newton's method.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 0., 1., 1.]]
    >>> np.random.seed(0)
    >>> tx = np.random.rand(4, 3)
    >>> w = np.array([[0.1], [0.5], [0.5]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_newton_method(y, tx, w, gamma)
    >>> round(loss, 8)
    0.71692036
    >>> w
    array([[-1.31876014],
           [ 1.0590277 ],
           [ 0.80091466]])
    """
    
    # Calculating loss, gradient and hessian
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    
    # Calculate w_t+1 using Newtons method
    w -= gamma * np.linalg.inv(hessian) @ gradient
    
    return loss, w





def logistic_regression(y, tx, initial_w, max_iter, gamma, threshold = 1e-8):
    """
    Do logistic regression until the threshold or max_iter is reached.
    
    Args:
        y:            shape=(N, 1)
        tx:           shape=(N, D+1)
        initial_w:    shape=(D+1, 1)
        max_iter:     int
        gamma:        float
        
    Returns: 
        w:         shape=(D+1, 1)
        losses:    array with losses throughout iteration
        
    Example:
    w_final, losses = logistic_regression(y, tx, initial_w, max_iter, gamma)
    
    
    """
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    
    return w, losses




# Regularized Logistic Regression ####################

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient and hessian matrix at given w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    
    # Calculate log-loss L
    
    # Loss with regularization (excluding bias term)
    #loss =  - 1 / len(y) * L.item() + (lambda_ / 2) * (w[1:].T @ w[1:]).item()
    
    
    # New vectorized loss computation
    sig_pred = sigmoid(tx @ w)
    loss = -1 / len(y) * (y.T @ np.log(sig_pred) + (1 - y).T @ np.log(1 - sig_pred)).item() + (lambda_ / 2) * (w[1:].T @ w[1:]).item()
    
    
    # Gradient with regularization (excluding bias term)
    
    reg_term = np.copy(w)
    reg_term[0] = 0  # No regularization for the bias term
    gradient = 1 / len(y) * tx.T @ (sigmoid(tx @ w) - y) + lambda_ * reg_term
    
    
    #Calculating the hessian
    #Creating S matrix with sigma values in diagonale
    sigmas = []
    for xi in tx:
        sig_xi = sigmoid(xi.T @ w)
        temp = sig_xi * (1 - sig_xi)
        sigmas.append(temp.item())
    
    # Matrix with sig (1- sig) in diagonale
    S = np.diag(sigmas)
    
    hess_regul = np.identity(len(tx[0])) # Regularisation term for hessian matrix
    hess_regul[0, 0] = 0 # Not regularizing the w0
    hessian =  1 / len(y) * tx.T @ S @ tx + hess_regul
    
    
    return loss, gradient, hessian
    

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_, hessian_w = True):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    
    if hessian_w == True:
        w -= gamma * np.linalg.inv(hessian) @ gradient
    else:
        w -= gamma * gradient
    
    return loss, w
    
    

# END Max area #######################################################