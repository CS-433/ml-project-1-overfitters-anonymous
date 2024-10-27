# Contains all functions needed for project 1

import numpy as np

#### Additional Functions ##########

def compute_MSE(error):
    """
    Compute the Mean Squared Error (MSE)

    Args:
       - error: numpy array of shape=(N, ), N is the number of samples.

    Returns:
       - MSE: the value of the loss (a scalar), depending on the given input error
    """
    # Computing the MSE loss
    MSE_loss = 0.5*np.mean(error**2)
    
    return MSE_loss


def compute_loss(y, tx, w):
    """
    Calculate the loss using MSE.

    Args:
       - y: numpy array of shape=(N, ), N is the number of samples.
       - tx: shape=(N,D), D is the number of features.
       - w: optimal weights, numpy array of shape(D,), D is the number of features.

    Returns:
       - loss: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # Error vector 
    error = y - tx.dot(w)

    return compute_MSE(error)


def compute_grad(y, tx, w):
    # computes the gradient at w.
    e = y - tx.dot(w) # the error vector
    gradient = (-1/len(y))*tx.T.dot(e)
    return gradient, e


def compute_stoch_grad(y, tx, w):
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e)
    return gradient


def sigmoid(t):
    t = np.clip(t, -500, 500)  # Clipping to avoid overflow
    return 1 / (1 + np.exp(-t))


def calculate_loss_log(y, tx, w):
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
    >>> round(calculate_loss_log(y, tx, w), 8)
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


def calculate_gradient_log(y, tx, w):
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


def calculate_hessian_log(y, tx, w):
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
    loss = calculate_loss_log(y, tx, w)
    gradient = calculate_gradient_log(y, tx, w)
    hessian = calculate_hessian_log(y, tx, w)
    
    # Calculate w_t+1 using Newtons method
    w -= gamma * np.linalg.inv(hessian) @ gradient
    
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """    
    
    # New vectorized loss computation
    sig_pred = sigmoid(tx @ w)
    loss = -1 / len(y) * (y.T @ np.log(sig_pred) + (1 - y).T @ np.log(1 - sig_pred)).item() + (lambda_ / 2) * (w[1:].T @ w[1:]).item()
    
    
    # Gradient with regularization (excluding bias term)
    
    reg_term = np.copy(w)
    reg_term[0] = 0  # No regularization for the bias term
    gradient = 1 / len(y) * tx.T @ (sigmoid(tx @ w) - y) + lambda_ * reg_term
    
    
    return loss, gradient


def log_learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
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
    
    loss, gradient = penalized_logistic_regression(y.reshape(-1, 1), tx, w, lambda_)
    w -= gamma * gradient
    
    return loss, w


#### Main Functions ##################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma,tol=1e-10, verbose=False):

    # compute the first step
    # w(t+1) (i.e. w at step t+1) is written w_next, and w(t) (i.e. w at step t) is written w. 
    w = initial_w
    grad, e = compute_grad(y, tx, w)
    loss = compute_MSE(e)
    w_next = w - gamma*grad

    # make steps in the opposite direction of the gradient and stops if the parameters does not vary anymore ( i.e. we reached a minimum ), or if the maximum iteration number is reached. 
    step_count = 0 # the counter of steps
    while step_count < max_iters and np.linalg.norm(w - w_next) > tol : # we could also instead check the condition abs(grad) > tol since we expect grad = 0 at minimum.
        step_count += 1
        w = w_next # w(t) is the w(t+1) of the step before. 
        grad, e = compute_grad(y, tx, w)
        loss = compute_MSE(e)
        w_next = w - gamma*grad
        if verbose:
            print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=step_count, ti=max_iters - 1, l=loss, w0=w_next[0], w1=w_next[1]))
    return loss, w_next


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, verbose=False) :

    w = initial_w

    for n in range(max_iters):

        index = np.random.randint(0, len(y)) # pick an index at random ( batch of size 1 )
        grad = compute_stoch_grad(y[index], tx[index], w) # compute the gradient using that index
        w = w - gamma * grad # updating w
        loss = compute_loss(y, tx, w) # the error is computed with the entire dataset, not just the one we picked when computing gradient

        if verbose:
            print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w


def least_squares(y, tx):
    """
    Calculate the least squares solution.
    returns optimal weights and mse.

    Args:
       - y: numpy array of shape (N,), N is the number of samples.
       - tx: numpy array of shape (N,D), D is the number of features.

    Returns:
       - w: optimal weights, numpy array of shape(D,), D is the number of features.
       - loss: MSE (scalar)
    """
    # constructing & solving the system Aw=b
    A = tx.T.dot(tx)
    b = tx.T.dot(y)    
    w = np.linalg.solve(A, b)
    
    # computing the MSE (loss)
    error = y - tx.dot(w)
    loss = compute_MSE(error)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression.

    Args:
       - y: numpy array of shape (N,), N is the number of samples.
       - tx: numpy array of shape (N,D), D is the number of features.
       - lambda_: scalar.

    Returns:
       - w: optimal weights, numpy array of shape(D,), D is the number of features.
       - loss: MSE (scalar)
    """
    # Constructing & and solving the linear system
    m = (tx.T.dot(tx)).shape[0] # Matrix size
    A = tx.T.dot(tx)+2*m*lambda_*np.eye(m)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)

    # computing the MSE (loss)
    error = y - tx.dot(w)
    loss = compute_MSE(error)

    return w, loss


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
        loss:      loss at final w
        
    Example:
    w_final, losses = logistic_regression(y, tx, initial_w, max_iter, gamma)
    
    
    """
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w, gamma)
        losses.append(loss)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    print("loss={l}".format(l=compute_loss(y, tx, w)))
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []

    # build tx
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = log_learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    print("Lambda={lam}, Training loss={l}".format(lam=lambda_, l=loss))
    return w, loss



