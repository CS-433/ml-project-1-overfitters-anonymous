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
    gradient = (-1/len(y))*(tx.T).dot(e)
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
    >>> w, loss = learning_by_newton_method(y, tx, w, gamma)
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
    >>> w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    
    loss, gradient = penalized_logistic_regression(y.reshape(-1, 1), tx, w, lambda_)
    w -= gamma * gradient
    
    return w, loss


#### Main Functions ##################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma,tol=1e-12, verbose=False):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        tol: gives tolerance
        verbose: if true, prints useful information for debugging
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    for n in range(max_iters+1):
        grad, e = compute_grad(y, tx, w)
        loss = compute_MSE(e)
        
        # GD step
        w = w - gamma*grad

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, verbose=False) :
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        verbose: if true, prints useful information for debugging
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w

    for n in range(max_iters):

        index = np.random.randint(0, len(y)) # pick an index at random ( batch of size 1 )
        grad = compute_stoch_grad(y[index], tx[index], w) # compute the gradient using that index
        w = w - gamma * grad # updating w
        loss = compute_loss(y, tx, w) # the error is computed with the entire dataset, not just the one we picked when computing gradient

        if verbose:
            print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


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
        initial_w:    shape=(D+1, )
        max_iter:     int
        gamma:        float
        
    Returns: 
        w:         shape=(D+1, )
        loss:      loss at final w
        
    Example:
    w_final, losses = logistic_regression(y, tx, initial_w, max_iter, gamma)
    
    
    """
    losses = []
    w = initial_w.reshape(-1, 1)

    # start the logistic regression
    for iter in range(max_iter+1):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w, gamma)
        losses.append(loss)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    print("loss={l}".format(l=compute_loss(y, tx, w)))
    
    return w.ravel(), loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Args:
        y:            shape=(N, 1)
        tx:           shape=(N, D+1)
        initial_w:    shape=(D+1, )
        max_iters:     int
        gamma:        float
        
    Returns: 
        w:         shape=(D+1, )
        loss:      loss at final w
    """
    
    threshold = 1e-9
    losses = []

    # build tx
    w = initial_w.reshape(-1, 1)

    # start the logistic regression
    for iter in range(max_iters+1):
        # get loss and update w.
        w, loss = log_learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    #print("Lambda={lam}, Training loss={l}".format(lam=lambda_, l=loss))
    return w.ravel(), loss


# Additional stuff

def reg_logistic_regression_cheat(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-9
    losses = []

    # build tx
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters+1):
        # get loss and update w.
        w, loss = log_learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    print("Lambda={lam}, Training loss={l}".format(lam=lambda_, l=loss))
    return w, loss


def logistic_regression_cheat(y, tx, initial_w, max_iter, gamma, threshold = 1e-8):
    """
    Do logistic regression until the threshold or max_iter is reached.
    
    Args:
        y:            shape=(N, 1)
        tx:           shape=(N, D+1)
        initial_w:    shape=(D+1, )
        max_iter:     int
        gamma:        float
        
    Returns: 
        w:         shape=(D+1, )
        loss:      loss at final w
        
    Example:
    w_final, losses = logistic_regression(y, tx, initial_w, max_iter, gamma)
    
    
    """
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter+1):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w, gamma)
        losses.append(loss)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    print("loss={l}".format(l=compute_loss(y, tx, w)))
    
    return w, loss


