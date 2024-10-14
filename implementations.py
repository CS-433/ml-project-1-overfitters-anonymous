# Contains all functions needed for project 1
import numpy as np

def compute_mse(e):
    # computes the Mean Squared Error
    return 0.5*np.mean(e**2)

def compute_mae(e):
    # computes the Mean Absolute Error
    return 0.5*np.mean(np.abs(e))

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return compute_mse(e)

def compute_grad(y, tx, w):
    # computes the gradient at w.
    e = y - tx.dot(w) # the error vector
    gradient = (-1/len(y))*tx.T.dot(e)
    return gradient, e

def compute_stoch_grad(y, tx, w):
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma,tol=1e-10, verbose=False):

    # compute the first step
    # w(t+1) (i.e. w at step t+1) is written w_next, and w(t) (i.e. w at step t) is written w. 
    w = initial_w
    grad, e = compute_grad(y, tx, w)
    loss = compute_mse(e)
    w_next = w - gamma*grad

    # make steps in the opposite direction of the gradient and stops if the parameters does not vary anymore ( i.e. we reached a minimum ), or if the maximum iteration number is reached. 
    step_count = 0 # the counter of steps
    while step_count < max_iters and abs(w - w_next) > tol : # we could also instead check the condition abs(grad) > tol since we expect grad = 0 at minimum.
        step_count += 1
        w = w_next # w(t) is the w(t+1) of the step before. 
        grad, e = compute_grad(y, tx, w)
        loss = compute_mse(e)
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



""" def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, tol=1e-10, verbose=False) :
    # with a stopping condition. However, 'stochastic noise' makes this condition never fulfield, so let's forget this.
    # first step
    w = initial_w
    index = np.random.randint(0, len(y)) # pick an index at random ( batch of size 1 )
    grad, _ = compute_stoch_grad(y[index], tx[index], w) # compute the gradient using that index
    w_next = w - gamma * grad # updating w
    loss = compute_loss(y, tx, w) # the error is computed with the entire dataset, not just the one we picked when computing gradient

    n = 0 # the step count
    while n < max_iters and np.linalg.norm(w_next - w) > tol:
        n = n + 1
        w = w_next

        index = np.random.randint(0, len(y))
        grad = compute_stoch_grad(y[index], tx[index], w)
        w_next = w - gamma * grad
        loss = compute_loss(y, tx, w)

        if verbose:
            print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w """