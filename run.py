import numpy as np
import os
from implementations import *

def load_csv_data(X_path, y_path=None, frac=1):
    """
    Loads X (features) and optionally y (labels) from CSV files.
    Converts -1, 1 targets to 0, 1 targets if y is provided, and returns only a fraction of the data.
    
    Parameters:
        X_path (str): Path to the CSV file containing the X data (features).
        y_path (str, optional): Path to the CSV file containing the y data (labels). Default is None for test data.
        frac (float): Fraction of the data to return, between 0 and 1. Default is 1.0 (100% of the data).
    
    Returns:
        X (np.ndarray): Array of the feature data (X) with columns as features.
        y (np.ndarray or None): Array of the target labels (y) if provided, otherwise None.
    """
    X = np.genfromtxt(X_path, delimiter=',', skip_header=1)
    if y_path:
        y = np.genfromtxt(y_path, delimiter=',', skip_header=1)
        data = np.hstack((X, y[:, 1:]))  # Assuming y's ID is in the first column
        if 0 < frac < 1:
            np.random.seed(42)
            np.random.shuffle(data)
            data = data[:int(frac * data.shape[0])]
        
        y = data[:, -1]
        X = data[:, :-1]
        y = np.where(y == -1, 0, y)
        return X, y.reshape(-1, 1)
    else:
        if 0 < frac < 1:
            np.random.seed(42)
            np.random.shuffle(X)
            X = X[:int(frac * X.shape[0])]
        return X, None

def clean_and_standardize(X, y=None):
    """
    Standardizes the feature matrix X, replaces NaN values with column means,
    and adds a bias term (column of ones).
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray, optional): Target variable. If None, only X is processed.

    Returns:
        tx (np.ndarray): Cleaned and standardized feature matrix with a bias term.
        y (np.ndarray or None): Target variable reshaped as a column vector, or None if not provided.
    """
    X = X[:, 1:]  # Remove the first column (ID column)
    X, mean_x, std_x = standardize(X)
    
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    X[np.isnan(X)] = np.take(col_means, np.where(np.isnan(X))[1])

    tx = np.c_[np.ones((X.shape[0], 1)), X]
    
    return (tx, y.reshape(-1, 1)) if y is not None else (tx, None)

def standardize(x):
    """Standardize the original data set column-wise."""
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    std_x[std_x == 0] = 1
    x_standardized = (x - mean_x) / std_x
    return x_standardized, mean_x, std_x

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, lambda_, gamma, max_iters):
    """Return the loss of ridge regression for a fold corresponding to k_indices."""
    tx_test = tx[k_indices[k]]
    y_test = y[k_indices[k]]
    k_train = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    tx_train = tx[k_train]
    y_train = y[k_train]
    initial_w = np.zeros((tx.shape[1], 1))

    w, _ = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
    loss_tr = calculate_loss_log(y_train, tx_train, w)
    loss_te = calculate_loss_log(y_test, tx_test, w)
    return loss_tr, loss_te

def cross_validation_reg_log(y, tx, k_fold, lambdas, max_iters, gamma):
    """Perform cross-validation over the regularization parameter lambda for logistic regression."""
    losses_tr = []
    losses_te = []
    
    for lambda_ in lambdas:
        loss_temp_train = 0
        loss_temp_test = 0
        seed = 0  #np.random.randint(0, 10000)
        k_indices = build_k_indices(y, k_fold, seed)
        
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, lambda_, gamma, max_iters)
            loss_temp_train += loss_tr
            loss_temp_test += loss_te
        
        losses_tr.append(loss_temp_train / k_fold)
        losses_te.append(loss_temp_test / k_fold)
        
    return lambdas, losses_tr, losses_te

def save_array_as_csv(array, array_name, directory_path, precision=10):
    """Saves a given NumPy array as a CSV file at a specified directory."""
    if not isinstance(array, np.ndarray):
        raise ValueError("The provided input is not a valid NumPy array.")
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    file_path = os.path.join(directory_path, f'{array_name}.csv')
    np.savetxt(file_path, array, delimiter=",", fmt=f'%.{precision}f')
    return file_path

def safe_results(w, path_X, path, name):
    """Computes prediction of tx with w and saves them at given path under name."""
    X, _ = load_csv_data(path_X)
    tx, _ = clean_and_standardize(X, y=None)
    y = np.sign(tx @ w).astype(int)
    y_Ids = X[:, 0]
    results = np.column_stack((y_Ids, y))
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    file_path = os.path.join(path, f'{name}.csv')
    np.savetxt(file_path, results, fmt='%i', delimiter=",", header="Id,Prediction", comments='')
    return y, file_path


# Setting up training data

# Paths to X and y data
X_path = 'data\\x_train.csv'
y_path = 'data\\y_train.csv'

# Load the data
X, y = load_csv_data(X_path, y_path, frac=1)
# Clean and standardise the data
tx, y = clean_and_standardize(X, y)


# Setting the inital parameters, lambda was the best one from cross validation
lambda_ = 1.4677992676220705e-06
initial_w = np.zeros((tx.shape[1], 1))
max_iters = 10000  # Number of iterations
gamma = 1  # Learning rate

# Doing regularized logistic regression
final_w, final_loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

# Safe results
path = '.\\final_results'
name = 'resuts'
X_path = 'data\\x_test.csv'
y, _ = safe_results(final_w, X_path, path, name)