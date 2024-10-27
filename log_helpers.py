import pandas as pd
import numpy as np
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
        X_df (pd.DataFrame): DataFrame of the feature data (X) with columns as features.
        y_df (pd.DataFrame or None): DataFrame of the target labels (y) if provided, otherwise None.
    """
    # Load the X (features) data from the CSV file
    X_df = pd.read_csv(X_path)

    # If y_path is provided, load the y (target) data
    if y_path is not None:
        y_df = pd.read_csv(y_path)
        # Optional: Merge both DataFrames on 'Id' column to ensure correct alignment
        data = pd.merge(X_df, y_df, on='Id')
        
        # Select only a fraction of the data if needed
        if 0 < frac < 1:
            data = data.sample(frac=frac, random_state=42)  # Use random_state for reproducibility
        
        # Separate X (features) and y (labels) after merging
        y_df = data[['_MICHD']]
        X_df = data.drop(columns=['_MICHD'])
        
        # Convert -1, 1 targets to 0, 1 targets using .loc to avoid SettingWithCopyWarning
        y_df.loc[:, '_MICHD'] = y_df['_MICHD'].replace({-1: 0, 1: 1})

        return X_df, y_df
    
    # If y_path is None (for test data), return only X_df and None for y_df
    else:
        # Select only a fraction of the data if needed
        if 0 < frac < 1:
            X_df = X_df.sample(frac=frac, random_state=42)  # Use random_state for reproducibility
        
        return X_df, None

def clean_and_standardize(X, y=None):
    """
    Standardizes the feature matrix X, replaces NaN values with column means,
    and adds a bias term (column of ones). Converts X and optionally y to NumPy arrays.

    Args:
        X (pd.DataFrame): The feature matrix as a pandas DataFrame.
        y (pd.Series or pd.DataFrame, optional): The target variable as a pandas DataFrame or Series.
            If None, only X will be processed (for test data).

    Returns:
        tx (np.ndarray): The cleaned and standardized feature matrix with a bias term.
        y (np.ndarray or None): The target variable reshaped as a column vector, or None if not provided.

    Example:
        tx, y = clean_and_standardize(X, y)
        tx, _ = clean_and_standardize(X)  # For test data with no labels
    """
    
    # Step 1: Standardize the feature matrix X
    X, mean_x, std_x = standardize(X)  # Standardize X column-wise
    
    # Step 2: Replace NaN values with column means
    xtest = np.copy(X)  # Create a copy of the standardized matrix for modification
    
    # Identify where NaN values are
    nan_mask = np.isnan(xtest)
    
    # Compute the column means, ignoring NaN values
    col_means = np.nanmean(xtest, axis=0)
    
    # Replace NaN column means with 0 if the entire column is NaN
    col_means = np.where(np.isnan(col_means), 0, col_means)
    
    # Replace NaN values in the original array with the computed column means
    xtest[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Step 3: Add a column of ones to include the bias term
    tx = np.c_[np.ones((X.shape[0], 1)), xtest]
    
    # If y is provided, process it; otherwise, return None for y
    if y is not None:
        # Convert y to a NumPy array and reshape it into a column vector
        y = y.to_numpy().reshape(-1, 1)
        return tx, y
    else:
        return tx, None

def standardize(x):
    """Standardize the original data set column-wise."""
    mean_x = np.mean(x, axis=0)  # Compute the mean for each feature (column)
    std_x = np.std(x, axis=0)  # Compute the standard deviation for each feature (column)
    
    # Avoid division by zero by replacing zero std values with 1 (or a very small number)
    std_x[std_x == 0] = 1
    
    # Standardize each feature
    x_standardized = (x - mean_x) / std_x
    
    return x_standardized, mean_x, std_x


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, k_indices, k, lambda_, gamma, max_iters):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        tx:         shape=(N+1,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527595362, 0.33555914361295547)
    """

    # get k'th subgroup in test, others in train
    tx_test = tx[k_indices[k]]
    y_test = y[k_indices[k]]
    k_train = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    tx_train = tx[k_train]
    y_train = y[k_train]
    initial_w = np.zeros((tx.shape[1], 1))
    
    
    # ridge regression: 
    w, _ = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
    
    # calculate the loss for train and test data with the same function: 
    loss_tr = calculate_loss_log(y_train, tx_train, w)
    loss_te = calculate_loss_log(y_test, tx_test, w)
    return loss_tr, loss_te


def cross_validation_reg_log(y, tx, k_fold, lambdas, max_iters, gamma):
    """
    Perform cross-validation over the regularization parameter lambda for logistic regression.
    This function computes the average training and test loss for each lambda over k-folds.

    Args:
        y (np.ndarray): Target variable, shape (N, 1)
        tx (np.ndarray): Feature matrix, shape (N, D)
        k_fold (int): Number of folds for cross-validation
        lambdas (np.ndarray): Array of lambda values to test
        max_iters (int): Maximum number of iterations for gradient descent
        gamma (float): Learning rate for gradient descent

    Returns:
        lambdas (np.ndarray): The array of tested lambda values.
        losses_tr (list): List of average training losses for each lambda.
        losses_te (list): List of average test losses for each lambda.

    Example:
        >>> y = np.array([[0], [1], [1], [0]])
        >>> tx = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> k_fold = 4
        >>> lambdas = np.logspace(-4, 0, 3)
        >>> max_iters = 1000
        >>> gamma = 0.01
        >>> lambdas, losses_tr, losses_te = cross_validation_reg_log(y, tx, k_fold, lambdas, max_iters, gamma)
        >>> print(lambdas)
        [0.0001 0.01   1.    ]
        >>> print(losses_tr)
        [0.6931471805599453, 0.6845738817651234, 0.6898326590128794]
        >>> print(losses_te)
        [0.692314718559623, 0.6864934729561212, 0.690987321879532]
    """
    
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    
    # cross validation over lambdas
    for lambda_ in lambdas:
        loss_temp_train = 0
        loss_temp_test = 0
        seed = np.random.randint(0, 10000)  # Generate a random seed for each lambda
        k_indices = build_k_indices(y, k_fold, seed)
        
        # Looping over ks so that every section is test once
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, k_indices, k, lambda_, gamma, max_iters)
            loss_temp_train += loss_tr
            loss_temp_test += loss_te
            
        # Average loss over folds
        loss_temp_train /= k_fold
        loss_temp_test /= k_fold
        
        # Append result
        losses_tr.append(loss_temp_train)
        losses_te.append(loss_temp_test)
        
    # Find the best lambda (based on the lowest test loss)
    best_loss = min(losses_te)
    best_lambda = lambdas[np.argmin(losses_te)]
        
    return lambdas, losses_tr, losses_te

import os

def save_array_as_csv(array, array_name, directory_path, precision=10):
    """
    Saves a given NumPy array as a CSV file at a specified directory.
    
    The CSV file is named based on the variable name provided (array_name).
    If the directory does not exist, it is created automatically.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The NumPy array to be saved.
        
    array_name : str
        The name of the array (used to name the CSV file).
        
    directory_path : str
        The path where the CSV file will be saved. The directory will be created
        if it does not already exist.
        
    precision : int, optional
        The number of decimal places to use when saving the array. Default is 10.
    
    Returns:
    --------
    str
        The full path to the saved CSV file.
    
    Raises:
    -------
    ValueError
        If the input is not a valid NumPy array.
    
    Example:
    --------
    >>> lambdas = np.array([0.1, 0.01, 0.001])
    >>> save_array_as_csv(lambdas, 'lambdas', './results')
    './results/lambdas.csv'
    """
    
    # Ensure the input is a NumPy array
    if not isinstance(array, np.ndarray):
        raise ValueError("The provided input is not a valid NumPy array.")
    
    # Create the directory if it does not exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Construct the full file path
    file_path = os.path.join(directory_path, f'{array_name}.csv')
    
    # Save the array as CSV with specified precision
    np.savetxt(file_path, array, delimiter=",", fmt=f'%.{precision}f')
    
    # Return the full path to the saved file
    return file_path

def safe_results(w, path_X, path, name):
    """
    Computes prediction of tx with w and saves them at given path under name.
    
    Args:
        path_X: str; path to CSV with the unnormalized test data
        w: np.array; weights from optimization
        
    Returns:
        y, file_path: The array of predictions and the path to the saved file.
    """
    # Load the x data from the path
    X, _ = load_csv_data(path_X)
    
    # Assuming y is loaded with X; if not, we only standardize X
    tx, _ = clean_and_standardize(X, y=None)
    y = np.sign(tx @ w)
    y_int = y.astype(int)
    
    # Extract 'Id' column from X (assuming it's a DataFrame)
    y_Ids = X['Id'].to_numpy()
    
    # Combine y_Ids and y_int as columns for saving
    results = np.column_stack((y_Ids, y_int))
    
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Construct the full file path
    file_path = os.path.join(path, f'{name}.csv')
    
    # Save the results as CSV with specified precision for `y_int`
    np.savetxt(file_path, results, fmt=['%i', '%i'], delimiter=",", header="Id,Prediction", comments='')
    
    # Return y values and the full path to the saved file
    return y, file_path
