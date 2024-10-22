import pandas as pd
import numpy as np

def load_csv_data(X_path, y_path, frac=1):
    """
    Loads X (features) and y (labels) from two CSV files, 
    converts -1, 1 targets to 0, 1 targets, and returns only a fraction of the data.
    
    Parameters:
        X_path (str): Path to the CSV file containing the X data (features).
        y_path (str): Path to the CSV file containing the y data (labels).
        frac (float): Fraction of the data to return, between 0 and 1. Default is 1.0 (100% of the data).
    
    Returns:
        X_df (pd.DataFrame): DataFrame of the feature data (X) with columns as features.
        y_df (pd.DataFrame): DataFrame of the target labels (y) with columns as labels (converted to 0 and 1).
    """
    # Load the X (features) data from the CSV file
    X_df = pd.read_csv(X_path)
    
    # Load the y (target) data from the other CSV file
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

def clean_and_standardize(X, y):
    """
    Standardizes the feature matrix X, replaces NaN values with column means,
    and adds a bias term (column of ones). Converts both X and y to NumPy arrays.

    Args:
        X (pd.DataFrame): The feature matrix as a pandas DataFrame.
        y (pd.Series or pd.DataFrame): The target variable as a pandas DataFrame or Series.

    Returns:
        tx (np.ndarray): The cleaned and standardized feature matrix with a bias term.
        y (np.ndarray): The target variable reshaped as a column vector.

    Example:
        tx, y = clean_and_standardize(X, y)
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
    
    # Step 4: Convert y to a NumPy array and reshape it into a column vector
    y = y.to_numpy().reshape(-1, 1)
    
    return tx, y

def standardize(x):
    """Standardize the original data set column-wise."""
    mean_x = np.mean(x, axis=0)  # Compute the mean for each feature (column)
    std_x = np.std(x, axis=0)  # Compute the standard deviation for each feature (column)
    
    # Avoid division by zero by replacing zero std values with 1 (or a very small number)
    std_x[std_x == 0] = 1
    
    # Standardize each feature
    x_standardized = (x - mean_x) / std_x
    
    return x_standardized, mean_x, std_x