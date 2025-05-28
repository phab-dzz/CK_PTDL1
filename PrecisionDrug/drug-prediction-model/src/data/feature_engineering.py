from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd

def select_features(X, y, k=10):
    """
    Select the top k features based on univariate statistical tests.
    
    Parameters:
    X (pd.DataFrame): Feature set.
    y (pd.Series): Target variable.
    k (int): Number of top features to select.
    
    Returns:
    pd.DataFrame: DataFrame containing the selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    return X.loc[:, mask]

def scale_features(X):
    """
    Scale features using StandardScaler.
    
    Parameters:
    X (pd.DataFrame): Feature set.
    
    Returns:
    pd.DataFrame: Scaled feature set.
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)