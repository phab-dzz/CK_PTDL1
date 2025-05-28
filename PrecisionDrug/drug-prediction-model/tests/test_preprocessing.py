import pytest
from src.data.preprocessing import handle_missing_values, encode_categorical_variables, scale_features
import pandas as pd

def test_handle_missing_values():
    data = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 'cat', 'dog', 'cat']
    })
    processed_data = handle_missing_values(data)
    assert processed_data['A'].isnull().sum() == 0
    assert processed_data['B'].isnull().sum() == 0

def test_encode_categorical_variables():
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['cat', 'dog', 'cat']
    })
    processed_data = encode_categorical_variables(data, ['B'])
    assert 'B_cat' in processed_data.columns
    assert 'B_dog' in processed_data.columns
    assert processed_data['B_cat'].sum() == 2
    assert processed_data['B_dog'].sum() == 1

def test_scale_features():
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    scaled_data = scale_features(data)
    assert scaled_data['A'].mean() == pytest.approx(0, rel=1e-2)
    assert scaled_data['B'].mean() == pytest.approx(0, rel=1e-2)