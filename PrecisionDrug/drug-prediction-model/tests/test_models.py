import pytest
from src.models.decision_tree import BestDTC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

@pytest.fixture
def setup_data():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_train_model(setup_data):
    X_train, _, y_train, _ = setup_data
    model = BestDTC(max_depth=5, criterion='gini', max_leaf_nodes=6)
    model.train(X_train, y_train)
    assert model is not None

def test_predict_model(setup_data):
    X_train, X_test, y_train, _ = setup_data
    model = BestDTC(max_depth=5, criterion='gini', max_leaf_nodes=6)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)

def test_save_load_model(setup_data):
    X_train, _, y_train, _ = setup_data
    model = BestDTC(max_depth=5, criterion='gini', max_leaf_nodes=6)
    model.train(X_train, y_train)
    model.save('test_model.pkl')
    loaded_model = BestDTC.load('test_model.pkl')
    assert loaded_model is not None
    assert loaded_model.predict(X_train).shape == model.predict(X_train).shape

def test_model_evaluation(setup_data):
    X_train, X_test, y_train, y_test = setup_data
    model = BestDTC(max_depth=5, criterion='gini', max_leaf_nodes=6)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    
    from src.models.model_evaluation import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
    
    accuracy = calculate_accuracy(y_test, predictions)
    precision = calculate_precision(y_test, predictions)
    recall = calculate_recall(y_test, predictions)
    f1 = calculate_f1_score(y_test, predictions)
    
    assert accuracy >= 0
    assert precision >= 0
    assert recall >= 0
    assert f1 >= 0