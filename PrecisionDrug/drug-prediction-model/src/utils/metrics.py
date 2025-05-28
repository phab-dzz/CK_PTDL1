def accuracy_score(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def precision_score(y_true, y_pred, average='binary'):
    if average == 'binary':
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    else:
        raise NotImplementedError("Only binary precision is implemented.")

def recall_score(y_true, y_pred, average='binary'):
    if average == 'binary':
        tp = sum((y_true == 1) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        raise NotImplementedError("Only binary recall is implemented.")

def f1_score(y_true, y_pred, average='binary'):
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0