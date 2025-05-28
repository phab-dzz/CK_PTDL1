class BestDTC:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        import joblib
        joblib.dump(self.model, file_path)

    @classmethod
    def load_model(cls, file_path):
        import joblib
        model = joblib.load(file_path)
        return cls(model)