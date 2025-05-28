class PredictionService:
    def __init__(self, model):
        self.model = model

    def predict(self, features):
        prediction = self.model.predict([features])
        return prediction[0]

    def load_model(self, model_path):
        import joblib
        self.model = joblib.load(model_path)

    def save_model(self, model_path):
        import joblib
        joblib.dump(self.model, model_path)