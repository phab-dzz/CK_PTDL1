import pandas as pd
from models.decision_tree import BestDTC
from data.preprocessing import preprocess_data
from data.feature_engineering import feature_engineering
from api.prediction_service import PredictionService

def main():
    # Load and preprocess the data
    df = pd.read_csv('data/raw/drug200.csv')
    processed_data = preprocess_data(df)
    features, target = feature_engineering(processed_data)

    # Initialize and train the model
    model = BestDTC()
    model.train(features, target)

    # Start the prediction service
    prediction_service = PredictionService(model)
    prediction_service.run()

if __name__ == "__main__":
    main()