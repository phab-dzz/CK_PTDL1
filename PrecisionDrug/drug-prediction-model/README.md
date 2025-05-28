# Drug Prediction Model

This project implements a drug prediction model using a Decision Tree Classifier. The model is designed to predict the type of drug a patient should receive based on various health metrics.

## Project Structure

```
drug-prediction-model
├── src
│   ├── models
│   │   ├── __init__.py
│   │   ├── decision_tree.py
│   │   └── model_evaluation.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   ├── api
│   │   ├── __init__.py
│   │   └── prediction_service.py
│   └── main.py
├── data
│   ├── raw
│   │   └── drug200.csv
│   └── processed
├── notebooks
│   └── exploratory_analysis.ipynb
├── tests
│   ├── __init__.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── config
│   └── model_config.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd drug-prediction-model
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**: Use the functions in `src/data/preprocessing.py` to clean and prepare the dataset.
2. **Feature Engineering**: Utilize `src/data/feature_engineering.py` to select and engineer features for the model.
3. **Model Training**: Train the Decision Tree Classifier using the `BestDTC` class in `src/models/decision_tree.py`.
4. **Model Evaluation**: Evaluate the model's performance using the functions in `src/models/model_evaluation.py`.
5. **Prediction Service**: Start the prediction service by running `src/main.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.