# model_config.py

MODEL_CONFIG = {
    'model_type': 'DecisionTreeClassifier',
    'criterion': 'gini',
    'max_depth': 5,
    'max_leaf_nodes': 6,
    'min_samples_split': 2,
    'model_path': 'src/models/bestDTC_model.pkl',
    'data_path': 'data/processed/drug_data_processed.csv',
    'target_column': 'Drug',
    'features': [
        'Age',
        'Blood_Pressure',
        'Cholesterol',
        'Sodium_to_Potassium',
        'Sex_Male',
        'Sex_Female'
    ]
}