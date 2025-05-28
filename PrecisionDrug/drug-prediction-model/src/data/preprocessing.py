from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    imputer = SimpleImputer(strategy='mean')
    df['Sodium_to_Potassium'] = imputer.fit_transform(df[['Sodium_to_Potassium']])
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using OneHotEncoder and LabelEncoder."""
    # OneHotEncoding for 'Sex'
    if 'Sex' in df.columns:
        onehot_encoder = OneHotEncoder(sparse=False)
        sex_encoded = onehot_encoder.fit_transform(df[['Sex']])
        df = df.join(pd.DataFrame(sex_encoded, columns=onehot_encoder.get_feature_names_out(['Sex'])))
        df.drop('Sex', axis=1, inplace=True)

    # LabelEncoding for 'Blood_Pressure' and 'Cholesterol'
    label_encoder_bp = LabelEncoder()
    label_encoder_chol = LabelEncoder()
    df['Blood_Pressure'] = label_encoder_bp.fit_transform(df['Blood_Pressure'])
    df['Cholesterol'] = label_encoder_chol.fit_transform(df['Cholesterol'])

    return df

def scale_features(df):
    """Scale features if necessary (placeholder for future scaling methods)."""
    # Placeholder for scaling features
    return df

def preprocess_data(filepath):
    """Main function to preprocess the data."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    df = scale_features(df)
    return df

def split_data(df, target_column, test_size=0.4, random_state=42):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)