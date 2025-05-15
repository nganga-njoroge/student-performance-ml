import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(path="data/student-mat.csv"):
    # Load dataset
    df = pd.read_csv(path, sep=";")

    # Target variable
    target = "G3"

    # Features to use
    features = ["sex", "age", "studytime", "failures", "absences", "G1", "G2"]

    # Select data
    X = df[features]
    y = df[target]

    # Encode categorical variables
    X = pd.get_dummies(X, columns=["sex"], drop_first=True)

    # Normalize numeric features
    numeric_cols = ["age", "studytime", "failures", "absences", "G1", "G2"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
