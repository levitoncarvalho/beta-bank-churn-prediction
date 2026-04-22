import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def load_and_clean_data(filepath=config.DATA_PATH):
    # Load the dataset and perform initial cleaning.
    df = pd.read_csv(filepath)

    # Fill missing Tenure values with median
    df['Tenure'].fillna(df['Tenure'].median(), inplace=True)

    # Drop irrelevant columns
    df = df.drop(config.COLS_TO_DROP, axis=1)

    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    return df

def split_and_scale_data(df, target_col=config.TARGET_COL, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    # Split data into train/test sets and scale numeric features.
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train[config.NUMERIC_COLS])
    X_train[config.NUMERIC_COLS] = scaler.transform(X_train[config.NUMERIC_COLS])
    X_test[config.NUMERIC_COLS] = scaler.transform(X_test[config.NUMERIC_COLS])

    return X_train, X_test, y_train, y_test, scaler