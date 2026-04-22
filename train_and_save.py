# train_and_save.py
import os
import joblib
from src.data_preparation import load_and_clean_data, split_and_scale_data
from src.model_training import optimize_hyperparameters

def main():
    # Load and prepare data using config defaults
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df)

    print("Optimizing model with RandomizedSearchCV...")
    best_model = optimize_hyperparameters(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("Model and scaler saved to 'models/' directory.")

if __name__ == '__main__':
    main()