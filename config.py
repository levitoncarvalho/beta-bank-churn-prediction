# Global Configuration

DATA_PATH = 'data/Churn.csv'
RANDOM_STATE = 54321
TARGET_COL = 'Exited'
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_SAVE_PATH = 'models/rf_model.joblib'

NUMERIC_COLS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
COLS_TO_DROP = ['RowNumber', 'CustomerId', 'Surname']