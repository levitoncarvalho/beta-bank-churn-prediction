from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.utils import upsample
import config

def train_base_model(X_train, y_train, random_state=config.RANDOM_STATE):
    #Train a baseline Random Forest without balancing.
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_with_class_weight(X_train, y_train, random_state=config.RANDOM_STATE):
    #Train Random Forest using class_weight='balanced'.
    model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_with_upsampling(X_train, y_train, repeat=4, random_state=config.RANDOM_STATE):
    # Train Random Forest on upsampled data.
    X_up, y_up = upsample(X_train, y_train, repeat=repeat, random_state=random_state)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_up, y_up)
    return model

def optimize_hyperparameters(X_train, y_train, random_state=config.RANDOM_STATE, cv_folds=config.CV_FOLDS):
    #Perform hyperparameter tuning using RandomizedSearchCV on upsampled data.
    X_up, y_up = upsample(X_train, y_train, repeat=4, random_state=random_state)

    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    rf = RandomForestClassifier(random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv_folds,
        scoring='f1',
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(X_up, y_up)
    return random_search.best_estimator_