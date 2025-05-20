from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os


def create_model_pipeline(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("features_selector", SelectPercentile(chi2, percentile=5)),
        ("model", RandomForestClassifier())
    ])


def train_model(x_train, y_train, model_pipeline, param_grid=None):
    if param_grid is None:
        param_grid = {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "features_selector__percentile": [1, 5, 10]
        }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring="recall_weighted",
        cv=4,
        verbose=2
    )

    grid_search.fit(x_train, y_train)
    return grid_search


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return classification_report(y_test, y_pred)


def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path):
    return joblib.load(model_path)
