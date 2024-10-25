import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from model import preprocess_data


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluates the given models using cross-validation and calculates AUC score on test set.

    Args:
        models (list): A list of tuples, where each tuple contains the model name and the model instance.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test labels.

    Returns:
        pd.DataFrame: A dataframe containing the evaluation results for each model.
    """
    results = []

    for model_name, model in models:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring='roc_auc')

        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'Model': model_name,
            'Mean cv AUC ': scores.mean(),
            'Test AUC': auc_score
        })

    return pd.DataFrame(results)


def train_and_save_model(file_path):
    """
    Trains and saves the best performing model to disk.

    The function reads the data from the specified file path, preprocesses it,
    splits it into train and test sets, extracts features, trains several models,
    evaluates their performance, selects the best model based on mean AUC score,
    retrains the best model on the entire training set, and saves the trained model to disk.

    Args:
        file_path (str): The path to the CSV file containing the data.
    """
    df = pd.read_csv(file_path, delimiter=';')

    processed_data = preprocess_data(df)
    train_set, test_set = train_test_split_data(processed_data)
    train_set, test_set = extract_features(train_set, test_set)

    numerical_columns = test_set.select_dtypes(
        include=['int64', 'float64']).columns

    X_train = train_set.loc[:, numerical_columns].drop(
        columns=['Target', 'ID'])
    X_test = test_set.loc[:, numerical_columns].drop(columns=['Target', 'ID'])
    y_train = train_set['Target']
    y_test = test_set['Target']

    models = [
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1))
    ]

    results = evaluate_models(models, X_train, y_train, X_test, y_test)

    best_model_name = results.sort_values(
        'Mean AUC', ascending=False).iloc[0]['Model']
    best_model = next(
        model for name, model in models if name == best_model_name)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    pipeline.fit(X_train, y_train)

    output_path = '../data/trained_model.pkl'
    joblib.dump(pipeline, output_path)
    joblib.dump(X_train.columns, '../data/model_features.pkl')
    print(f"Best model '{best_model_name}' saved to {output_path}")


if __name__ == "__main__":
    file_path = "../data/train.csv"
    train_and_save_model(file_path)
