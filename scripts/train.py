import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os

from model import preprocess_data, train_test_split_data, extract_features

MODEL_CONFIGS = [
    ('logreg', LogisticRegression(max_iter=1000)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
]

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')

    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    return {
        'Model': model_name,
        'Mean AUC': scores.mean(),
        'Test AUC': auc_score
    }

def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = [evaluate_model(name, model, X_train, y_train, X_test, y_test) for name, model in models]
    return pd.DataFrame(results)

def train_and_save_model(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    processed_data = preprocess_data(df)
    train_set, test_set = train_test_split(processed_data)
    # train_set, test_set = extract_features(train_set, test_set)

    numerical_columns = test_set.select_dtypes(include=['int64', 'float64']).columns
    X_train = train_set.loc[:, numerical_columns].drop(columns=['Target', 'ID'])
    X_test = test_set.loc[:, numerical_columns].drop(columns=['Target', 'ID'])
    y_train = train_set['Target']
    y_test = test_set['Target']

    results = evaluate_models(MODEL_CONFIGS, X_train, y_train, X_test, y_test)
    best_model_name = results.sort_values('Mean AUC', ascending=False).iloc[0]['Model']
    best_model = next(model for name, model in MODEL_CONFIGS if name == best_model_name)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    pipeline.fit(X_train, y_train)

    output_path = 'data/trained_model.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    joblib.dump(X_train.columns, 'data/model_features.pkl')
    print(f"Best model '{best_model_name}' saved to {output_path}")
    return f"Best model '{best_model_name}' saved to {output_path}"

if __name__ == "__main__":
    file_path = "../data/train.csv"
    train_and_save_model(file_path)
