import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import spacy
import nltk
import string
from tqdm import tqdm
import time

nlp = spacy.load('fr_core_news_sm')

def extract_syntactic_features(original_txt):
    """
    Extract various syntactic features from the given text.
    """
    text = original_txt.lower()
    expected_keys = {
        'avg_sentence_length': 0,
        'uppercase_count': 0,
        **{f'punctuation_counts.{char}': 0 for char in string.punctuation},
        **{f'function_word_counts.{word}': 0 for word in ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais']},
        **{f'pos_{tag}': 0 for tag in ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']}
    }

    features = expected_keys.copy()

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    pos_counts = nltk.FreqDist(tag for word, tag in pos_tags)

    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    punctuation_counts = nltk.FreqDist(char for char in text if char in string.punctuation)
    function_words = ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais']
    function_word_counts = nltk.FreqDist(word for word in words if word.lower() in function_words)

    uppercase_count = sum(1 for char in original_txt if char.isupper())

    features['avg_sentence_length'] = avg_sentence_length
    features['uppercase_count'] = uppercase_count
    for char, count in punctuation_counts.items():
        features[f'punctuation_counts.{char}'] = count
    for word, count in function_word_counts.items():
        features[f'function_word_counts.{word}'] = count
    for tag, count in pos_counts.items():
        features[f'pos_{tag}'] = count

    return features

def preprocess_data(data):
    """
    Preprocess the data by removing duplicates, adding new features, and filtering.
    """
    data = data.drop_duplicates(subset=['review_content', 'product'])

    data['review_length'] = data['review_content'].apply(len)
    data['word_count'] = data['review_content'].apply(lambda x: len(word_tokenize(x)))
    data['unique_words_ratio'] = data['review_content'].apply(lambda x: len(set(word_tokenize(x))) / len(word_tokenize(x)))
    data['review_freq'] = data.groupby('review_content')['review_content'].transform('count')
    data['product_review_count'] = data.groupby('product')['product'].transform('count')

    data = data[data['product_review_count'] <= 100]
    data = data[data['review_freq'] < 5]
    data = data[(data['review_length'] >= 50) & (data['review_length'] <= 500)]

    data = data.reset_index(drop=True)

    return data

def train_test_split_data(data):
    """
    Split the data into train and test sets using stratified sampling.
    """
    data['product_stratify'] = data['product']

    min_reviews = 10
    data.loc[data['product_review_count'] < min_reviews, 'product_stratify'] = 'Other'

    data['combined_stratify'] = data['product_stratify'] + '_' + data['Target'].astype(str)

    combined_counts = data['combined_stratify'].value_counts()
    single_sample_combinations = combined_counts[combined_counts == 1].index
    filtered_data = data[~data['combined_stratify'].isin(single_sample_combinations)]

    filtered_data = filtered_data.reset_index(drop=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(filtered_data, filtered_data['combined_stratify']):
        train_set = filtered_data.loc[train_index]
        test_set = filtered_data.loc[test_index]

    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)

def convert_float_to_int(df):
    """
    Convert float columns to int if all values are integers.
    """
    for col in df.columns:
        if df[col].dtype == 'float64' and df[col].apply(lambda x: x.is_integer()).all():
            df[col] = df[col].astype('int64')
    return df

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluate the performance of multiple models using cross-validation and test set.
    """
    results = []

    for model_name, model in models:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        start_time = time.time()
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        cross_val_time = time.time() - start_time

        print(f'AUC scores during cross-validation for {model_name}: {scores}')
        print(f'Mean AUC score for {model_name}: {scores.mean()}')

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time

        start_time = time.time()
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        predict_time = time.time() - start_time

        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f'AUC score on test set for {model_name}: {auc_score}')

        results.append({
            'Model': model_name,
            'Cross-Validation AUC Scores': scores,
            'Mean Cross-Validation AUC Score': scores.mean(),
            'Test AUC Score': auc_score,
            'Cross-Validation Time (s)': cross_val_time,
            'Fit Time (s)': fit_time,
            'Predict Time (s)': predict_time
        })

    return pd.DataFrame(results)

def extract_features(train_set, test_set):
    """
    Extract syntactic features from the review content and add them to the train and test sets.
    """
    tqdm.pandas()

    train_set['syntactic_features'] = train_set['review_content'].progress_apply(extract_syntactic_features)
    test_set['syntactic_features'] = test_set['review_content'].progress_apply(extract_syntactic_features)

    train_features_df = pd.json_normalize(train_set['syntactic_features'])
    test_features_df = pd.json_normalize(test_set['syntactic_features'])

    train_set = pd.concat([train_set.drop(columns=['syntactic_features']), train_features_df], axis=1)
    test_set = pd.concat([test_set.drop(columns=['syntactic_features']), test_features_df], axis=1)

    return train_set, test_set

def preprocess_review_data(review_content, review_title, review_stars):
    """
    Preprocess a single review by extracting syntactic features and other relevant information.
    """
    review_text = review_title + " " + review_content

    syntactic_features = extract_syntactic_features(review_text)

    review_df = pd.DataFrame([syntactic_features])

    review_df['review_length'] = len(review_content)
    review_df['word_count'] = len(word_tokenize(review_content))
    review_df['unique_words_ratio'] = len(set(word_tokenize(review_content))) / len(word_tokenize(review_content))

    review_df['review_freq'] = 1
    review_df['product_review_count'] = 1
    review_df['review_stars'] = review_stars

    review_df = convert_float_to_int(review_df)

    return review_df