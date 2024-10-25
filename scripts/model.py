import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import StratifiedShuffleSplit
import spacy
import nltk
import string
from tqdm import tqdm

nlp = spacy.load('fr_core_news_sm')


def extract_syntactic_features(original_txt):
    """
    Extracts syntactic features from a given text.

    Args:
        original_txt (str): The input text.

    Returns:
        dict: A dictionary containing the extracted syntactic features.
    """
    text = original_txt.lower()
    features = {
        'avg_sentence_length': 0,
        'uppercase_count': 0,
        **{f'punctuation_counts.{char}': 0 for char in string.punctuation},
        **{f'function_word_counts.{word}': 0 for word in ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais']},
        **{f'pos_{tag}': 0 for tag in ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']}
    }

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    pos_counts = nltk.FreqDist(tag for word, tag in pos_tags)

    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    features['avg_sentence_length'] = sum(sentence_lengths) / \
        len(sentence_lengths) if sentence_lengths else 0

    features['uppercase_count'] = sum(1 for char in original_txt if char.isupper())

    features.update(dict(nltk.FreqDist(
        char for char in text if char in string.punctuation)))
    features.update(dict(nltk.FreqDist(
        word for word in words if word.lower() in ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais'])))
    features.update(dict(pos_counts))

    return features


def preprocess_data(data):
    """
    Preprocesses the input dataframe.

    Args:
        data (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    data = data.drop_duplicates(subset=['review_content', 'product'])

    data['review_length'] = data['review_content'].apply(len)
    data['word_count'] = data['review_content'].apply(
        lambda x: len(word_tokenize(x)))
    data['unique_words_ratio'] = data['review_content'].apply(
        lambda x: len(set(word_tokenize(x))) / len(word_tokenize(x)))
    data['review_freq'] = data.groupby('review_content')[
        'review_content'].transform('count')
    data['product_review_count'] = data.groupby(
        'product')['product'].transform('count')

    data = data[data['product_review_count'] <= 100]
    data = data[data['review_freq'] < 5]
    data = data[(data['review_length'] >= 50) & (data['review_length'] <= 500)]

    return data.reset_index(drop=True)


def prepare_data(data, test_size=0.2):
    """
    Prepares the data for training and testing.

    This function combines the data preprocessing, train-test splitting, and feature extraction steps.

    Args:
        data (pd.DataFrame): The input dataframe.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training features, training labels, test features, and test labels.
    """
    data['product_stratify'] = data['product']
    data.loc[data['product_review_count']
             < 10, 'product_stratify'] = 'Other'
    data['combined_stratify'] = data['product_stratify'] + \
        '_' + data['Target'].astype(str)

    data = data[~data['combined_stratify'].isin(
        data['combined_stratify'].value_counts()[data['combined_stratify'].value_counts() == 1].index)].reset_index(drop=True)

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(data, data['combined_stratify']):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    train_set, test_set = train_set.reset_index(
        drop=True), test_set.reset_index(drop=True)

    tqdm.pandas()

    train_set['syntactic_features'] = train_set['review_content'].progress_apply(
        extract_syntactic_features)
    test_set['syntactic_features'] = test_set['review_content'].progress_apply(
        extract_syntactic_features)

    train_features_df = pd.json_normalize(train_set['syntactic_features'])
    test_features_df = pd.json_normalize(test_set['syntactic_features'])

    train_set = pd.concat(
        [train_set.drop(columns=['syntactic_features']), train_features_df], axis=1)
    test_set = pd.concat(
        [test_set.drop(columns=['syntactic_features']), test_features_df], axis=1)

    numerical_columns = test_set.select_dtypes(
        include=['int64', 'float64']).columns

    X_train = train_set.loc[:, numerical_columns].drop(
        columns=['Target', 'ID'])
    X_test = test_set.loc[:, numerical_columns].drop(columns=['Target', 'ID'])
    y_train = train_set['Target']
    y_test = test_set['Target']

    return X_train, y_train, X_test, y_test


def preprocess_review_data(review_content, review_title, review_stars):
    """
    Preprocesses the review data for a single review.

    Args:
        review_content (str): The content of the review.
        review_title (str): The title of the review.
        review_stars (float): The star rating of the review.

    Returns:
        pd.DataFrame: A dataframe containing the preprocessed review data.
    """
    review_text = review_title + " " + review_content

    syntactic_features = extract_syntactic_features(review_text)

    review_df = pd.DataFrame([syntactic_features])

    review_df['review_length'] = len(review_content)
    review_df['word_count'] = len(word_tokenize(review_content))
    review_df['unique_words_ratio'] = len(
        set(word_tokenize(review_content))) / len(word_tokenize(review_content))

    review_df['review_freq'] = 1
    review_df['product_review_count'] = 1
    review_df['review_stars'] = review_stars

    for col in review_df.columns:
        if review_df[col].dtype == 'float64' and review_df[col].apply(lambda x: x.is_integer()).all():
            review_df[col] = review_df[col].astype('int64')
    return review_df

