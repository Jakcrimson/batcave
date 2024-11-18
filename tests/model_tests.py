import unittest
import pandas as pd
from scripts.model import (
    extract_syntactic_features,
    preprocess_data,
    train_test_split_data,
    convert_float_to_int,
    preprocess_review_data
)

class TestSemantiquityModel(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_content': ['Ce produit est excellent !', 'Je n\'aime pas ce produit.', 'Produit moyen.'],
            'product': ['A', 'B', 'C'],
            'Target': [1, 0, 1],
            'ID': [1, 2, 3]
        })
        self.sample_data['product_review_count'] = 1
        self.sample_data['review_freq'] = 1
        self.sample_data['review_length'] = [20, 25, 15]

    def test_extract_syntactic_features(self):
        text = "Ce produit est excellent !"
        features = extract_syntactic_features(text)
        self.assertIn('avg_sentence_length', features)
        self.assertIn('uppercase_count', features)
        self.assertIn('punctuation_counts.!', features)
        self.assertIn('function_word_counts.le', features)
        self.assertIn('pos_NOUN', features)

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.sample_data)
        expected_columns = [
            'review_length', 'word_count', 'unique_words_ratio', 'review_freq', 'product_review_count'
        ]
        for col in expected_columns:
            self.assertIn(col, processed_data.columns)

    def test_train_test_split_data(self):
        train_set, test_set = train_test_split_data(self.sample_data)
        self.assertGreater(len(train_set), 0)
        self.assertGreater(len(test_set), 0)

    def test_convert_float_to_int(self):
        df = pd.DataFrame({
            'float_col': [1.0, 2.0, 3.0],
            'int_col': [1, 2, 3]
        })
        converted_df = convert_float_to_int(df)
        self.assertEqual(converted_df['float_col'].dtype, 'int64')
        self.assertEqual(converted_df['int_col'].dtype, 'int64')

    def test_preprocess_review_data(self):
        review_content = "Ce produit est excellent !"
        review_title = "Super produit"
        review_stars = 5.0
        review_df = preprocess_review_data(review_content, review_title, review_stars)
        expected_columns = [
            'review_length', 'word_count', 'unique_words_ratio', 'review_freq',
            'product_review_count', 'review_stars'
        ]
        for col in expected_columns:
            self.assertIn(col, review_df.columns)


if __name__ == '__main__':
    unittest.main()