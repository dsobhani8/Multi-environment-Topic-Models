#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessor Module

This script processes text data for machine learning tasks, including tokenization, 
lemmatization, and environment-specific indexing. It supports training and testing 
data preprocessing with configurable parameters.

Usage:
    Set environment variables for data paths:
    - TRAIN_DATA_PATH: Path to training data
    - TEST_DATA_PATH: Path to test data (optional)

Run the script:
    python preprocessor.py
"""
import os
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import argparse

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]

class TextPreprocessor:
    def __init__(self, stop_words=None, max_df=0.4, min_df=0.0006,  
                 ngram_range=(1, 1), has_environments=True):
        self.tokenizer = LemmaTokenizer()
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.has_environments = has_environments
        self.vectorizer = None
        self.env_mapping = None
        
    def _create_vectorizer(self):
        return CountVectorizer(
            tokenizer=self.tokenizer,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df
        )
    
    def _process_environments(self, data):
        if not self.has_environments:
            return None
            
        if self.env_mapping is None:
            self.env_mapping = {value: index for index, value 
                              in enumerate(data['source'].unique())}
            
        num_docs = len(data)
        num_envs = len(self.env_mapping)
        env_index_matrix = np.zeros((num_docs, num_envs), dtype=int)
        
        for doc_idx, source in enumerate(data['source']):
            env_idx = self.env_mapping[source]
            env_index_matrix[doc_idx, env_idx] = 1
            
        return torch.from_numpy(env_index_matrix).float()
    
    def prepare_for_training(self, train_data, test_data=None, device='cuda'):
        if self.vectorizer is None:
            self.vectorizer = self._create_vectorizer()
            
        docs_word_matrix_raw = self.vectorizer.fit_transform(train_data['text'])
        docs_word_matrix_tensor = torch.from_numpy(docs_word_matrix_raw.toarray()).float().to(device)
        
        env_index_tensor = None
        num_envs = 1
        if self.has_environments:
            env_index_tensor = self._process_environments(train_data)
            env_index_tensor = env_index_tensor.to(device)
            num_envs = len(self.env_mapping)
        
        test_tensor = None
        if test_data is not None:
            test_matrix_raw = self.vectorizer.transform(test_data['text'])
            test_tensor = torch.from_numpy(test_matrix_raw.toarray()).float().to(device)
        
        vocab_size = len(self.vectorizer.get_feature_names_out())
        
        return {
            'training': {
                'docs_word_matrix_tensor': docs_word_matrix_tensor,
                'env_index_tensor': env_index_tensor,
                'num_envs': num_envs,
                'vocab_size': vocab_size
            },
            'test': {
                'tensor': test_tensor
            } if test_data is not None else None,
            'vectorizer': self.vectorizer,
            'env_mapping': self.env_mapping
        }

def load_political_stopwords(file_path):
    """
    Load additional stopwords from a file and merge them with NLTK stopwords.
    
    Args:
        file_path: Path to the file containing additional stopwords.
    
    Returns:
        list: Combined list of stopwords.
    """
    nltk_stopwords = set(stopwords.words('english'))
    try:
        with open(file_path, 'r') as f:
            additional_stopwords = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        raise ValueError(f"Stopwords file not found: {file_path}")
    
    # Combine stopwords and convert to a list
    return list(nltk_stopwords.union(additional_stopwords))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the Preprocessor for text data.")
    parser.add_argument('--min_df', type=float, default=0.0006, help="Minimum document frequency (default: 0.0006)")
    parser.add_argument('--max_df', type=float, default=0.4, help="Maximum document frequency (default: 0.4)")
    args = parser.parse_args()
    
    # Get the current script directory to construct relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_file_path = os.path.join(script_dir, 'political_stopwords.txt')  # Relative path to the stopwords file
    
    # Get paths for training and testing data
    train_data_path = os.getenv('TRAIN_DATA_PATH')
    test_data_path = os.getenv('TEST_DATA_PATH')
    
    if not train_data_path:
        raise ValueError("Please set the TRAIN_DATA_PATH environment variable to the training data location.")
    if not os.path.exists(stopwords_file_path):
        raise ValueError(f"Stopwords file not found: {stopwords_file_path}")
    if not test_data_path:
        print("TEST_DATA_PATH environment variable not set. Only training data will be processed.")
    
    # Load datasets
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path) if test_data_path else None

    # Load stopwords
    combined_stopwords = load_political_stopwords(stopwords_file_path)

    # Initialize the preprocessor with dynamic arguments
    preprocessor = TextPreprocessor(
        stop_words=combined_stopwords,
        max_df=args.max_df,
        min_df=args.min_df,
        has_environments=True
    )
    
    # Prepare data for training
    prepared_data = preprocessor.prepare_for_training(train_data, test_data, device='cpu')
    
    # Print the output for verification
    print("Training Data Processed:")
    print(prepared_data['training'])
    if test_data_path:
        print("Test Data Processed:")
        print(prepared_data['test'])

if __name__ == "__main__":
    main()