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
    python preprocessor.py [--filename <file_prefix>]
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
    
    def prepare_for_training(self, train_data, test_data=None, device='cpu', output_dir=None, filename="mtm"):
        if not output_dir:
            # Set default output directory: repo/data/preprocessed_data/<filename>
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            output_dir = os.path.join(repo_root, 'data', 'preprocessed_data', filename)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.vectorizer is None:
            self.vectorizer = self._create_vectorizer()
            
        docs_word_matrix_raw = self.vectorizer.fit_transform(train_data['text'])
        docs_word_matrix_tensor = torch.from_numpy(docs_word_matrix_raw.toarray()).float().to(device)
        
        env_index_tensor = None
        if self.has_environments:
            env_index_tensor = self._process_environments(train_data)
            env_index_tensor = env_index_tensor.to(device)
        
        # Save the preprocessed data
        train_file = f"{filename}_train.pt"
        env_file = f"{filename}_env.pt"
        torch.save(docs_word_matrix_tensor, os.path.join(output_dir, train_file))
        if env_index_tensor is not None:
            torch.save(env_index_tensor, os.path.join(output_dir, env_file))
        
        if test_data is not None:
            test_matrix_raw = self.vectorizer.transform(test_data['text'])
            test_tensor = torch.from_numpy(test_matrix_raw.toarray()).float().to(device)
            test_file = f"{filename}_test.pt"
            torch.save(test_tensor, os.path.join(output_dir, test_file))
        
        print(f"Preprocessed files saved in '{output_dir}' with prefix '{filename}'.")

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
    
    return list(nltk_stopwords.union(additional_stopwords))


def main():
    parser = argparse.ArgumentParser(description="Run the Preprocessor for text data.")
    parser.add_argument('--min_df', type=float, default=0.0006, help="Minimum document frequency (default: 0.0006)")
    parser.add_argument('--max_df', type=float, default=0.4, help="Maximum document frequency (default: 0.4)")
    parser.add_argument('--filename', type=str, default="mtm", help="Filename prefix for output files (default: 'mtm').")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_file_path = os.path.join(script_dir, 'political_stopwords.txt') 
    
    train_data_path = os.getenv('TRAIN_DATA_PATH')
    test_data_path = os.getenv('TEST_DATA_PATH')
    
    if not train_data_path:
        raise ValueError("Please set the TRAIN_DATA_PATH environment variable to the training data location.")
    if not os.path.exists(stopwords_file_path):
        raise ValueError(f"Stopwords file not found: {stopwords_file_path}")
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path) if test_data_path else None

    combined_stopwords = load_political_stopwords(stopwords_file_path)

    preprocessor = TextPreprocessor(
        stop_words=combined_stopwords,
        max_df=args.max_df,
        min_df=args.min_df,
        has_environments=True
    )
    
    preprocessor.prepare_for_training(
        train_data,
        test_data,
        device='cpu',
        filename=args.filename
    )

if __name__ == "__main__":
    main()
