from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer as CV  # Rename to be explicit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# preprocessor.py
print("Starting imports...")

import numpy as np
print("numpy imported")

import torch
print("torch imported")

try:
    from sklearn.feature_extraction.text import CountVectorizer
    print("CountVectorizer imported successfully")
except Exception as e:
    print(f"Error importing CountVectorizer: {str(e)}")

from nltk.corpus import stopwords
print("stopwords imported")

from nltk.stem import WordNetLemmatizer
print("WordNetLemmatizer imported")

import nltk
print("nltk imported")

# Rest of your code...

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]


class TextPreprocessor:
    def __init__(self, tokenizer=None, stop_words=None, max_df=0.4, min_df=0.0006, 
                 ngram_range=(1, 1), has_environments=True):
        self.tokenizer = tokenizer if tokenizer else LemmaTokenizer()
        self.stop_words = stop_words if stop_words else all_stopwords
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.has_environments = has_environments
        self.vectorizer = None
        self.env_mapping = None
        
    def _create_vectorizer(self):
        """Create and return a CountVectorizer with specified parameters."""
        return CV(  # Use CV instead of CountVectorizer
            tokenizer=self.tokenizer,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df
        )
    
    def _process_environments(self, data):
        """
        Process environment data and create environment index tensor.
        
        Args:
            data: Pandas DataFrame containing the data
            
        Returns:
            torch.Tensor: Environment index tensor
        """
        if not self.has_environments:
            return None
            
        # Create environment mapping if it doesn't exist
        if self.env_mapping is None:
            self.env_mapping = {value: index for index, value 
                              in enumerate(data['source'].unique())}
            
        # Create environment index matrix
        num_docs = len(data)
        num_envs = len(self.env_mapping)
        env_index_matrix = np.zeros((num_docs, num_envs), dtype=int)
        
        # Fill environment matrix
        for doc_idx, source in enumerate(data['source']):
            env_idx = self.env_mapping[source]
            env_index_matrix[doc_idx, env_idx] = 1
            
        return torch.from_numpy(env_index_matrix).float()
    
    def prepare_for_training(self, train_data, test_data=None, device='cuda'):
        """
        Process data and return everything needed for model training.
        
        Args:
            train_data: DataFrame containing training data
            test_data: Optional DataFrame containing test data
            device: Device to place tensors on
            
        Returns:
            dict: All components needed for model training
        """
        # Initialize vectorizer if not already done
        if self.vectorizer is None:
            self.vectorizer = self._create_vectorizer()
            
        # Process text data
        docs_word_matrix_raw = self.vectorizer.fit_transform(train_data['text'])
        docs_word_matrix_tensor = torch.from_numpy(docs_word_matrix_raw.toarray()).float().to(device)
        
        # Process environment data if it exists
        env_index_tensor = None
        num_envs = 1  # default for no environments
        if self.has_environments:
            env_index_tensor = self._process_environments(train_data)
            env_index_tensor = env_index_tensor.to(device)
            num_envs = len(self.env_mapping)
        
        # Process test data if provided
        test_tensor = None
        if test_data is not None:
            test_matrix_raw = self.vectorizer.transform(test_data['text'])
            test_tensor = torch.from_numpy(test_matrix_raw.toarray()).float().to(device)
        
        # Get vocabulary size
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
