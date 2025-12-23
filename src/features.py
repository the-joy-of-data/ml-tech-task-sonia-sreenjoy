import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

import spacy
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GermanMedicalFeatureExtractor:
    """
    Build and extracyts features for German medical text classification.

    Features:
    1. TF-IDF (char n-grams 2-5 for compound words)
    2. Medical vocabulary overlap score
    3. POS tag distributions
    4. Sentence-level statistics
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.nlp = None
        self.tfidf_vectorizer = None
        self.medical_vocab = None

        # SpaCy model in DE
        try:
            self.nlp = spacy.load("de_core_news_lg")
            logger.info("Loaded SpaCy German model: de_core_news_lg")
        except OSError:
            logger.error("SpaCy model not found. Run: python -m spacy download de_core_news_lg")
            raise

    def _build_medical_vocabulary(self, corpus: List[str], labels: np.ndarray) -> set:
        """
        Extract medical vocabulary from positive samples (is_medical=True).
        Uses frequency analysis and POS filtering (nouns, adjectives).

        Args:
            corpus: List of text samples
            labels: Binary labels (1=medical, 0=non-medical)

        Returns:
            Vocabulary of Medical terms (set)
        """
        medical_samples = [text for text, label in zip(corpus, labels) if label == 1]

        medical_tokens = []

        for doc in self.nlp.pipe(medical_samples, batch_size=50):
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and
                    not token.is_stop and
                    len(token.text) >= 3):
                    medical_tokens.append(token.lemma_.lower())

        term_freq = Counter(medical_tokens)
        top_medical_terms = {term for term, count in term_freq.most_common(1000)}

        # adding common German medical suffixes/prefixes (asked an LLM to generate this list of medical patterns in German)
        medical_patterns = {
            'krank', 'schmerz', 'patient', 'diagnos', 'behandl', 'therapie',
            'operation', 'medikament', 'symptom', 'infektion', 'blut', 'herz',
            'lunge', 'thrombos', 'entzünd', 'fieber', 'husten', 'arzt'
        }
        top_medical_terms.update(medical_patterns)

        logger.info(f"Built medical vocabulary: {len(top_medical_terms)} terms")
        return top_medical_terms


    def _extract_tfidf_features(self, corpus: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features with char n-grams for German compounds.

        Args:
            corpus: List of text samples

        Returns:
            TF-IDF feature matrix (n_samples, n_features)
        """
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),   #unigrams + bigrams
                analyzer='char_wb',   #character n-grams within word boundaries
                min_df=2,            # <2 documents ignore
                max_df=0.8,          # >80% documents ignore
                sublinear_tf=True,   # log scaling to TFs
                strip_accents=None   # keep special German characters
            )
            X_tfidf = self.tfidf_vectorizer.fit_transform(corpus)
            logger.info(f"TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        else:
            X_tfidf = self.tfidf_vectorizer.transform(corpus)

        return X_tfidf.toarray()

    def _extract_medical_vocab_overlap(self, corpus: List[str]) -> np.ndarray:
        """
        Compute overlap score with medical vocabulary (self made).

        Returns:
            Array of shape (n_samples, 1) with overlap ratios
        """
        overlap_scores = []

        for text in corpus:
            doc = self.nlp(text)
            lemmas = {token.lemma_.lower() for token in doc if not token.is_stop}

            if len(lemmas) == 0:
                overlap_scores.append(0.0)
            else:
                overlap = len(lemmas & self.medical_vocab) / len(lemmas)
                overlap_scores.append(overlap)

        return np.array(overlap_scores).reshape(-1, 1)


    def _extract_pos_features(self, corpus: List[str]) -> np.ndarray:
        """
        Extract Parts of Speech (POS) tag distribution features.
        Medical texts have higher noun/adjective ratios.

        Returns:
            Array of shape (n_samples, 5) with POS ratios (5 because of the 5 different speecg parts)
        """
        pos_features = []

        for doc in self.nlp.pipe(corpus, batch_size=50):
            total_tokens = len(doc)
            if total_tokens == 0:
                pos_features.append([0, 0, 0, 0, 0])
                continue

            pos_counts = Counter([token.pos_ for token in doc])
            features = [
                pos_counts.get('NOUN', 0) / total_tokens,
                pos_counts.get('VERB', 0) / total_tokens,
                pos_counts.get('ADJ', 0) / total_tokens,
                pos_counts.get('PROPN', 0) / total_tokens,  # Proper nouns (drug names)
                pos_counts.get('NUM', 0) / total_tokens,    # Numbers (dosages)
            ]
            pos_features.append(features)

        return np.array(pos_features)

    def _extract_statistical_features(self, corpus: List[str]) -> np.ndarray:
        """
        Extract sentence-level statistics.

        Returns:
            Array of shape (n_samples, 4) with:
            - Sentence length (tokens)
            - Average word length
            - Lexical diversity (unique words / total words)
            - Contains numbers (0/1)
        """
        stat_features = []

        for doc in self.nlp.pipe(corpus, batch_size=50):
            tokens = [t for t in doc if not t.is_punct]
            n_tokens = len(tokens)

            if n_tokens == 0:
                stat_features.append([0, 0, 0, 0])
                continue

            avg_word_len = np.mean([len(t.text) for t in tokens])
            lexical_diversity = len(set([t.text for t in tokens])) / n_tokens
            has_numbers = int(any(t.like_num for t in tokens))

            stat_features.append([
                n_tokens,
                avg_word_len,
                lexical_diversity,
                has_numbers
            ])

        return np.array(stat_features)

    def fit_transform(self, corpus: List[str], labels: np.ndarray) -> np.ndarray:
        """
        Fit feature extractors on training data and transform.

        Args:
            corpus: List of text samples
            labels: Binary labels for medical vocabulary extraction

        Returns:
            Feature matrix (n_samples, n_features)
        """
        logger.info("Fitting feature extractors...")

        # build medical vocabulary from training data we received (HF data card)
        self.medical_vocab = self._build_medical_vocabulary(corpus, labels)

        # extract features
        tfidf_features = self._extract_tfidf_features(corpus)
        vocab_features = self._extract_medical_vocab_overlap(corpus)
        pos_features = self._extract_pos_features(corpus)
        stat_features = self._extract_statistical_features(corpus)

        # stack features
        X = np.hstack([
            tfidf_features,
            vocab_features,
            pos_features,
            stat_features
        ])

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"  - TF-IDF: {tfidf_features.shape[1]} features")
        logger.info(f"  - Medical vocab overlap: {vocab_features.shape[1]} features")
        logger.info(f"  - POS distributions: {pos_features.shape[1]} features")
        logger.info(f"  - Statistical: {stat_features.shape[1]} features")

        return X

    def transform(self, corpus: List[str]) -> np.ndarray:
        """
        Transform new or inference samples using previously fitted extractors.

        Args:
            corpus: List of text samples

        Returns:
            Feature matrix (n_samples, n_features)
        """
        if self.tfidf_vectorizer is None or self.medical_vocab is None:
            raise ValueError("No Feature Extractor fitted previously!")

        tfidf_features = self._extract_tfidf_features(corpus)
        vocab_features = self._extract_medical_vocab_overlap(corpus)
        pos_features = self._extract_pos_features(corpus)
        stat_features = self._extract_statistical_features(corpus)

        # stack features
        X = np.hstack([
            tfidf_features,
            vocab_features,
            pos_features,
            stat_features
        ])

        return X

    def get_feature_names(self) -> List[str]:
        """Get better feature names to understand impact of each on model (INTERPRETABILITY)."""
        feature_names = []

        # TF-IDF features
        if self.tfidf_vectorizer:
            feature_names.extend([f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()])

        # Manual features
        feature_names.extend(['medical_vocab_overlap'])
        feature_names.extend(['pos_noun_ratio', 'pos_verb_ratio', 'pos_adj_ratio',
                            'pos_propn_ratio', 'pos_num_ratio'])
        feature_names.extend(['sentence_length', 'avg_word_length',
                            'lexical_diversity', 'has_numbers'])

        return feature_names