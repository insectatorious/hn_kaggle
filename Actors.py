#! /usr/bin/env python

import pykka
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

class Word_to_Vec(pykka.ThreadingActor):
    def __init__(self, model_path="GoogleNews-vectors-negative300.bin"):
        super(Word_to_Vec, self).__init__()
        self.model = Word2Vec.load_word2vec_format(model_path, binary=True)
        self.index2word_set = set(self.model.index2word)
        self.num_features = len(self.model[self.model.index2word[0]])

    def resolve(self, word):
        vector = np.zeros((self.num_features), dtype="float64")
        if word in self.index2word_set:
            vector = self.model[word]
        return vector

    def num_features(self, msg):
        return self.num_features

    def index2word_set(self, msg):
        return self.index2word_set

class Title_to_Vec(pykka.ThreadingActor):
    def __init__(self, vectoriser, model):
        super(Title_to_Vec, self).__init__()
        self.vectoriser = vectoriser # Assume the TF-IDF matrix has been fit
        self.feature_names = self.vectoriser.get_feature_names()
        self.model = model # Should be a Word_to_Vec actor proxy
        self.index2word_set = self.model.index2word_set.get()
        self.num_features = self.model.num_features.get()

    def resolve(self, words):
        title_vector = np.zeros((self.num_features), dtype="float64")
        nwords = 0
        response = self.vectoriser.transform([words])
        for col in response.nonzero()[1]:
            word = self.feature_names[col]
            if word in self.index2word_set:
                word_tfidf = response[0, col]
                word_vector = self.model.resolve(word).get()
                nwords = nwords + 1
                title_vector = np.add(title_vector, word_vector*word_tfidf)
        if nwords > 0:
            title_vector = np.divide(title_vector, nwords)
        return title_vector 
