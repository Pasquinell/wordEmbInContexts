# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')


embeddings_index = {}
embeddings_vectors = []
# to download the data http://nlp.stanford.edu/data/glove.6B.zip
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_vectors.append(coefs)
        embeddings_index[word] = coefs


from scipy import spatial
tree = spatial.KDTree(embeddings_vectors)
# tree.query(queen) -> (5.927124273586729, 1) (hence, the secod possition)

n_simWords = 10
# Extract the embedding vector
word_embedding = embeddings_index['woman']
# query the tree to obtain a (numpy array, nuympy array, .... n_simWoerds time)
similarWords = tree.query(word_embedding, k = n_simWords)
# Iterato thru the first n_simWords
for i in range(n_simWords):
    # Extract the word index
    wordIndex = similarWords[1][i]
    closeWord = list(embeddings_index.keys())[wordIndex]
    print(closeWord)
    ''' 
    This print:
    woman
    girl
    man
    mother
    boy
    child
    herself
    victim
    wife
    she
    '''