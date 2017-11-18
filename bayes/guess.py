import re
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import *

# Constants
NAME = 'classifier'

# NLTK stuff
stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

# title cleaning
re_nonword = re.compile('^\W*$')

def validate_token(token):
    return not (re_nonword.match(token) or token in stopwords)

def clean_title(title):
    title = title.replace('\n', '')
    tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
    return [stemmer.stem(t) for t in tokens]

# Classifier stuff
with open('{0}.words'.format(NAME), 'r') as f:
    word_features = list(map(lambda x: x.replace('\n', ''), f.readlines()))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def guess(tokens):
    return classifier.prob_classify(extract_features(tokens))

try:
    with open('{0}.pickle'.format(NAME), 'rb') as f:
        classifier = pickle.load(f)
except FileNotFoundError:
    print('Classifier not found. Exiting...')
    exit()

# Run
title = input('Insert a title to verify: ')
tokens = clean_title(title)
print('Cleaned and tokenized title: {0}\n'.format(tokens))
guess = guess(tokens)
print('Result:')
for s in guess.samples():
    print('{0}: {1}'.format(s, guess.prob(s)))
