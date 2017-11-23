import re
import redis

from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords

# Constants
HOST = '172.17.0.2'

# Token cleaning stuff
stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

re_nonword = re.compile('^\W*$')

def validate_token(token):
    return not (re_nonword.match(token) or token in stopwords)

def clean_title(title):
    title = title.replace('\n', '')
    tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
    return [stemmer.stem(t) for t in tokens]

# Redis stuff
session = redis.StrictRedis(host=HOST, port=6379, db=0)

# Classifier stuff
def similarity_index(tokens, label):
    if len(tokens) == 0:
        return 0
    res = 0
    for t in tokens:
        query = session.get('{0}:{1}'.format(label, t))
        res += float(query) if query else 0
    return res / len(tokens)

def guess(tokens):
    good_index = similarity_index(tokens, 'Good')
    bait_index = similarity_index(tokens, 'Bait')
    sum_index = good_index + bait_index
    return {
        'Good': good_index / sum_index if sum_index > 0 else good_index,
        'Bait': bait_index / sum_index if sum_index > 0 else bait_index
        }

# Run
title = input('Insert a title to verify: ')
tokens = clean_title(title)
print('Cleaned and tokenized title: {0}\n'.format(tokens))
guess = guess(tokens)
print('Result:')
for k, v in guess.items():
    print('{0}: {1}'.format(k, v))
