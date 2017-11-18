import random
import re
import redis

from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords

# Constants
GOOD_FILE = '../corpus/{0}/good.txt'
BAIT_FILE = '../corpus/{0}/bait.txt'
HOST = '172.17.0.2'

# Token cleaning stuff
stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

re_nonword = re.compile('^\W*$')

def validate_token(token):
    return not (re_nonword.match(token) or token in stopwords)

# Redis stuff
session = redis.StrictRedis(host=HOST, port=6379, db=0)

# Title loading
train_titles = []
with open(GOOD_FILE.format('train'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        train_titles.append((stemmed_tokens, 'Good'))
        
with open(BAIT_FILE.format('train'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        train_titles.append((stemmed_tokens, 'Bait'))

test_titles = []
with open(GOOD_FILE.format('test'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        test_titles.append((stemmed_tokens, 'Good'))
        
with open(BAIT_FILE.format('test'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        test_titles.append((stemmed_tokens, 'Bait'))

random.shuffle(train_titles)
random.shuffle(test_titles)

print('Size of sets:')
print('Train: {0}'.format(len(train_titles)))
print('Test: {0}'.format(len(test_titles)))
print()

# Redis training
i = 0
max_words = {
    'Bait': ('', 0),
    'Good': ('', 0)
}
print('Training classifier...')
print('Inserted titles: 0')
for tokens, label in train_titles:
    i += 1
    for t in tokens:
        try:
            session.incr("{0}:{1}".format(label, t))
        except:
            print(t)
        val = int(session.get("{0}:{1}".format(label, t)))
        if val > max_words[label][1]:
            max_words[label] = (t, val)
    if not i % 500:
        print(str(i))

for label, word in max_words.items():
    keys = session.keys("{0}:*".format(label))
    for k in keys:
        v = session.get(k)
        session.set(k, int(v) / word[1])
print('Total inserted: {0}\n'.format(i))

# Cross validation
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

i = 0
print('Testing classifier...')
print('Tested titles: 0')
tp = 0
fp = 0
tn = 0
fn = 0
for (tokens, label) in test_titles:
    i += 1
    results = guess(tokens)
    if label == 'Good':
        if results['Good'] > results['Bait']:
            tp += 1
        else:
            fn += 1
    else:
        if results['Bait'] > results['Good']:
            tn += 1
        else:
            fp += 1

    if not i % 500:
        print(i)
print('Total tested: {0}\n'.format(i))

print('Precision: {0}'.format(tp / (tp + fp)))
print('Accuracy: {0}'.format((tp + tn) / (tp + tn + fp + fn)))
print('Recall: {0}'.format(tp / (tp + fn)))