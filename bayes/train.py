import re

from nltk import SnowballStemmer, word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

# Constants
GOOD_FILE = '../corpus/{0}/good.txt'
BAIT_FILE = '../corpus/{0}/bait.txt'
NAME = 'classifier'

# Token cleaning stuff
stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

re_nonword = re.compile('^\W*$')

def validate_token(token):
    return not (re_nonword.match(token) or token in stopwords)

# Feature stuff
def get_words(titles):
    all_words = []
    for words in titles:
      all_words.extend(words[0])
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# Title loading
train_titles = []
with open(GOOD_FILE.format('train'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        train_titles.append(([stemmed_tokens, 'Good']))
        
with open(BAIT_FILE.format('train'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        train_titles.append(([stemmed_tokens, 'Bait']))

test_titles = []
with open(GOOD_FILE.format('test'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        test_titles.append(([stemmed_tokens, 'Good']))
        
with open(BAIT_FILE.format('test'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        title = line.replace('\n', '')
        tokens = [t.lower() for t in word_tokenize(title) if validate_token(t.lower())]
        stemmed_tokens = [stemmer.stem(t) for t in tokens]
        test_titles.append(([stemmed_tokens, 'Bait']))

random.shuffle(train_titles)
random.shuffle(test_titles)

print('Size of sets:')
print('Train: {0}'.format(len(train_titles)))
print('Test: {0}'.format(len(test_titles)))
print()

# Classifier training
trainer = NaiveBayesClassifier.train
sentim_analyzer = SentimentAnalyzer()

word_features = get_word_features(get_words(train_titles))

train_set = nltk.classify.apply_features(extract_features, train_titles)
test_set = nltk.classify.apply_features(extract_features, test_titles)

classifier = sentim_analyzer.train(trainer, train_set)

for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

# Storing classifier and words
with open('{0}.pickle'.format(NAME), 'wb') as f:
    pickle.dump(classifier, f)

with open('{0}.words'.format(NAME), 'w') as f:
    for word in word_features:
        f.write(word + '\n')