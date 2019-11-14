#import packages needed
import skipthoughts
import csv
import random
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# define skip-thoughts vectorizer class for scikit-learn
class SkipThoughtsVectorizer(object):
    def __init__(self, **kwargs):
        self.model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(self.model)

    def fit_transform(self, raw_documents, y):
        return self.encoder.encode(raw_documents, verbose=False)

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents, copy=True):
        return self.fit_transform(raw_documents, None)


# Load training data
f = open('train.csv')
train_rows = [row for row in csv.reader(f)][1:]  # discard the first row
random.shuffle(train_rows)
try:
    tweets_train = [row[0].decode('utf8') for row in train_rows]
except AttributeError:  # it's python 3
    tweets_train = [row[0] for row in train_rows]
classes_train = [row[1] for row in train_rows]

# Load testing data
f = open('test.csv')
test_rows = [row for row in csv.reader(f)][1:]  # discard the first row
try:
    tweets_test = [row[0].decode('utf8') for row in test_rows]
except AttributeError:  # it's python 3
    tweets_test = [row[0] for row in test_rows]
classes_test = [row[1] for row in test_rows]

# Define pipelines for skip-thougts and tf-idf
pipeline_skipthought = Pipeline(steps=[('vectorizer', SkipThoughtsVectorizer()),
                        ('classifier', LogisticRegression())])
pipeline_tfidf = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
                        ('classifier', LogisticRegression())])

feature_union = ('feature_union', FeatureUnion([
    ('skipthought', SkipThoughtsVectorizer()),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
]))
pipeline_both = Pipeline(steps=[feature_union,
                        ('classifier', LogisticRegression())])

# Train and test the models
for train_size in (20, 50, 100, 200, 500, 1000, 2000, 3000, len(tweets_train)):
    print(train_size, '--------------------------------------')
    # skipthought
    pipeline_skipthought.fit(tweets_train[:train_size], classes_train[:train_size])
    print ('skipthought', pipeline_skipthought.score(tweets_test, classes_test))

    # tfidf
    pipeline_tfidf.fit(tweets_train[:train_size], classes_train[:train_size])
    print('tfidf', pipeline_tfidf.score(tweets_test, classes_test))

    # both
    pipeline_both.fit(tweets_train[:train_size], classes_train[:train_size])
    print('skipthought+tfidf', pipeline_both.score(tweets_test, classes_test))
