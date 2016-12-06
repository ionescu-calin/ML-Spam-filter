from filter import * 
import pandas as pd
import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


root = os.path.dirname(os.path.realpath('__file__'))
email_dir = 'test/'
# Set path to test emails
email_dir = os.path.join(root, email_dir)
filename = sys.argv[1]
filename = os.path.join(email_dir, filename)


vocab = np.load('vocab.npy')

#Load classifier from disk
clf = joblib.load('NB_Trained.pkl')

text = extract_words(filename, 'nan')

count_vectorizer = CountVectorizer()
features_counts = count_vectorizer.fit_transform(vocab)

test_counts = count_vectorizer.transform(text)
print count_vectorizer
predictions = clf.predict(test_counts)

#print predictions
