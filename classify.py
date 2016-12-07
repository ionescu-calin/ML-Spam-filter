from filter import *
from train import* 
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


# Set path to test emails
root = os.path.dirname(os.path.realpath('__file__'))
email_dir = 'test/'
email_dir = os.path.join(root, email_dir)

# Get email name from argument
filename = sys.argv[1]
filename = os.path.join(email_dir, filename)

# Load training vocabulary
#vocab = np.load('trainedvocab.npy')
data = pd.DataFrame.from_csv('out.csv') 

# Load classifier from disk
clf = joblib.load('NB_Trained.pkl')

# Load content of text file
text = extract_words(filename, 'nan')
test_data = build_dataframe_email(filename, text)

# Load the vocabulary
dictionary_filepath = 'trained_vocab.pkl'
vocabulary_to_load = joblib.load(dictionary_filepath)
# loaded_vectorizer = CountVectorizer(vocabulary=vocabulary_to_load)
# loaded_vectorizer._validate_vocabulary()
# print loaded_vectorizer.size
# print('loaded_vectorizer.get_feature_names(): {0}'.
  # format(loaded_vectorizer.get_feature_names()))

# Train the vocabulary
# count_vectorizer = CountVectorizer()
# features_counts = count_vectorizer.fit_transform(data['text'].values)
# print loaded_vectorizer.size
print vocabulary_to_load.vocabulary_
test_counts = vocabulary_to_load.transform(test_data['text'].values)
print test_counts
# print test_counts.size

# Classify new email
predictions = clf.predict(test_counts)
print predictions