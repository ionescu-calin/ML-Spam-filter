import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# File containing data
filename = 'out.csv'

# Load data from csv file
data = pd.DataFrame.from_csv(filename)

# Init classifier
clf = MultinomialNB()

# Get values, labeles and train classifier
targets = data.index.values
counts  = data.values
clf.fit(counts, targets)

# Predict result on training values
predictions = clf.predict(counts)