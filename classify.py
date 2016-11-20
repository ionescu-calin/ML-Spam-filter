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

print test_data
#Go through all of the emails in the directory and extract their words
for main, dirs, files in os.walk(email_dir):
    for file in files:
    	filename = os.path.join(email_dir, file);
        if file.endswith(".txt"):
        	test_email_frequency = defaultdict( int )
        	extract_words(filename, "test")
        	test = pd.Series(test_email_frequency).filter(test_data)
        	print test_data
