import sys, os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from preprocess import build_dataframe_email, extract_words

# Get email name from argument
filename = sys.argv[1]

# Load classifier from disk
clf = joblib.load('NB_Trained.pkl')

# Load content of text file
text = extract_words(filename)
test_data = build_dataframe_email(filename, text)

# Load the vocabulary
dictionary_filepath = 'trained_vocab.pkl'
count_vectorizer = joblib.load(dictionary_filepath)

if text:
	test_counts = count_vectorizer.transform(test_data['text'].values)

	# Classify new email
	predictions = clf.predict(test_counts)
	print predictions[0]
else:
	print "ham"
