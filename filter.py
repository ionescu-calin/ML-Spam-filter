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
if text:
	test_counts = vocabulary_to_load.transform(test_data['text'].values)
	# print test_counts.size

	# Classify new email
	predictions = clf.predict(test_counts)
	print predictions[0]
else:
	print "ham"