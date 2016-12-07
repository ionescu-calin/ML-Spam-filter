import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.dummy import DummyClassifier
def evaluate_classifier_kf(data, kfold, clf):
	scores = []
	confusion = np.array([[0, 0], [0, 0]])	
	for train_indices, test_indices in k_fold:
	    train_text = data.iloc[train_indices]['text'].values
	    train_y = data.iloc[train_indices]['class'].values

	    test_text = data.iloc[test_indices]['text'].values
	    test_y = data.iloc[test_indices]['class'].values

	    features_counts = count_vectorizer.fit_transform(train_text)
	    clf.fit(features_counts, train_y)

	    test_counts = count_vectorizer.transform(test_text)
	    predictions = clf.predict(test_counts)

	    confusion += confusion_matrix(test_y, predictions)
	    score = f1_score(test_y, predictions, pos_label='ham')
	    scores.append(score)
	print('Total emails classified:', len(data))
	print('Score:', sum(scores)/len(scores))
	print('Confusion matrix:')
	print(confusion)


def train_classifier(data, clf_trained):
	train_text = data['text'].values
	features_counts = count_vectorizer.fit_transform(train_text)
	clf_trained.fit(features_counts, data['class'].values)

	dictionay_filepath = "trained_vocab.pkl"
	joblib.dump(count_vectorizer, dictionay_filepath)
	return clf_trained

# File containing data
filename = 'out.csv'

# Load data from csv file
data = pd.DataFrame.from_csv(filename)

# Init count vectorizer
count_vectorizer = CountVectorizer()

# Randomize data indices
data = data.reindex(np.random.permutation(data.index))

# Get the label values
y = data['class'].values

# Initialize kf and skf
k_fold = KFold(n=len(data), n_folds=10)
skf = StratifiedKFold(y, n_folds=10)

# Init classifier Multinomial classifier
clf = MultinomialNB()

# Apply classifier evaluation
evaluate_classifier_kf(data, skf, clf)

# Init Baseline classifier (Random or based on the most likely class distribution)
# strategy="most_frequent" if the class distribution is not balanced
clf = DummyClassifier(strategy="uniform")
evaluate_classifier_kf(data, skf, clf)
# Save classifier to disk 
trained_classifier = train_classifier(data, clf);

# Save classifier to disk
joblib.dump(trained_classifier, 'NB_Trained.pkl', compress=9)