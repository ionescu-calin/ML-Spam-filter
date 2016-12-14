import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
import matplotlib.pyplot as plt

def evaluate_classifier_kf(data, kfold, clf):
	scores = []
	confusion = np.array([[0, 0], [0, 0]])
	acc_per_fold = []
	recall_per_fold = []
	for train_indices, test_indices in k_fold:
	    train_text = data.iloc[train_indices]['text'].values
	    train_y = data.iloc[train_indices]['class'].values

	    test_text = data.iloc[test_indices]['text'].values
	    test_y = data.iloc[test_indices]['class'].values

	    features_counts = count_vectorizer.fit_transform(train_text)
	    # clf.fit(features_counts, train_y)
	    features_counts_tfidf = tfidf.fit_transform(features_counts)
	    # print features_counts_tfidf
	    clf.fit(features_counts, train_y) #sample_weight=features_counts_tfidf)

	    # print features_counts.shape
	    # clf.fit(SelectKBest(chi2, k=93389).fit_transform(features_counts, train_y), train_y)


	    test_counts = count_vectorizer.transform(test_text)
	    test_counts_tfidf = tfidf.fit_transform(test_counts)

	    predictions = clf.predict(test_counts)

	    target_names = ['ham', 'spam']
	    confusion += confusion_matrix(test_y, predictions)
	    # score = f1_score(test_y, predictions, pos_label='ham')
	    # scores.append(score)
	    acc_per_fold.append( accuracy_score(test_y, predictions) )
	    # recall_per_fold.append(recall_score(test_y, predictions, average=None))
	    print(classification_report(test_y, predictions, target_names=target_names))
	print('Total emails classified:', len(data))
	# print('Score:', sum(scores)/len(scores))
	# print ("Accuracy per fold:", acc_per_fold)
	print ("Average accuracy:", np.mean(acc_per_fold))
	# print("Average recall:", np.mean(recall_per_fold))
	print('Confusion matrix:')
	print(confusion)


def train_classifier(data, clf_trained):
	train_text = data['text'].values
	features_counts = count_vectorizer.fit_transform(train_text)
	clf_trained.fit(features_counts, data['class'].values)

	# print features_counts[:,:]
 	dictionay_filepath = "trained_vocab.pkl"
	joblib.dump(count_vectorizer, dictionay_filepath)
	return clf_trained

# File containing data
filename = 'out.csv'

# Load data from csv file
data = pd.DataFrame.from_csv(filename)

# Init count vectorizer
# count_vectorizer = CountVectorizer()

# Init count vectorizer with ngrams	
count_vectorizer = CountVectorizer(lowercase=False) #lowercase=False) #max_df=0.3)# min_df=0.01) #stop_words='english',)

tfidf = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=False)

# token_pattern=r'\b\w+\b'
# Randomize data indices
data = data.reindex(np.random.permutation(data.index))

# Get the label values
y = data['class'].values

# Initialize kf and skf
k_fold = KFold(n=len(data), n_folds=10)
skf = StratifiedKFold(y, n_folds=10)

# Init Baseline classifier (Random or based on the most likely class distribution)
# strategy="most_frequent" if the class distribution is not balanced
clf = DummyClassifier(strategy="uniform")
# evaluate_classifier_kf(data, skf, clf)

# Init classifier Multinomial classifier
clf = MultinomialNB(alpha=0.05)

# Apply classifier evaluation
evaluate_classifier_kf(data, skf, clf)

# Save classifier to disk
trained_classifier = train_classifier(data, clf);

# Save classifier to disk
joblib.dump(trained_classifier, 'NB_Trained.pkl', compress=9)

sample_weight = np.random.RandomState(42).rand(data['class'].values.shape[0])

train_text = data['text'].values
features_counts = count_vectorizer.fit_transform(train_text)

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = \
train_test_split(features_counts, data['class'].values, sample_weight, test_size=0.9, random_state=42)

trained_classifier.fit(X_train, y_train)  
prob_pos_clf = trained_classifier.predict_proba(X_test)[:, 1]

clf_isotonic = CalibratedClassifierCV(trained_classifier, cv=2, method='isotonic')
clf_isotonic.fit(X_train, y_train, sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(trained_classifier, cv=2, method='sigmoid')
clf_sigmoid.fit(X_train, y_train, sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier scores: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sw_test)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sw_test)
print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)


plt.figure()
order = np.lexsort((prob_pos_clf, ))
plt.plot(prob_pos_clf[order], 'r', label='No calibration (%1.3f)' % clf_score)
plt.plot(prob_pos_isotonic[order], 'g', linewidth=3,
         label='Isotonic calibration (%1.3f)' % clf_isotonic_score)
plt.plot(prob_pos_sigmoid[order], 'b', linewidth=3,
         label='Sigmoid calibration (%1.3f)' % clf_sigmoid_score)

plt.ylim([-0.05, 1.05])
plt.xlabel("Instances sorted according to predicted probability "
           "(uncalibrated MNB)")
plt.ylabel("P(y=1)")
plt.legend(loc="upper left")
plt.title("Naive Bayes probabilities")
plt.show()