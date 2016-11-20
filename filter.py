import os
import math
import re
import email
import glob
from collections import defaultdict 
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

#Python dictionaries to hold word frequency
spam_word_frequency = defaultdict( int )
ham_word_frequency = defaultdict( int )
test_email_frequency = defaultdict( int )

#Uses the global dictionaries, is there a better way?
#Adds the words to the dictionary based on the email's type
def populate_dictionary(text, all_text):
	for word in text:
		if re.match("^[A-Za-z]*$", word):
			all_text = all_text + " " + word
   	return all_text

#Recursively goes through the emails to count the words
def decode_email(parsed_email, email_type, all_text):
	if parsed_email.is_multipart():
	    for part in parsed_email.get_payload():
	    	if part.is_multipart():
	    		 for part2 in parsed_email.get_payload():
	    		 	return decode_email(part2, email_type, all_text)
	    	else:
		        return decode_email(part, email_type, all_text)
	else:
		return populate_dictionary(parsed_email.get_payload().split(), all_text)

def extract_words(filename, email_type, all_text):
	print 'Analysing ' + filename
	#print all_text
	f = open(filename, 'r')

	raw_email = f.read()
	parsed_email = email.message_from_string(raw_email)

	all_text = decode_email(parsed_email, email_type, all_text)

	return all_text

def words_from_files(files):
	count_vectorizer = CountVectorizer()
	all_text = ""
	for file in files:
		all_text = extract_words(file, 'spam', all_text)

	counts = count_vectorizer.fit_transform(all_text.split())
	print counts
	print 'DONE'

#Get current path
root = os.path.dirname(os.path.realpath('__file__'))
email_dir = 'public/'
#Set path to test emails
email_dir = os.path.join(root, email_dir)

files = []
for main, dirs, files_aux in os.walk(email_dir):
    for file in files_aux:
    	filename = os.path.join(email_dir, file);
        if file.endswith(".txt"):
        	files.append(filename)

print files

words_from_files(files);
