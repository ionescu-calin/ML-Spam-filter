import os
import math
import re
import email
import glob
from collections import defaultdict 
import pandas as pd

#Python dictionaries to hold word frequency
spam_word_frequency = defaultdict( int )
ham_word_frequency = defaultdict( int )

#Uses the global dictionaries, is there a better way?
#Adds the words to the dictionary based on the email's type
def populate_dictionary(text, text_type):
    for word in text:
    	if re.match("^[A-Za-z]*$", word):
    		if text_type == "spam":
    			spam_word_frequency[word] += 1
    		else: 
    			ham_word_frequency[word] += 1

#Recursively goes through the emails to count the words
def decode_email(parsed_email, email_type):
	if parsed_email.is_multipart():
	    for part in parsed_email.get_payload():
	    	if part.is_multipart():
	    		 for part2 in parsed_email.get_payload():
	    		 	decode_email(part2, email_type)
	    	else:
		        decode_email(part, email_type)
	else:
	    populate_dictionary(parsed_email.get_payload().split(), email_type)

def extract_words(filename, email_type):
	print 'Analysing ' + filename
	f = open(filename, 'r')

	raw_email = f.read()
	parsed_email = email.message_from_string(raw_email)

	#How to access the email fields in case we need this later
	#print parsed_email.get('From')
	#print parsed_email.get('To')
	#print parsed_email.get('Subject')

	decode_email(parsed_email, email_type)

	return


#Get current path
root = os.path.dirname(os.path.realpath('__file__'))

email_dir = 'public/'
#Set path to test emails
email_dir = filename = os.path.join(root, email_dir)

#Go through all of the emails in the directory and extract their words
for root, dirs, files in os.walk(email_dir):
    for file in files:
    	filename = os.path.join(email_dir, file);
        if file.endswith(".txt"):
        	if "spam" not in file:
        		extract_words(filename, "ham")
        	else:
        		extract_words(filename, "spam")

#Merge the two frequency arrays into one pandas DataFrame so it can be used with scikit learn
ham = pd.Series(ham_word_frequency)
spam = pd.Series(spam_word_frequency)

conc = pd.concat([ham, spam], axis=1, keys=['ham','spam'])
conc = pd.DataFrame(conc)

#Apply Laplace correction
#Incredibly slow, replace with something else
for i in range(len(conc.ix[:,1])):
	if math.isnan(conc.ix[i,1]):
		conc.ix[i,1] = 1 
	else:
		conc.ix[i,1] += 1
	if math.isnan(conc.ix[i,0]):
		conc.ix[i,0] = 1 
	else:
		conc.ix[i,0] += 1

print conc

