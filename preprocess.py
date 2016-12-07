import os, sys
import re
import email
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
#stemmer = SnowballStemmer("english")
stemmer = PorterStemmer()
#lmtzr = WordNetLemmatizer()

#Adds the words to the dictionary based on the email's type
def populate_dictionary(text, all_text):
	for word in text:
		if re.match("^[A-Za-z]*$", word):
			if(word not in stop): 
				#word = lmtzr.lemmatize(word)
				word = stemmer.stem(word)
				all_text = all_text + " " + word
   	return all_text

#Recursively goes through the email to count the words
def decode_email(parsed_email, all_text):
	if parsed_email.is_multipart():
	    for part in parsed_email.get_payload():
	    	if part.is_multipart():
	    		 for part2 in parsed_email.get_payload():
	    		 	return decode_email(part2, all_text)
	    	else:
		        return decode_email(part, all_text)
	else:
		return populate_dictionary(parsed_email.get_payload().split(), all_text)

# Extract words from file email
def extract_words(filename):
	f = open(filename, 'r')

	all_text = ""
	raw_email = f.read()
	parsed_email = email.message_from_string(raw_email)

	all_text = decode_email(parsed_email, all_text)

	return all_text

# Build data frame from directory of emails
def build_dataframe_dir(email_dir):
	files = []
	rows  = [] 
	for main, dirs, files_aux in os.walk(email_dir):
	    for file in files_aux:
	    	filename = os.path.join(email_dir, file);
	        if file.endswith(".txt"):
	        	text = []
	        	if "spam" not in file:
        			text = extract_words(filename)
        			if text:
        				rows.append({'text': text, 'class': 'ham'})
        		else:
        			text = extract_words(filename)
        			if text:
        				rows.append({'text': text, 'class' : 'spam'})
	        	if text:
	        		files.append(filename)
	dataframe = pd.DataFrame(rows, index=files)
	return dataframe

# Build data frame from an email
def build_dataframe_email(filename, extracted_words):
	rows = []
	files = []
	text = extracted_words
	if 'spam' not in filename:
		if text:
			rows.append({'text': text, 'class': 'ham'})
	else:
		if text:
			rows.append({'text': text, 'class' : 'spam'})
	if text:
		files.append(filename)
	dataframe = pd.DataFrame(rows, index=files)
	return dataframe

def main(): 
	# Get current path to directory
	root = os.path.dirname(os.path.realpath('__file__'))
	email_dir = 'public/'
	# Set path to test emails
	email_dir = os.path.join(root, email_dir)

	# Build the dataframe and write it to a file	
	dataframe = build_dataframe_dir(email_dir)
	dataframe.to_csv('out.csv', sep=',')

if(len(sys.argv) == 2):
	if(sys.argv[1] == "-r"):
		main()