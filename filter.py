import os
import re
import email
import pandas as pd
import numpy as np 

#TODO: delete email type from extract words and decode email

#Uses the global dictionaries, is there a better way?
#Adds the words to the dictionary based on the email's type
def populate_dictionary(text, all_text):
	for word in text:
		if re.match("^[A-Za-z]*$", word):
			all_text = all_text + " " + word
   	return all_text

#Recursively goes through the email to count the words
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

# Extract words from file email
def extract_words(filename, email_type):
	f = open(filename, 'r')

	all_text = ""
	raw_email = f.read()
	parsed_email = email.message_from_string(raw_email)

	all_text = decode_email(parsed_email, email_type, all_text)

	return all_text

# Build data frame from directory of emails
def build_dataframe(email_dir):
	files = []
	rows  = [] 
	for main, dirs, files_aux in os.walk(email_dir):
	    for file in files_aux:
	    	filename = os.path.join(email_dir, file);
	        if file.endswith(".txt"):
	        	text=[]
	        	if "spam" not in file:
        			text = extract_words(filename, "ham")
        			if text:
        				rows.append({'text': text, 'class': 'ham'})
        		else:
        			text = extract_words(filename, "spam")
        			if text:
        				rows.append({'text': text, 'class' : 'spam'})
	        	if text:
	        		files.append(filename)
	dataframe = pd.DataFrame(rows, index=files)
	return dataframe

# Get current path to directory
root = os.path.dirname(os.path.realpath('__file__'))
email_dir = 'public/'
# Set path to test emails
email_dir = os.path.join(root, email_dir)

# Build the dataframe and write it to a file	
dataframe = build_dataframe(email_dir)
dataframe.to_csv('out.csv', sep=',')