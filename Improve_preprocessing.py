#TODO: look at NLTK for python (from nltk.corpus import stopwords) for better preprocessing
# It can do 3 an 4 from the following list:
""" Nice paper: http://www.ijste.org/articles/IJSTEV1I11008.pdf 
The basic data pre-processing steps of spam detection are:
1) The words having length <=2 are removed.         
2) All the special characters are removed.
3) Stop words are removed.
4) Porter’s Stemming Algorithm is applied to bring the word in their most basic form.
5) The word frequency of all the words.
6) The normalized term frequency of all the words.
7) The inverse document frequency of all the words.
8) Term Document Frequency inverse document frequency (TF-IDF)

TODO: to remove stopwords: filtered_words = [word for word in word_list if word not in stopwords.words('english')]
TODO: to stem words: similar to stopwords: 
	wordnet_lemmatizer = WordNetLemmatizer()
	stemmed_words = [wordnet_lemmatizer.lemmatize(word) for word in word_list] #not sure if it works p

Counting the number of alpha numeric words in subject line or in the entire email might be helpful 
as spam’s are reported to contain large number of alpha numeric words. So a single feature containing 
the number of alpha numeric words in an email might be helpful.
"""
