import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english')) 

def readFile(dataName):
	train  = pd.read_csv(dataName)
	train['text'] =  train['text'].str.replace("[^a-zA-Z*]", " ") #remove punctuations, special characters and numbers except for *
	train['text'] =  train['text'].str.lower() #to lower case
	#[textID, text, label]
	data = np.array
	if dataName == "train.csv":
		data = np.concatenate((np.array(train.to_numpy()[:,0:2]),train.to_numpy()[:,3:4]), axis=1 )
	else:
		data = train.to_numpy()
	
	return data #train['text'], train['sentiment']

def tokenize(data):
	#data = data.str.replace("[^a-zA-Z*]", " ") #remove punctuations, special characters and numbers except for *
	#data = data.str.lower() #to lower case
	stemmer = PorterStemmer()
	word_tokens = []
	cleanedData = []
	#neutralStr = ""
	#negativeStr = ""
	#positiveStr = ""
	for i in range(len(data)):
		if type(data[i][1]) == float :
			continue
		
		splitData = []
		for word in data[i][1].split():
			if word not in stop_words and len(word) > 2: #remove stop words and short words
				word = stemmer.stem(word)
				splitData.append(word)

		joinData =  ' '.join(splitData)
		'''
		if data[i][2] == "neutral":
			neutralStr += joinData
		elif data[i][2] == "positive":
			positiveStr += joinData
		else:
			negativeStr += joinData
		'''
		#[textID, processedText, label]
		word_tokens.append([data[i][0] ,joinData, data[i][2]])
		cleanedData.append(joinData)

	
	return word_tokens, cleanedData
	
	
	
def extractFeatures(corpus):

	#tfidf_vectorizer = TfidfVectorizer()
	#tfidf = tfidf_vectorizer.fit_transform(corpus)
	#print(tfidf.toarray())
	bow_vectorizer = CountVectorizer()
	bow = bow_vectorizer.fit_transform(corpus)
	print(bow_vectorizer.get_feature_names())
	print(bow.toarray())
	return bow.toarray()


		
		




	#print(word_tokens)
def main():
   data = readFile("train.csv")
   word_tokens, corpus = tokenize(data)
   dataArray = extractFeatures(corpus)

   testData = readFile('test.csv')
   word_tokens_test, corpus_test = tokenize(testData)
   testArray = extractFeatures(corpus_test)
   
if __name__ == "__main__":
    main()