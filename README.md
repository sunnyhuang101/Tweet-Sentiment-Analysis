# Tweet Sentiment Analysis
Built a Long Short-Term Memory (LSTM) based model to predict the overall sentiment (positive, negative, or neural) in tweet dataset. Used a training dataset with 27471 tweets and testing dataset with 3534 tweets from Kaggle. Achieved training accuracy up to 88%.<br>

#### Data Preprocessing
a. Remove all rows containing null values.<br>
b. Remove all punctuations and special symbols except those that are meaningful for sentiment
extraction (For example: **** in the sentence of "Son of ****" helps us predict the sentiment as
negative.). As a result, we can obtain more precise training result.<br>
c. Convert all characters to lower case so the same letter with different cases will be seen as the same.<br>
d. Tokenize every tweet into separate word using the tokenizer in the deep learning library Keras.<br>
e. Convert distinct words and labels into numeric values (for example: ‘son’, ‘you’, ‘whether’ are
converted to number 1, 2,3). Note: we combine the distinct tokenized words in training set and test set before converting them to numbers in order to make sure these 2 data sets have same dimensions of tokenized text.

