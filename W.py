

import math
import random, string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.model_selection import train_test_split
import re
import tweepy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, naive_bayes, svm
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score,recall_score,f1_score
porter_stemmer = PorterStemmer()

import time
start_time = time.time()
"""def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else :
        return wn.NOUN
    # for easy if-statement 
def lemma(str_input):
    lem= WordNetLemmatizer()
    words = re.sub(r"http\S+[^A-Za-z0-9^,!.\/'+-=]", " ", str_input).lower().split()
    return [lem.lemmatize(w, pos=get_wordnet_pos(tag)) for w, tag in pos_tag(words) ]
def stemm(str_input):
    ps = PorterStemmer()
    words = re.sub(r"http\S+[^A-Za-z0-9\-]", " ", str_input).lower().split()
    return [ps.stem(w) for w in words]"""
def boring(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    return words

df = pd.read_csv(r"C:\Users\User\Desktop\W\IMDB1.csv", encoding="latin-1")

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df.text, df.label, test_size=0.2)
"""vect = CountVectorizer(max_features=1000, binary=True,stop_words='english',tokenizer=boring_tokenizer)
X_train_vect = vect.fit_transform(X_train)"""


tfidf=TfidfVectorizer(max_df=0.90,tokenizer=boring,stop_words=stopwords.words('english'), min_df = 5)


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Train_XTF = tfidf.fit_transform(Train_X)
Test_XTF = tfidf.transform(Test_X)

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_XTF,Train_Y)

NB = Naive.predict(Test_XTF)

tweetp = []
consumer_key = "plLPvtsfC4TXcsfQ0KYGS6KML" 
consumer_secret = "fMY52Iy1mjxYMt30iG09zJp4bLGI4JMNWOCXPvEGYs3oMLEFWZ" 
access_token="1475232163855749120-sIn6fQwDKXbxSCS0Sqfhm6k53ApDdd"
access_token_secret="D8YrIRy64jAJ8lFFb5ERxMcdIEMswaenkeBGByCxFp3fU"
        
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
search_words = "trump" + " -filter:retweets"
date_since = "2025-10-17"
api = tweepy.API(auth, wait_on_rate_limit=True)
tweets = tweepy.Cursor(api.search,
                      q=search_words,
                      lang="en",
                      since=date_since).items(10)
k=[]
for tweet in tweets:
        n=[(tweet.text)]
        d=n, Naive.predict(tfidf.transform(n))
        """k.append(', '.join(map(str,d)))"""
        k.append(d)

n=pd.DataFrame(k, columns=['text', 'value'])
n['value']=n['value'].str.get(0).astype(int)

n['text']=n['text'].str.get(0).astype(str)
print(n.to_csv(columns=['text', 'value'], sep=',', index=False))
"""print(n.dtypes)
print(n)"""
counts = df.label.value_counts()
print(counts)
print(confusion_matrix(NB,Test_Y))
print("Naive Accuracy Score -> ",accuracy_score(NB, Test_Y)*100)
print("NB Precision Score -> ",precision_score(NB, Test_Y)*100)
print("NB Recall Score -> ",recall_score(NB, Test_Y)*100)
print("NB f1 Score -> ",f1_score(NB, Test_Y)*100)
print("--- %s seconds ---" % (time.time() - start_time))
print(Train_XTF.shape)


