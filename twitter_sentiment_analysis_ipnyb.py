# -*- coding: utf-8 -*-
"""Twitter Sentiment Analysis.ipnyb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DkqNvOnh-Jn1EgXGxjVRUf8PW1KJb1lK
"""

!pip install kaggle

"""**Uploading kaggle.json file**"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

"""**Importing dataset of twitter APi's from kaggle using APi**"""

!kaggle datasets download -d kazanova/sentiment140

import zipfile
dataset = "/content/sentiment140.zip"

with zipfile.ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

"""**Importing Libraries**"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords  #nltk means natural language
from nltk.stem.porter import PorterStemmer  # Reduce word to its root words
from sklearn.feature_extraction.text import TfidfVectorizer #converting textual data in to numerical data (TDIF)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords')

"""Stopwords are the words that if we remove them from textual data then they have no effect"""

print(stopwords.words('english'))

"""**Data Preprocessing**"""

twitter_dataset = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')

#checking no. of rows and col
twitter_dataset.shape

twitter_dataset.head()

columns_name = ['target','ids','date','flag','user','text']
twitter_dataset.columns = columns_name

twitter_dataset = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',names=columns_name, encoding='ISO-8859-1')

twitter_dataset.head()

twitter_dataset.shape

twitter_dataset.isnull().sum()

twitter_dataset['target'].value_counts()

"""0--- Negative tweets, 1--- Positive tweets


"""

twitter_dataset.replace({'target':{4:1}},inplace=True)

twitter_dataset['target'].value_counts()

"""**Stemming**
It's a process that reduce word to its root word e.g actor,acting, actress to **act**
"""

port_stem = PorterStemmer()

# we are passng text column here
# in this we just build a func of stemming to apply in further code

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', content) # ^ this is used to remove
  stemmed_content = stemmed_content.lower() # all the letters are converted in lower letters
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

twitter_dataset['stemmed_content'] = twitter_dataset['text'].apply(stemming)

twitter_dataset.head()

print(twitter_dataset['stemmed_content'])

print (twitter_dataset['target'])

X = twitter_dataset['stemmed_content'].values
Y = twitter_dataset['target'].values

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

print (X_train)

print(X_test)

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train)

print(X_test)

"""**Training of model in Logistic Regression**"""

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

"""**Model Evaluation**"""

X_train_Prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_Prediction, Y_train)

print('Accuracy of training data is = ', training_data_accuracy)

X_test_Prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_Prediction, Y_test)

print('Accuracy of testing data = ', test_data_accuracy)

"""**Saving the Trained Model **"""

import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open('/content/trained_model.sav', 'rb'))

X_new = X_test[0]
print(Y_test[0])

prediction = loaded_model.predict(X_new)
print(prediction)
prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Negative')
else:
  print('The news is Positive')

X_new = X_test[3]
print(Y_test[3])

prediction = loaded_model.predict(X_new)
print(prediction)
prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Negative')
else:
  print('The news is Positive')