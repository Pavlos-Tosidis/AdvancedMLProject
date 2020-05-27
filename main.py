
import multilabel as ml
# Libraries for data loading and preproccessing.
import re
import nltk 
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import iterative_train_test_split
# Libraries for classification model.
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC



# Load the data.
myData = pd.read_csv('train.csv')

X = myData.iloc[:,1].values
y = myData.iloc[:,2:].values
label_names = ["toxic","severe_toxic","obscene", "threat","insult","identity_hate"]


# Preprocessing of Wikipedia edits.
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
 
for e,s in enumerate(X):
    # Make all letters lowercase.
    s = s.lower()
    # Remove digits.
    s = pt = re.sub(r'\d+', '', s)
    # Remove special characters.
    s  = s.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace.
    s = " ".join(s.split())
    # Tokenize and remove stopwords.
    word_tokens = word_tokenize(s) 
    s = [word for word in word_tokens if word not in stop_words] 
    # Lemmatization.
    s = [lemmatizer.lemmatize(word, pos ='v') for word in s]
    X[e] = s
    

# Transforming data using tf-idf.
def identity_tokenizer(text):
    return text
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)    
X = tfidf.fit_transform(X)




# Split the data to train and test set.
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size = 0.2)

# Define a classification pipeline.
clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC())



# Binary Relevance models training
print('Training Binary Relevance models...')
pred_BR = []
for e,(data,target) in enumerate(ml.BinaryRelevance(X_train,y_train)):
    clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC())
    clf.fit(data,target)
    pred_BR.append(clf.predict(X_test))
pred_BR = np.transpose(pred_BR)
print('Done!!!')
print()


# Calibrated Label Ranking models training
print('Training Calibrated Label Ranking models...')
pred_CLR = []
for e,(data,target) in enumerate(ml.CalibratedLabelRanking(X_train,y_train)):
    if e==0:
        pred_CLR.append([1 for i in range(len(y_test))])
        continue
    clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC())
    clf.fit(data,target)
    pred_CLR.append(clf.predict(X_test))
pred_CLR = np.transpose(pred_CLR)
print('Done!!!')
print()

preds = []
for pred in pred_CLR:
    preds.append([1 if list(pred).count(i+1)>list(pred).count(0) else 0 for i in range(6)])



print('// Accuracy of multi-labeled approaches //')
print()
print('Binary Relevance Accuracy:',accuracy_score(pred_BR,y_test))
print('Calibrated Label Ranking Accuracy:',accuracy_score(np.array(preds),y_test))

