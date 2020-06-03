
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
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter("ignore")


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



# Binary Relevance models training
print()
print('Training Binary Relevance models...')
pred_BR = []
for e,(data,target) in enumerate(ml.BinaryRelevance(X_train,y_train)):
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression())
    clf.fit(data,target)
    prediction = clf.predict(X_test)
    pred_BR.append(clf.predict(X_test))
pred_BR = np.transpose(pred_BR)
print('Done!!!')
print()



# Calibrated Label Ranking models training
print('Training Calibrated Label Ranking models...')
preds = []
for e,(data,target) in enumerate(ml.CalibratedLabelRanking(X_train,y_train)):
    if e==0:
        preds.append([1 for i in range(len(y_test))])
        continue
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression())
    clf.fit(data,target)
    prediction = clf.predict(X_test)
    preds.append(prediction)
preds = np.transpose(preds)
print('Done!!!')
print()

pred_CLR = []
for pred in preds:
    pred_CLR.append([1 if list(pred).count(i+1)>list(pred).count(0) else 0 for i in range(6)])
pred_CLR = np.array(pred_CLR)



# Print classification results
print('// Accuracy of multi-labeled approaches //')

print()
print('\t-Binary Relevance-')
print()
print('Accuracy per example:',accuracy_score(pred_BR,y_test))
print('Accuracy per label')
for i in range(6):
    label_acc = accuracy_score(pred_BR[:,i],y_test[:,i])
    print(label_names[i],"->",label_acc)
    
print()  
print('\t-Calibrated Label Ranking-')
print()
print('Accuracy per example:',accuracy_score(pred_CLR,y_test))
print('Accuracy per label')
for i in range(6):
    label_acc = accuracy_score(pred_CLR[:,i],y_test[:,i])
    print(label_names[i],"->",label_acc)