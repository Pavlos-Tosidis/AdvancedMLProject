import multilabel as ml
import classimbalance

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
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import iterative_train_test_split
# Libraries for classification model.
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.utils.class_weight import compute_class_weight
from costcla.metrics import cost_loss
from costcla.models import BayesMinimumRiskClassifier
from sklearn.calibration import CalibratedClassifierCV
warnings.simplefilter("ignore")

# Load the data.
myData = pd.read_csv('train.csv')

X = myData.iloc[:, 1].values
y = myData.iloc[:, 2:].values
cost_list = [3, 4, 2, 6, 5, 7]
label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# Preprocessing of Wikipedia edits.
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

for e, s in enumerate(X):
    # Make all letters lowercase.
    s = s.lower()
    # Remove digits.
    s = pt = re.sub(r'\d+', '', s)
    # Remove special characters.
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace.
    s = " ".join(s.split())
    # Tokenize and remove stopwords.
    word_tokens = word_tokenize(s)
    s = [word for word in word_tokens if word not in stop_words]
    # Lemmatization.
    s = [lemmatizer.lemmatize(word, pos='v') for word in s]
    X[e] = s


# Transforming data using tf-idf.
def identity_tokenizer(text):
    return text


tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
X = tfidf.fit_transform(X)

# Split the data to train and test set.
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

# Classifier used throughout the methodologies
# LogReg showed decent results, and fast runs.
clf = LogisticRegression(C=12.0)
# clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True))



#############
# Multilabe methods (ml)
# 1: Binary Relevance method
# 2: Calibrated Label Ranking method
mlb = 1
# class imbalance method (cim)
# 1: no class imbalance method applied
# 2: random undersampler
# 3: SMOTE
cim = 3
# Cost Minimization method (cm)
# 1 : Probability Calibration using the isotronic method
# 2 : Probability Calibration using CostCla built-in function
# 3 : Class_Weighting using build in 'balanced' mode from sk_learn
#  Method 3 takes too long to complete on the full dataset, change to myData.iloc[:50000, 1].values to run on smaller set
cm = 1
############
# Binary Relevance models training
print()
print('Training Binary Relevance models...')
pred_BR = []
cost_BR = []

if mlb == 1:
    for e, (data, target) in enumerate(ml.BinaryRelevance(X_train, y_train)):
        cost = cost_list[e]
        fp = np.full((y_test.shape[0], 1), 1)
        fn = np.full((y_test.shape[0], 1), cost)
        tp = np.zeros((y_test.shape[0], 1))
        tn = np.zeros((y_test.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))
        if cim == 2:
            data, target = classimbalance.random_undersampler(data, target)
        elif cim == 3:
            data, target = classimbalance.smote(data, target)
            
        if cm == 1:
            # Probability calibration using Isotonic Method
            cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            model = cc.fit(data, target)
            prob_test = model.predict_proba(X_test)
            bmr = BayesMinimumRiskClassifier(calibration=False)
            prediction = bmr.predict(prob_test, cost_matrix)
            loss = cost_loss(y_test[:, e], prediction, cost_matrix)
            pred_BR.append(prediction)
            cost_BR.append(loss)
            
        elif cm == 2:
            # Probability calibration using CostCla calibration            
            model = clf.fit(data, target)
            prob_train = model.predict_proba(data)
            bmr = BayesMinimumRiskClassifier(calibration=True)
            bmr.fit(target, prob_train)
            prob_test = model.predict_proba(X_test)
            prediction = bmr.predict(prob_test, cost_matrix)
            loss = cost_loss(y_test[:, e], prediction, cost_matrix)
            pred_BR.append(prediction)
            cost_BR.append(loss)

        elif cm == 3:
            # Cost minimization using class weighting
            clf = LogisticRegression(C=12.0, class_weight={0: 1, 1: cost})
            model = clf.fit(data, target)
            prediction = model.predict(X_test)
            loss = cost_loss(y_test[:, e], prediction, cost_matrix)
            pred_BR.append(prediction)
            cost_BR.append(loss)

    pred_BR = np.transpose(pred_BR)

elif mlb == 2:
    cost_list = [1, 3, 4, 2, 6, 5, 7]
    # Calibrated Label Ranking models training
    print('Training Calibrated Label Ranking models...')
    preds = []
    cost_CLR = []
    for e, (data, target) in enumerate(ml.CalibratedLabelRanking(X_train, y_train)):
       
        labels = np.unique(target)
        if e == 0 or len(labels) == 1:
            preds.append([1 for i in range(len(y_test))])
            continue

        if cost_list[labels[1]] > cost_list[labels[0]]:
            cost_1 = cost_list[labels[0]]
            cost_2 = cost_list[labels[1]]
            target = (target == labels[1]).astype(int)
        else:
            cost_1 = cost_list[labels[1]]
            cost_2 = cost_list[labels[0]]
            target = (target == labels[0]).astype(int)
            
        fp = np.full((y_test.shape[0], 1), cost_1)
        fn = np.full((y_test.shape[0], 1), cost_2)
        tp = np.zeros((y_test.shape[0], 1))
        tn = np.zeros((y_test.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))
        
        if cim == 2:
            data, target = classimbalance.random_undersampler(data, target)
        elif cim == 3:
            data, target = classimbalance.smote(data, target)
            
        if cm == 1:
            # Probability calibration using Isotonic Method
            cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            model = cc.fit(data, target)
            prob_test = model.predict_proba(X_test)
            bmr = BayesMinimumRiskClassifier(calibration=False)
            prediction = bmr.predict(prob_test, cost_matrix)
            if cost_list[labels[1]] > cost_list[labels[0]]:
                prediction = [labels[1] if i == 1 else labels[0] for i in prediction]
            else:
                prediction = [labels[0] if i == 1 else labels[1] for i in prediction]
            preds.append(prediction)
            
        elif cm == 2:
            # Probability calibration using CostCla calibration
            model = clf.fit(data, target)
            prob_train = model.predict_proba(data)
            bmr = BayesMinimumRiskClassifier(calibration=True)
            bmr.fit(target, prob_train)
            prob_test = model.predict_proba(X_test)
            prediction = bmr.predict(prob_test, cost_matrix)
            if cost_list[labels[1]] > cost_list[labels[0]]:
                prediction = [labels[1] if i == 1 else labels[0] for i in prediction]
            else:
                prediction = [labels[0] if i == 1 else labels[1] for i in prediction]
            preds.append(prediction)
            
        elif cm == 3:
            # Cost minimization using class weighting
            clf = LogisticRegression(C=12.0, class_weight={0: cost_1, 1:cost_2})
            model = clf.fit(data, target)
            prediction = model.predict(X_test)
            if cost_list[labels[1]] > cost_list[labels[0]]:
                prediction = [labels[1] if i == 1 else labels[0] for i in prediction]
            else:
                prediction = [labels[0] if i == 1 else labels[1] for i in prediction]
            preds.append(prediction)

    cost_list = [3, 4, 2, 6, 5, 7]
    preds = np.transpose(preds)
    pred_CLR = []
    cost = 0
    
    for pred in preds:
        pred_CLR.append([1 if list(pred).count(i + 1) > list(pred).count(0) else 0 for i in range(6)])
    pred_CLR = np.array(pred_CLR)
    
    for cnt, item in enumerate(pred_CLR):
        for e,i in enumerate(item):
            if i != y_test[cnt][e]:
                cost += cost_list[e]
                
# Print classification results
print('// Accuracy and Cost of the approaches used //')

if mlb == 1:
    print()
    print('\t-Binary Relevance-')
    print()
    print('Accuracy per example:', accuracy_score(pred_BR, y_test))
    print('Accuracy and Cost per label')
    for i in range(6):
        label_acc = accuracy_score(pred_BR[:, i], y_test[:, i])
        print('Accuracy: ',label_names[i], "->", label_acc)
        print('Cost: ',label_names[i], '->', cost_BR[i])
    print('Total Cost: ', sum(cost_BR))
    
elif mlb == 2:
    print()
    print('\t-Calibrated Label Ranking-')
    print()
    print('Accuracy per example:', accuracy_score(pred_CLR, y_test))
    print('Accuracy per label')
    for i in range(6):
        label_acc = accuracy_score(pred_CLR[:, i], y_test[:, i])
        print('Accuracy: ',label_names[i], "->", label_acc)
    print('Total Cost: ', cost)
