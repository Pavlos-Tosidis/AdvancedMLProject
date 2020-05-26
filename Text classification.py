import multilabel as ml
#sadasdasd

import re
import nltk 
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

myData = pd.read_csv('C:/Users/spgen/OneDrive/Desktop/Tsoumakas/Data/train.csv')

X = myData.iloc[:,1].values
y = myData.iloc[:,2:].values

label_names = ["toxic","severe_toxic","obscene",
               "threat","insult","identity_hate"]


# Preprocessing of Wikipedia edits.
# -----------------------------------------------------------------------------

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
    
def identity_tokenizer(text):
    return text

# Transforming data using tf-idf
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)    
X = tfidf.fit_transform(X)