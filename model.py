import numpy as np
import pandas as pd

true=pd.read_csv('dataset/True.csv')
false=pd.read_csv('dataset/Fake.csv')
true['label']=1
false['label']=0
news=pd.concat([true,false], axis=0)
news=news.drop(['title','subject','date'],axis=1)
news=news.sample(frac=1)
news.reset_index(inplace=True)
news.drop(['index'], axis=1, inplace=True)

import re
def clean_text(text):
    #lowercase text
    text= text.lower()
    # removing any url
    text= re.sub(r'https?://\S+|www\.S+','',text)
    # removing html tags
    text= re.sub(r'<.*?>','',text)
    #removing punctuation
    text= re.sub(r'[^\w\s]','',text)
    #removing digits
    text= re.sub(r'\d','',text)
    #removing newline character
    text= re.sub(r'\n','',text)

    return text

news['text']=news['text'].apply(clean_text)

# dividing dependent and independent variables
x=news['text']
y=news['label']

# splitting into train and test set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=32)

# changing data(words) into numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization= TfidfVectorizer()
xv_train= vectorization.fit_transform(x_train)
xv_test= vectorization.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier()
classifier.fit(xv_train,y_train)    





