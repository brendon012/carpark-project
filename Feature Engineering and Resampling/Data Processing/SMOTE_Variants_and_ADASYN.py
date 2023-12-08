import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imblearn
import sklearn
from imblearn.over_sampling import SMOTE, SMOTEN, ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer
from sent2vec.vectorizer import Vectorizer

# load data
train_data = pd.read_csv("C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/train.tsv", sep='\t')
train_sentences = train_data.iloc[:,3]
train_labels = train_data.iloc[:,1]

#original distribution
#print("original distribution:", train_labels.value_counts()/len(train_labels))
#train_labels.value_counts().plot(kind="bar")

#tokenize and vectorize train_sentences first before putting into sampling algo

#tokenize using TFIDF
#this form of vectorizing might not work for some sampling methods
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(train_sentences)

#tokenize using sen2vec
#data too large, need to split into smaller chunks to be vectorized
split = np.array_split(train_sentences, 3)

vectorizer_s2v1 = Vectorizer()
vectorizer_s2v1.run(split[0].tolist(), remove_stop_words = [])
vectors_s2v1 = vectorizer_s2v1.vectors

vectorizer_s2v2 = Vectorizer()
vectorizer_s2v2.run(split[1].tolist(), remove_stop_words = [])
vectors_s2v2 = vectorizer_s2v2.vectors

vectorizer_s2v3 = Vectorizer()
vectorizer_s2v3.run(split[2].tolist(), remove_stop_words = [])
vectors_s2v3 = vectorizer_s2v3.vectors

X = vectors_s2v1 + vectors_s2v2 + vectors_s2v3

#Oversampling using SMOTE
#X_resampled, y_resampled = SMOTE().fit_resample(X=X, y=train_labels)
#resampled distribution
#print(y_resampled.value_counts()/len(y_resampled))

#Oversampling using SMOTEN
#sampler = SMOTEN(random_state=0)
#X_resampled, y_resampled = sampler.fit_resample(X=X, y=train_labels)
#resampled distribution
#print(y_resampled.value_counts()/len(y_resampled))

#Oversampling using ADASYN
ada = ADASYN(random_state=42)
X_resampled, y_resampled = ada.fit_resample(X=X, y=train_labels)
#resampled distribution
#print(y_resampled.value_counts()/len(y_resampled))

#decide on ADASYN