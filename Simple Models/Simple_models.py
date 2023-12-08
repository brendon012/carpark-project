import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# Load your data using pandas, assuming your data is in a DataFrame

train_data = pd.read_csv("C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/train.tsv",sep='\t')
train_texts = train_data.iloc[:,3]
train_labels = train_data.iloc[:,1]

valid_data = pd.read_csv("C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/dev.tsv",sep='\t')
val_texts = valid_data.iloc[:,3]
val_labels = valid_data.iloc[:,1]

test_data = pd.read_csv("C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/test.tsv",sep='\t')
test_texts = test_data.iloc[:,3]
test_labels = test_data.iloc[:,1]

# Feature extraction using CountVectorizer and n_grams
vectorizer = CountVectorizer(ngram_range=(3, 3))  
train_vec = vectorizer.fit_transform(train_texts)
val_vec = vectorizer.transform(val_texts)
test_vec = vectorizer.transform(test_texts)

# Resampling training data with Adasyn
ada = ADASYN(random_state=42)
vec_resampled, labels_resampled = ada.fit_resample(X=train_vec, y=train_labels)

#classifiers, just change the model = "desired type of classifier":

#decision tree
dtc = DecisionTreeClassifier()

#logistic regression
#logreg = LogisticRegression()

#linear svm
#linear_svm = svm.SVC(kernel='linear')

#polynomial kernel svm
#polynomial_svm = svm.SVC(kernel='poly')

#rbf kernel svm
#rbf_svm = svm.SVC(kernel='rbf')

#sigmoid kernel svm
#sigmoid_svm = svm.SVC(kernel="sigmoid")

#change model variable respectively
model = dtc
model.fit(vec_resampled, labels_resampled)

validation_pred = model.predict(val_vec)
validation_accuracy = metrics.accuracy_score(val_labels, validation_pred)
validation_precision = metrics.precision_score(val_labels, validation_pred)
validation_recall = metrics.recall_score(val_labels, validation_pred)
validation_roc_auc = metrics.roc_auc_score(val_labels, validation_pred)
validation_mcc = metrics.matthews_corrcoef(val_labels, validation_pred)

test_pred = model.predict(test_vec)
test_accuracy = metrics.accuracy_score(test_labels, test_pred)
test_precision = metrics.precision_score(test_labels, test_pred)
test_recall = metrics.recall_score(test_labels, test_pred)
test_roc_auc = metrics.roc_auc_score(test_labels, test_pred)
test_mcc = metrics.matthews_corrcoef(test_labels, test_pred)

# Possible Metrics
print(f"Validation Accuracy: {validation_accuracy}") 
print(f"Validation Precision: {validation_precision}")
print(f"Validation Recall: {validation_precision}")
print(f"Validation ROC_AUC: {validation_roc_auc}")
print(f"Validation MCC: {validation_mcc}")

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_precision}")
print(f"Test ROC_AUC: {test_roc_auc}")
print(f"Test MCC: {test_mcc}")
