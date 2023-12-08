import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

nlp = spacy.load("en_core_web_sm")

# Load your data using pandas, assuming your data is in a DataFrame
data = pd.read_csv("C:/Users/65929/OneDrive/Desktop/in_domain_train.tsv", sep="\t", header= None)

# Tokenize and perform part-of-speech tagging
sentences = data.iloc[:,3]
labels = data.iloc[:,1]
tagged_data = []

for i in range(len(sentences)):
    doc = nlp(sentences[i])
    pos_tags = [token.pos_ for token in doc]
    tagged_data.append(" ".join(pos_tags))

data.insert(loc = 4, column = None, value = tagged_data)

# Feature extraction using CountVectorizer and n_grams

ngram_training_texts, ngram_test_texts, ngram_training_labels, ngram_test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)
ngram_train_texts, ngram_val_texts, ngram_train_labels, ngram_val_labels = train_test_split(ngram_training_texts, ngram_training_labels, test_size=0.5, random_state=42)
vectorizer = CountVectorizer(ngram_range=(1, 3))  # This considers unigrams, bigrams, and trigrams
train_vec = vectorizer.fit_transform(ngram_train_texts)
val_vec = vectorizer.transform(ngram_val_texts)
test_vec = vectorizer.transform(ngram_test_texts)

dtc = DecisionTreeClassifier()
dtc.fit(train_vec, ngram_train_labels)

validation_pred = dtc.predict(val_vec)
validation_accuracy = accuracy_score(ngram_val_labels, validation_pred)

test_pred = dtc.predict(test_vec)
test_accuracy = accuracy_score(ngram_test_labels, test_pred)

print(f"Ngram Validation Accuracy: {validation_accuracy}")
print(f"Ngram Test Accuracy: {test_accuracy}")