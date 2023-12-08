import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")

# Load your data using pandas, assuming your data is in a DataFrame
data = pd.read_csv("C:/Users/65929/OneDrive/Desktop/in_domain_train.tsv", sep="\t", header= None)

# Tokenize and perform part-of-speech tagging
sentences = data.iloc[:,3]
tagged_data = []

for i in range(len(sentences)):
    doc = nlp(sentences[i])
    pos_tags = [token.pos_ for token in doc]
    tagged_data.append(" ".join(pos_tags))

data.insert(loc = 4, column = None, value = tagged_data)

# Feature extraction using CountVectorizer
pos_vectorizer = CountVectorizer(binary = True)
pos_vectorized_tags = pos_vectorizer.fit_transform(data.iloc[:,4])
labels = data.iloc[:,1]
training_texts, test_texts, training_labels, test_labels = train_test_split(pos_vectorized_tags, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(training_texts, training_labels, test_size=0.5, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Initialize and train a Decision Tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(train_texts, train_labels)

pos_validation_pred = dtc.predict(val_texts)
pos_validation_accuracy = accuracy_score(val_labels, pos_validation_pred)

pos_test_pred = dtc.predict(test_texts)
pos_test_accuracy = accuracy_score(test_labels, pos_test_pred)

print(f"POS Validation Accuracy: {pos_validation_accuracy}")
print(f"POS Test Accuracy: {pos_test_accuracy}")


