import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report

from torch.utils.data import random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2ForSequenceClassification
# cmd
# C:\ProgramData\Miniconda3\Scripts\activate.bat
# conda activate cs3244_1
# python C:\Users\kohli\Downloads\CS3244_project_codes\la.py
# Load and preprocess your data
train_data = pd.read_csv('C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/train.tsv', sep='\t')
train_texts = train_data.iloc[:, 3]
train_labels = train_data.iloc[:, 1]

valid_data = pd.read_csv('C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/dev.tsv', sep='\t')
val_texts = valid_data.iloc[:, 3]
val_labels = valid_data.iloc[:, 1]

test_data = pd.read_csv('C:/Users/yilin/Documents/NUS/CS3244/project/cola_public/tokenized/test.tsv', sep='\t')
test_texts = test_data.iloc[:, 3]
test_labels = test_data.iloc[:, 1]



# Initialize the ALBERT tokenizer and model
# tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
# model = AlbertForSequenceClassification.from_pretrained("albert-base-v2")

# Initialize the Roberta tokenizer and model
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base") 
# model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Initialize XLNet tokenizer and model
# tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
# model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)

# Initialize GPT2 tokenizer and model, note that there are additional lines needed for GPT2 to run
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configuration = GPT2Config()
model = GPT2ForSequenceClassification(configuration).from_pretrained('gpt2').to(device)
model.config.pad_token_id = model.config.eos_token_id

# Tokenize and prepare the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors='pt')

train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, torch.tensor(train_labels))
val_dataset = TensorDataset(val_encodings.input_ids, val_encodings.attention_mask, torch.tensor(val_labels))
test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, torch.tensor(test_labels))

# Set up data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader))

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Evaluation on the validation set
#model.eval()
#val_predictions = []
#val_ground_truth = []

#with torch.no_grad():
    #for batch in val_loader:
        #input_ids, attention_mask, labels = batch
        #input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        #outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #probabilities = softmax(outputs.logits, dim=1)
        # probabilities = outputs.logits
        #predicted_labels = torch.argmax(probabilities, dim=1)
        #val_predictions.extend(predicted_labels.cpu().numpy())
        #val_ground_truth.extend(labels.cpu().numpy())

# Evaluation metrics
#accuracy = accuracy_score(val_ground_truth, val_predictions)
#report = classification_report(val_ground_truth, val_predictions)
#print(f"Validation Accuracy: {accuracy}")
#print(report)

# Test set evaluation
test_predictions = []
test_ground_truth = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = softmax(outputs.logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        test_predictions.extend(predicted_labels.cpu().numpy())
        test_ground_truth.extend(labels.cpu().numpy())

# Test set metrics
accuracy = accuracy_score(test_ground_truth, test_predictions)
test_roc_auc = roc_auc_score(test_ground_truth, test_predictions)
test_mcc = matthews_corrcoef(test_ground_truth, test_predictions)
report = classification_report(test_ground_truth, test_predictions)
print(f"Test Accuracy: {accuracy}")
print(f"Test ROC_AUC: {test_roc_auc}")
print(f"Test MCC: {test_mcc}")
print(report)