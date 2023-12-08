# File Structure

## Transformers
This folder contains the source code used to run and test the transformers used in our project.

### Contents:
- `Basic Transformer.py`: contains the code needed to run the 4 transformer models.
- `Albert_finetuned_model`: the folder contains the fine-tuned Albert model.
- `Albert.py`: contains the code used for the fine-tuned Albert model.

## Simple Models
This folder contains the source code needed to run the simple models and the improvements implemented.

### Contents:
- `Improvements and fine-tuning`: this folder contains the code needed to implement the random forest model and hyperparameter tuning for the SVM models.
- `Simple_models.py`: contains the code needed to run the 7 simple models used in the project with the chosen feature engineering and resampling methods implemented.
- `SoftMarginSVM.ipynb`: contains the code for the SoftMarginSVM, not included in the 7 simple models in `simple_models.py`.

## Feature Engineering and Resampling
This folder contains the source code needed to run all the feature engineering and resampling methods that we tried.

### Contents:
- `Feature Engineering`: contains the code for parts-of-speech, ngrams, dependency features.
- `Data Processing`: contains the code for SMOTE and its variants, ADASYN, and the oversampling and undersampling.

## Dataset
This folder contains the data taken from the Corpus of Linguistic Acceptability.

### Content:
- `Cola_public`: folder contains the tokenized and raw dataset.
- `Vocab_100k.tsv`: contains the vocabulary used in the running of the original LSTM model.
  - Source: [CoLA Baselines README](https://github.com/nyu-mll/CoLA-baselines/blob/master/README.md)

## LSTM_Basic_Run_(using research paper GitHub code).ipynb
Contains the code used to run the original LSTM model.

- **Code Source:** [CoLA Baselines README](https://github.com/nyu-mll/CoLA-baselines/blob/master/README.md)
- **Research Paper Source:** [arXiv:1805.12471](https://arxiv.org/pdf/1805.12471.pdf)

