# NLP-Based Job Advertisement Classification System

## Project Overview
This project aims to automate the classification of job advertisements into relevant categories using natural language processing (NLP) techniques. By analyzing job description texts, the system can predict appropriate categories, thus enhancing the accuracy and efficiency of job search platforms.

## Repository Contents
- **Milestone I_Natural Language Processing.pdf**: Project specifications and detailed description.
- **task1.py**: Python script for basic text preprocessing.
- **task2_3.py**: Python script for generating feature representations and performing job advertisement classification.
- **count_vectors.txt**: Contains sparse count vector representations of job advertisements.
- **preprocessed_ads.txt**: Preprocessed text of job advertisements.
- **stopwords_en.txt**: List of stopwords used in text preprocessing.
- **vocab.txt**: Vocabulary file listing words used in the model and their indexes.

## Getting Started
### Setup Instructions
1. **Environment Setup**:
   - Ensure Python is installed on your system.
   - Install necessary Python libraries such as `numpy`, `pandas`, `sklearn`, and `nltk`.
   - Clone the repository or download all provided files into a single directory.

2. **Running the Scripts**:
   - **Text Preprocessing**: Run `python task1.py` to preprocess the job advertisement texts based on specified criteria such as tokenization, stopword removal, and frequency filtering.
   - **Feature Generation and Classification**: Run `python task2_3.py` to generate feature representations and classify job advertisements. This script utilizes the outputs from `task1.py` including the vocabulary and preprocessed texts.

## Project Details
### Text Preprocessing (task1.py)
- Extract job advertisement information and perform text cleaning.
- Tokenize texts using specified regex patterns.
- Convert tokens to lowercase, remove short words, and filter out stopwords.
- Generate a vocabulary from the cleaned text and save it.

### Feature Representation and Classification (task2_3.py)
- Create bag-of-words models and TF-IDF weighted embeddings for job descriptions.
- Apply machine learning models like logistic regression to classify job advertisements into predefined categories based on their descriptions.

### Output Files
- **vocab.txt**: Each line contains a word and its corresponding index in the format `word_string:word_integer_index`, sorted alphabetically.
- **count_vectors.txt**: Sparse representation of job descriptions in the format specified in the project document.

## Conclusion
This NLP-based job advertisement classification system streamlines the process of categorizing job postings on digital platforms. By automating this process, the system significantly reduces human error and improves the relevancy of job searches for users.
"""
