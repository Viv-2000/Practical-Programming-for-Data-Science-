#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# ## Student Name: Vivek Aggarwal
# #### Student ID: s4015465
# 
# Date: 4-10-2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# os
# spacy
# sklearn
# 
# ## Introduction
# The offered code focuses on the use of machine learning and natural language processing (NLP) techniques to assess and categorize job adverts. Creating a vocabulary from a predetermined file is the first stage, after which count vectors indicating the frequency of words in preprocessed job descriptions are produced. The CountVectorizer from scikit-learn, a popular Python tool for machine learning, facilitates this vectorization procedure.
# 
# Next, the code makes use of the medium English model from spaCy to investigate the world of word embeddings. The word vectors inside each job description are averaged to create unweighted word embeddings, which offer a subtle representation of the text's substance. The code also generates TF-IDF weighted word embeddings, which capture the significance of particular words within the corpus.
# The blending of various feature representations is a crucial component of the code. These produce a rich set of features for each job advertisement, including count vectors, TF-IDF vectors, and unweighted word embeddings. The input for a logistic regression model—a popular option for multi-class classification tasks—is this integrated feature collection.
# 
# The code shows how to train and evaluate models holistically. The logistic regression model is trained on the training set and evaluated on the testing set after dividing the data into training and testing sets. The model's performance in classifying job advertising into predefined categories is measured using performance indicators like accuracy, classification report, and confusion matrix.
# 
# The algorithm extends beyond text content to handle folder and file processing. In order to extract titles from files located in various folders, it navigates through directories. The range of information available for the classification task is then increased by incorporating these titles as extra characteristics.
# 
# The code repeats feature extraction and model training for titles, creating unique feature representations in a simultaneous effort. The final product contains a thorough examination of the model's performance for each feature set, illuminating how well different textual representations and extra factors perform in predicting the types of job adverts.
# 
# In conclusion, the code exhibits a thorough workflow, from feature extraction and vocabulary construction to model training and evaluation. Its adaptability is demonstrated by the inclusion of several textual representations and other elements, which provide job advertisements and their categorization a sophisticated understanding.
# 

# ## Importing libraries 

# In[1]:


# Code to import libraries
from sklearn.feature_extraction.text import CountVectorizer
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# ...... Sections and code blocks on buidling different document feature represetations
# 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[2]:


# Function to read the vocabulary from file
def read_vocabulary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        vocabulary = {word.strip().split(':')[0]: int(word.strip().split(':')[1]) for word in lines}
    return vocabulary

# Get the current working directory
current_dir = os.getcwd()

# Path to the vocabulary file
vocab_file = os.path.join(current_dir, "vocab.txt")

# Read the vocabulary
vocabulary = read_vocabulary(vocab_file)

# Path to the preprocessed job advertisements file
preprocessed_ads_file = "preprocessed_ads.txt"

# Path to the output count vectors file
count_vectors_file = "count_vectors.txt"

# Initialize the CountVectorizer with the provided vocabulary
vectorizer = CountVectorizer(vocabulary=vocabulary, token_pattern=r'\S+')

# Read the preprocessed job advertisements
with open(preprocessed_ads_file, 'r', encoding='utf-8') as file:
    job_ads = file.readlines()

# Generate the count vectors
count_vectors = vectorizer.fit_transform(job_ads)


# In[ ]:





# In[10]:


# Download the spaCy model
spacy.cli.download("en_core_web_md")


# In[12]:


# Load spaCy medium English model
nlp = spacy.load("en_core_web_md")

# Read preprocessed descriptions from file
with open("preprocessed_ads.txt", "r", encoding="utf-8") as file:
    preprocessed_descriptions = file.readlines()

# Build vocabulary
vocabulary = set(word for description in preprocessed_descriptions for word in description.split())

# Generate Count Vectors
count_vectorizer = CountVectorizer(vocabulary=vocabulary)
count_vectors = count_vectorizer.fit_transform(preprocessed_descriptions)

# Generate TF-IDF Vectors
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_descriptions)

# Generate Word Embeddings (Average of word vectors)
def get_word_embedding(description):
    tokens = nlp(description)
    word_vectors = [token.vector for token in tokens if token.text in vocabulary]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros_like(nlp.vocab["apple"].vector)  # Use a vector of zeros if no words are found

# Apply the function to each description
unweighted_word_embeddings = [get_word_embedding(description) for description in preprocessed_descriptions]


# In[13]:


# Print unweighted word embeddings
print("Unweighted Word Embeddings:")
print(np.array(unweighted_word_embeddings))

# Print TF-IDF weighted word embeddings
tfidf_weighted_word_embeddings = tfidf_vectors.dot(count_vectors.T).todense()
print("TF-IDF Weighted Word Embeddings:")
print(np.array(tfidf_weighted_word_embeddings))


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[ ]:


# Save the count vectors to the output file
with open(count_vectors_file, 'w', encoding='utf-8') as file:
    for i, count_vector in enumerate(count_vectors):
        # Get the webindex of the job advertisement
        webindex = i + 1  # Assuming webindex starts from 1

        # Extract non-zero elements from the count vector
        non_zero_elements = [(index, count) for index, count in enumerate(count_vector.toarray()[0]) if count != 0]

        # Convert the non-zero elements to a string
        elements_str = ', '.join(f"{index}: {count}" for index, count in non_zero_elements)

        # Write to the file in the specified format
        file.write(f"{webindex}: {elements_str}\n")


# ## Task 3. Job Advertisement Classification

# ...... Sections and code blocks on buidling classification models based on different document feature represetations. 
# Detailed comparsions and evaluations on different models to answer each question as per specification. 
# 
# <span style="color: red"> You might have complex notebook structure in this section, please feel free to create your own notebook structure. </span>

# In[ ]:


###Question 1


# In[17]:


# Load spaCy medium English model
nlp = spacy.load("en_core_web_md")

# Read preprocessed descriptions from file
with open("preprocessed_ads.txt", "r", encoding="utf-8") as file:
    preprocessed_descriptions = file.readlines()

# Load labels from folder names
label_folders = ["Accounting_Finance", "Engineering", "Healthcare_Nursing", "Sales"]
labels = []
for folder in label_folders:
    labels.extend([folder] * len(os.listdir(os.path.join("data", folder))))

# Build vocabulary
vocabulary = set(word for description in preprocessed_descriptions for word in description.split())

# Generate Count Vectors
count_vectorizer = CountVectorizer(vocabulary=vocabulary)
count_vectors = count_vectorizer.fit_transform(preprocessed_descriptions)

# Generate TF-IDF Vectors
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_descriptions)

# Generate Word Embeddings (Average of word vectors)
def get_word_embedding(description):
    tokens = nlp(description)
    word_vectors = [token.vector for token in tokens if token.text in vocabulary]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros_like(nlp.vocab["apple"].vector)  # Use a vector of zeros if no words are found

# Apply the function to each description
unweighted_word_embeddings = [get_word_embedding(description) for description in preprocessed_descriptions]

# Create a dictionary to store your feature representations
feature_representations = {
    'count_vectors': count_vectors,
    'tfidf_vectors': tfidf_vectors,
    'unweighted_word_embeddings': unweighted_word_embeddings
}

# Convert labels to numerical format
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(labels)

# Separate features for each representation
X_count_vectors = count_vectors.toarray()
X_tfidf_vectors = tfidf_vectors.toarray()
X_unweighted_word_embeddings = np.array(unweighted_word_embeddings)

# Concatenate features
features = np.concatenate([X_count_vectors, X_tfidf_vectors, X_unweighted_word_embeddings], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, numerical_labels, test_size=0.2, random_state=42)

# Define a logistic regression model for multi-class classification
model = LogisticRegression(multi_class='multinomial', max_iter=500)

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)

# Print or store the results
print(f"Results for Combined Features:")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(confusion)


# In[ ]:


##Task 3 Question 2


# In[22]:


# Specify the path to the data folder
data_folder = "data"  # Assuming 'data' is the main folder containing subfolders

# Initialize an empty list to store titles
titles = []

# Loop through each folder in the data folder
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)

    # Check if the item in the data folder is a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the item is a file and ends with .txt
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                # Read titles from the file and add to the list
                with open(file_path, "r", encoding="utf-8") as file:
                    titles.extend(file.read().splitlines())

# Now, 'titles' should contain titles from all files in all folders
print(titles)


# In[24]:


# Load spaCy medium English model
nlp = spacy.load("en_core_web_md")

# Read preprocessed descriptions from file
with open("preprocessed_ads.txt", "r", encoding="utf-8") as file:
    preprocessed_descriptions = file.readlines()

# Initialize an empty list to store titles
titles = []

# Specify the path to the data folder
data_folder = "data/folder_with_titles"

# Initialize lists to store descriptions, titles, and labels
descriptions = []
titles = []
labels = []

# Loop through each folder in the data folder
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)

    # Check if the item in the data folder is a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the item is a file and ends with .txt
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                # Read titles from the file and add to the list
                with open(file_path, "r", encoding="utf-8") as file:
                    titles.extend(file.read().splitlines())
                
                # Add the corresponding label
                labels.extend([folder_name] * len(titles))

# Generate Count Vectors for descriptions
count_vectorizer_desc = CountVectorizer()
count_vectors_desc = count_vectorizer_desc.fit_transform(preprocessed_descriptions)

# Generate TF-IDF Vectors for descriptions
tfidf_vectorizer_desc = TfidfVectorizer()
tfidf_vectors_desc = tfidf_vectorizer_desc.fit_transform(preprocessed_descriptions)

# Generate Word Embeddings (Average of word vectors) for descriptions
unweighted_word_embeddings_desc = [get_word_embedding(desc) for desc in preprocessed_descriptions]

# Generate Count Vectors for titles
count_vectorizer_title = CountVectorizer()
count_vectors_title = count_vectorizer_title.fit_transform(titles)

# Generate TF-IDF Vectors for titles
tfidf_vectorizer_title = TfidfVectorizer()
tfidf_vectors_title = tfidf_vectorizer_title.fit_transform(titles)

# Generate Word Embeddings (Average of word vectors) for titles
unweighted_word_embeddings_title = [get_word_embedding(title) for title in titles]

# Create a dictionary to store your feature representations
feature_representations = {
    'count_vectors_desc': count_vectors_desc,
    'tfidf_vectors_desc': tfidf_vectors_desc,
    'unweighted_word_embeddings_desc': unweighted_word_embeddings_desc,
    'count_vectors_title': count_vectors_title,
    'tfidf_vectors_title': tfidf_vectors_title,
    'unweighted_word_embeddings_title': unweighted_word_embeddings_title
}

# Convert labels to numerical format
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_representations, numerical_labels, test_size=0.2, random_state=42)

# Define a logistic regression model for multi-class classification
model = LogisticRegression(multi_class='multinomial', max_iter=500)

# Train and predict for each feature representation
for feature_name, feature_representation in feature_representations.items():
    # Train the model
    model.fit(X_train[feature_name], y_train)

    # Predictions on the test set
    predictions = model.predict(X_test[feature_name])

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # Print or store the results
    print(f"Results for {feature_name}:")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("\n")


# ## Summary
# The code uses machine learning and natural language processing to do a thorough study of job postings. By mixing several feature representations, such as count vectors, TF-IDF vectors, and word embeddings, it can classify data using a logistic regression model. The script also broadens its capabilities by adding titles as supplementary features. The model's success across various representations is shown by the assessment measures, which offer a full comprehension of job advertisement categorization.
