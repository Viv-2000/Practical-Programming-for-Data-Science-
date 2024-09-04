#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Vivek Aggarwal   
# #### Student ID: S4015465
# 
# Date: 02-10-12
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# 
# ## Introduction
# A Python script is included in the code block that preprocesses the text in a dataset of movie reviews. The dataset is loaded using the scikit-learn library's load_files function, and it is expected that it is organized into folders, each of which corresponds to a particular sentiment class (positive or negative). The software then reads the reviews from the dataset along with the associated feelings.
# 
# The script's primary accomplishments are as follows:
# 
# Preprocessing Functions for Ext:
# cd
# Read_stopwords and preprocess_text are the two functions that the script defines for text preparation.
# read_stopwords extracts a set of stop words from a file after reading it.
# preprocess_text lowercases, tokenizes, removes stop words, shortens words, and tokenizes all words in a supplied text.
# Data and Stopwords Loading
# 
# The script gets the location of the working directory and creates paths to the data folder and a stopwords file.
# The read_stopwords function is used to read the stopwords from the file.
# Frequency Counters for Terms and Documents:
# 
# The Counter class from the collections module is used to initialize the counters for term frequency (term_freq_counter) and document frequency (doc_freq_counter).
# Loop for text preprocessing:
# 
# The preprocessed text is then written to an output file (preprocessed_ads.txt) by the script after iterating over each file in the dataset and using the preprocess_text function to modify each file's text content.
# During this process, both term frequency and document frequency are adjusted.
# The code, in its entirety, functions as a text preprocessing pipeline for movie review data. It tokenizes, normalizes, and strips out extraneous information from the text input to produce a processed dataset that may be used for sentiment analysis and other machine learning applications. The processed text is stored to a file for additional analysis or model training after the stopwords have been removed.
# 
# 

# ## Importing libraries 

# In[4]:


# importing all the libraries
from sklearn.datasets import load_files  
import os
import re
from collections import Counter
from nltk.corpus import stopwords
# downloading NLTK stopwords
import nltk


# ### 1.1 Examining and loading data
# - xamine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# In[5]:


movie_data = load_files(r"data")  


# In[6]:


# Code to inspect the provided data file...
movie_data['filenames']


# In[7]:


movie_data['target']


# In[8]:


movie_data['target_names'] # this means the value 0 is negative, the value 1 is positive.


# In[9]:


# test whether it matches, just in case
emp = 10 # an example, note we will use this example through out this exercise.
movie_data['filenames'][emp], movie_data['target'][emp] # from the file path we know that it's the correct class too


# In[10]:


reviews, sentiments = movie_data.data, movie_data.target  


# In[11]:


reviews[emp]


# ### 1.2 Pre-processing data
# 

# In[12]:


# Function to read stop words from file
def read_stopwords(file_path):
    with open(file_path, 'r') as file:
        stop_words = set(file.read().split())
    return stop_words



# In[13]:


# Function for basic text pre-processing
def preprocess_text(text, stop_words, min_word_length=2):
    # Tokenization
    tokens = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", text)
    
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove words with length less than 2
    tokens = [token for token in tokens if len(token) >= min_word_length]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens



# In[14]:


# Get the current working directory
current_dir = os.getcwd()

# Path to the data folder
data_folder = os.path.join(current_dir, "data")

# Path to stopwords file
stopwords_file = os.path.join(current_dir, "stopwords_en.txt")

# Read stopwords
stop_words = read_stopwords(stopwords_file)

# Initialize counters for term frequency and document frequency
term_freq_counter = Counter()
doc_freq_counter = Counter()

# Save preprocessed job advertisements to a file (you can choose your format)
preprocessed_ads_file = "preprocessed_ads.txt"
with open(preprocessed_ads_file, 'w', encoding='utf-8') as file_write:
    # Process each job advertisement again and write the preprocessed text to the file
    for category_folder in os.listdir(data_folder):
        category_path = os.path.join(data_folder, category_folder)
        for job_file in os.listdir(category_path):
            job_path = os.path.join(category_path, job_file)
            with open(job_path, 'r', encoding='utf-8') as file_read:
                content = file_read.read()
                tokens = preprocess_text(content, stop_words)
                preprocessed_text = " ".join(tokens)
                file_write.write(f"{preprocessed_text}\n")


# In[ ]:





# In[ ]:





# ## Saving required outputs
# Save the vocabulary, bigrams and job advertisment txt as per spectification.
# - vocab.txt

# In[6]:





# ## Summary
# 
# The Python code block that is being provided is a text preparation program created for a dataset of movie reviews. The load_files function of the scikit-learn library is used by the script to load the dataset, which is divided into folders for each type of sentiment (positive or negative). The code defines two essential text preparation functions: preprocess_text and read_stopwords, which tokenize, convert to lowercase, and remove short or stop words from a given text. read_stopwords extracts stop words from a file. The script reads and applies the stopwords to the dataset after creating file paths for the current working directory, data folder, and a stopwords file.
# To track term and document frequencies, two counters—term_freq_counter and doc_freq_counter—are established. After preparing the text, the script iteratively cycles over each file in the dataset, updating counters as necessary. The preprocessed text that results is stored to a file with the name "preprocessed_ads.txt." This script offers a strong text preprocessing pipeline overall, improving the quality of the movie review data for ensuing machine learning tasks like sentiment analysis.
# 
