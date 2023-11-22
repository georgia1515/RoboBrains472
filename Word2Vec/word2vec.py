import pandas as pd
import gensim.downloader as api

# Task 1 - Evaluation of the word2vec-google-news-300 Pre-trained Model
# Load the dataset
df = pd.read_csv('synonym.csv')
questions = df['question'].tolist()

# Word2Vec gooogle news model
wv = api.load('word2vec-google-news-300')


# Task 2 - Comparison with other pre-trained models

# Task 3 - Train own models