import gensim.downloader as api
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

# Task 1 - Evaluation of the word2vec-google-news-300 Pre-trained Model
# load the pretrained embedding model
wv = api.load('word2vec-google-news-300')

# load the dataset
synonyms = pd.read_csv('synonym.csv')

# compute cosine similarity between 2 embeddings (2 vectors)

# Function to find the closest synonym to a given question word
def find_closest_synonym_and_save(question_word, answer, *choices, result_csv_filename, name, total_correct, total_guess, total):
    try:
        total += 1
        # Compute cosine similarity between the question word and each choice
        similarity_scores = [(choice, wv.similarity(question_word, choice)) for choice in choices]

        # Find the most similar choice using cosine similarity
        most_similar_choice, _ = max(similarity_scores, key=lambda x: x[1])

        # Determine if the guess is correct or wrong
        result = 'correct' if most_similar_choice == answer else 'wrong'
        if result == 'correct':
            total_correct += 1

    except KeyError:
        most_similar_choice = None
        result = 'guess'
        total_guess += 1

    # Save the result to a CSV file
    result_data = {
        'question_word': question_word,
        'answer': answer,
        'system_guess': most_similar_choice,
        'result': result
    }
    result_df = pd.DataFrame([result_data])

    if isinstance(result_csv_filename, str):
        result_csv_filename += '-details.csv'
    elif name != None:
        result_csv_filename = name + '-details.csv'
    
    # Check if the result_csv_filename already exists
    result_df.to_csv(result_csv_filename, mode='a', header=False, index=False)
    return total_correct, total_guess, total

def write_Pre_trained_Analysis(model_name, total_correct, total_guess, total):
    # Get the size of the vocabulary
    vocabulary_size = len(wv.index_to_key)

    c = total_correct
    v = total - total_guess
    if v == 0:
        accuracy = -1
        print("Error: Division by zero. Cannot calculate accuracy.")
    else:
        accuracy = c / v

    analysis_filename = 'analysis.csv'

    # Open the CSV file in append mode and write the analysis
    with open(analysis_filename, mode='a', newline='') as csvfile:
        fieldnames = ['model_name', 'vocabulary_size', 'C', 'V', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write the analysis data
        writer.writerow({
            'model_name': model_name,
            'vocabulary_size': vocabulary_size,
            'C': c,
            'V': v,
            'accuracy': accuracy
        })
    return accuracy

def get_results_pre_trained_model(model_name):
    total_correct = 0
    total_guess = 0
    total = 0

    if os.path.isfile(model_name + '-details.csv'):
        with open(model_name + '-details.csv', 'w'):
            pass
    for index, row in synonyms.iterrows():
        total_correct, total_guess, total = find_closest_synonym_and_save(*row, result_csv_filename=model_name, name=None, total_correct=total_correct, total_guess=total_guess, total=total)

    print('Total correct: ', total_correct)
    print('Total guess: ', total_guess)
    print('Total: ', total)

    accuracy = write_Pre_trained_Analysis(model_name, total_correct, total_guess, total)
    return accuracy

analysis_filename = 'analysis.csv'
if os.path.isfile(analysis_filename):
    with open(analysis_filename, 'w'):
        pass

accuracies = []

# Load dataset from another file (replace 'your_dataset_file.csv' with the actual file name)
model_name = 'word2vec-google-news-300'
accuracy = get_results_pre_trained_model(model_name)
accuracies.append(accuracy)
# Task 2 - Comparison with other pre-trained models

# 2 new models from different corpora but same embedding size (200)
wv = api.load('glove-wiki-gigaword-200')
model_name = 'glove-wiki-gigaword-200'
accuracy = get_results_pre_trained_model(model_name)
accuracies.append(accuracy)

wv = api.load('glove-twitter-200')
model_name = 'glove-twitter-200'
accuracy = get_results_pre_trained_model(model_name)
accuracies.append(accuracy)
# 2 new models from same corpora (glove-wiki-gigaword) but different embedding size
wv = api.load('glove-wiki-gigaword-50')
model_name = 'glove-wiki-gigaword-50'
accuracy = get_results_pre_trained_model(model_name)
accuracies.append(accuracy)
wv = api.load('glove-wiki-gigaword-100')
model_name = 'glove-wiki-gigaword-100'
accuracy = get_results_pre_trained_model(model_name)
accuracies.append(accuracy)

model_names = ['word2vec-google-news-300', 'glove-wiki-gigaword-200', 'glove-twitter-200', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100']
# # Compare models with a bar chart
plt.bar(model_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Word Embedding Model Comparison')
plt.savefig('Accuracies.png')
plt.show()

# Task 3 - Train own models

def train_and_evaluate_model(window_size, embedding_size, tokenized_online_books, synonyms):
    # Train Word2Vec model on the online books
    model = Word2Vec(sentences=tokenized_online_books, vector_size=embedding_size, window=window_size, min_count=1, workers=4)

    # Save the trained model (optional)
    model.save(f'word2vec_model_W{window_size}_E{embedding_size}.model')

    # Load the trained model for further use
    loaded_model = Word2Vec.load(f'word2vec_model_W{window_size}_E{embedding_size}.model')

    # Example usage with your dataset
    result_csv_filename = f'output_results_W{window_size}_E{embedding_size}.csv'
    result_csv = f'output_results_W{window_size}_E{embedding_size}.csv'

    wv = loaded_model.wv
    total_correct = 0
    total_guess = 0
    total = 0

    for index, row in synonyms.iterrows():
        total_correct, total_guess, total = find_closest_synonym_and_save_own_corpus(
            *row, result_csv_filename=result_csv_filename, name=result_csv,
            total_correct=total_correct, total_guess=total_guess, total=total, wv=wv
        )
    print('Window size, embedding size: ', window_size, embedding_size)
    print('Total correct: ', total_correct)
    print('Total guess: ', total_guess)
    print('Total: ', total)

    accuracy = write_Pre_trained_Analysis(result_csv, total_correct, total_guess, total)
    return accuracy


def find_closest_synonym_and_save_own_corpus(question_word, answer, *choices, result_csv_filename, name, total_correct, total_guess, total, wv):
    try:
        total += 1
        # Compute cosine similarity between the question word and each choice
        similarity_scores = [(choice, wv.similarity(question_word, choice)) for choice in choices]

        # Find the most similar choice using cosine similarity
        most_similar_choice, _ = max(similarity_scores, key=lambda x: x[1])

        # Determine if the guess is correct or wrong
        result = 'correct' if most_similar_choice == answer else 'wrong'
        if result == 'correct':
            total_correct += 1

    except KeyError:
        most_similar_choice = None
        result = 'guess'
        total_guess += 1

    # Save the result to a CSV file
    result_data = {
        'question_word': question_word,
        'answer': answer,
        'system_guess': most_similar_choice,
        'result': result
    }
    result_df = pd.DataFrame([result_data])

    if isinstance(result_csv_filename, str):
        result_csv_filename += '-details.csv'
    elif name is not None:
        result_csv_filename = name + '-details.csv'

    # Check if the result_csv_filename already exists
    result_df.to_csv(result_csv_filename, mode='a', header=False, index=False)
    return total_correct, total_guess, total

# Assuming you have defined W1, W2, E5, E6, tokenized_online_books, and synonyms

W1 = 2
W2 = 5
E5 = 50
E6 = 100

# Download the NLTK data for tokenization
nltk.download('punkt')
# Define the folder containing your books
books_folder = 'books/'

# Function to preprocess books
def preprocess_books(folder_path):
    tokenized_books = []
    
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Preprocess the book and tokenize into words
            tokenized_words = []
            for line in lines:
                # Tokenize into words
                tokenized_words.extend(word_tokenize(line.lower()))

            tokenized_books.append(tokenized_words)

    return tokenized_books

# Call the function to preprocess books in the specified folder
tokenized_books = preprocess_books(books_folder)

# Iterate over different combinations of window size and embedding size
train_and_evaluate_model(W1, E5, tokenized_books, synonyms)
train_and_evaluate_model(W2, E5, tokenized_books, synonyms)
train_and_evaluate_model(W1, E5, tokenized_books, synonyms)
train_and_evaluate_model(W1, E6, tokenized_books, synonyms)
