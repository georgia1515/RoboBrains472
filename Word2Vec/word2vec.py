import gensim.downloader as api
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Task 1 - Evaluation of the word2vec-google-news-300 Pre-trained Model

# Load the dataset
synonyms = pd.read_csv('synonym.csv')

# Function to find the closest synonym to a given question word
def findClosestSynonym(question_word, answer, *choices, result_csv_filename, total_correct, total_guess, total, wv):
    try:
        total += 1
        # Compute cosine similarity between 2 embeddings (2 vectors): between the question word and each choice
        similarity_scores = [(choice, wv.similarity(question_word, choice)) for choice in choices]

        # Find the closest synonym to question-word: most similar choice using cosine similarity
        most_similar_choice, _ = max(similarity_scores, key=lambda x: x[1])

        # Find the result and label it: correct or wrong
        result = 'correct' if most_similar_choice == answer else 'wrong'
        if result == 'correct':
            total_correct += 1

    # If the question word/all 4 guess-words are not in the model: label as guess (cannot find the most similar choice)
    except KeyError:
        most_similar_choice = None
        result = 'guess'
        total_guess += 1

    # Output of the result
    result_details = {
        'question_word': question_word,
        'answer_word': answer,
        'system_guess': most_similar_choice,
        'result': result
    }
    result_details_df = pd.DataFrame([result_details])

    if isinstance(result_csv_filename, str):
        result_csv_filename += '-details.csv'
    
    # Write the result to file
    result_details_df.to_csv(result_csv_filename, mode='a', header=False, index=False)

    return total_correct, total_guess, total

# Function to compute the model analysis: vocabulary size, C (correct label), V (questions without guessing) & accuracy
def computeModelAnalysis(model_name, total_correct, total_guess, total, wv):

    # Get size of the vocabulary
    vocabulary_size = len(wv.index_to_key)
    # Get number of correct labels
    c = total_correct
    # Get number of questions without guessing
    v = total - total_guess
    # Get accuracy of model
    if v == 0:
        accuracy = -1
        print("Error: Division by zero. Cannot calculate accuracy.")
    else:
        accuracy = c / v

    # Write the analysis to file
    analysis_filename = 'analysis.csv'
    fieldnames = ['model_name', 'vocabulary_size', 'C', 'V', 'accuracy']
    with open(analysis_filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'model_name': model_name,
            'vocabulary_size': vocabulary_size,
            'C': c,
            'V': v,
            'accuracy': accuracy
        })

    return accuracy

# Function to evaluate the pre-trained model: finding closest syn & writing analysis
def evaluatePretrainedModel(model_name, wv):
    total_correct = 0
    total_guess = 0
    total = 0

    # Clear the result file if it exists
    if os.path.isfile(model_name + '-details.csv'):
        with open(model_name + '-details.csv', 'w'):
            pass

    # Iterate over each row in the dataset: find closest synonym for each question word
    for index, row in synonyms.iterrows():
        total_correct, total_guess, total = findClosestSynonym(*row, result_csv_filename=model_name, total_correct=total_correct, total_guess=total_guess, total=total, wv=wv)
    
    # Compute the model analysis
    accuracy = computeModelAnalysis(model_name, total_correct, total_guess, total, wv)

    return accuracy

# Clear content of analysis file & accuracies list
accuracies = []
analysis_filename = 'analysis.csv'
if os.path.isfile(analysis_filename):
    with open(analysis_filename, 'w'):
        pass

# Load the pretrained embedding model
wv = api.load('word2vec-google-news-300')
model_name = 'word2vec-google-news-300'
accuracy = evaluatePretrainedModel(model_name, wv)
accuracies.append(accuracy)

# Task 2 - Comparison with other pre-trained models

# Experiment with: 2 new models from different corpora but same embedding size (200)
wv = api.load('glove-wiki-gigaword-200')
model_name = 'glove-wiki-gigaword-200'
accuracy = evaluatePretrainedModel(model_name, wv)
accuracies.append(accuracy)

wv = api.load('glove-twitter-200')
model_name = 'glove-twitter-200'
accuracy = evaluatePretrainedModel(model_name, wv)
accuracies.append(accuracy)

# Experiment with: 2 new models from same corpora (glove-wiki-gigaword) but different embedding size
wv = api.load('glove-wiki-gigaword-50')
model_name = 'glove-wiki-gigaword-50'
accuracy = evaluatePretrainedModel(model_name, wv)
accuracies.append(accuracy)

wv = api.load('glove-wiki-gigaword-100')
model_name = 'glove-wiki-gigaword-100'
accuracy = evaluatePretrainedModel(model_name, wv)
accuracies.append(accuracy)

model_names = ['word2vec-google-news-300', 'glove-wiki-gigaword-200', 'glove-twitter-200', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100']

# Compare models with a bar graph
plt.bar(model_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Word Embedding Model Comparison')
plt.savefig('pretrained_Accuracies_Comparison.png')
# plt.show()

# todo: compare performance of models (graphs)
# todo: compare models to random baseline
# todo: compare models to human gold-standard
# todo: analyze data points & why some models perform better than others

# Task 3 - Train own models

# Function to preprocess books
def preprocess_books(folder_path):
    tokenized_books = []
    
    # Iterate over each book & tokenize into words
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            tokenized_words = []
            for line in lines:
                # Tokenize into words & split natural language texts
                tokenized_words.extend(word_tokenize(line.lower()))
            tokenized_books.append(tokenized_words)

    return tokenized_books

# Function to create word2vec embeddings (model), train, evaluate the model (finding closest syn) & write analysis
def createTrainEvaluateModel(embedding_size, window_size, tokenized_online_books, synonyms):

    # Create and train Word2Vec model with books
    model = Word2Vec(sentences=tokenized_online_books, vector_size=embedding_size, window=window_size, min_count=1, workers=4)

    model_name = f'own_corpus-E{embedding_size}-W{window_size}'
    wv = model.wv
    total_correct = 0
    total_guess = 0
    total = 0

    # Clear the result file if it exists
    if os.path.isfile(model_name):
        with open(model_name, 'w'):
            pass

    # Iterate over each row in the dataset: find closest synonym for each question word
    for index, row in synonyms.iterrows():
        total_correct, total_guess, total = findClosestSynonym(*row, result_csv_filename=model_name, total_correct=total_correct, total_guess=total_guess, total=total, wv=wv)
    
    # Compute the model analysis
    accuracy = computeModelAnalysis(model_name, total_correct, total_guess, total, wv)
    return accuracy

nltk.download('punkt')
books_folder = 'books/'
W1 = 2; W2 = 5; E5 = 50; E6 = 100

# Preprocess the 5 online books
tokenized_books = preprocess_books(books_folder)

# Train & evaluate the model with different window sizes & embedding sizes: create word2vec embeddings (model), find closest synonyms & write analysis
accuracy = createTrainEvaluateModel(E5, W1, tokenized_books, synonyms)
accuracies.append(accuracy)
accuracy = createTrainEvaluateModel(E5, W2, tokenized_books, synonyms)
accuracies.append(accuracy)
accuracy = createTrainEvaluateModel(E6, W1, tokenized_books, synonyms)
accuracies.append(accuracy)
accuracy = createTrainEvaluateModel(E6, W2, tokenized_books, synonyms)
accuracies.append(accuracy)

model_names = ['E50-W2', 'E50-W5', 'E100-W2', 'E100-W5']

# Compare models with a bar graph
# plt.bar(model_names, accuracies)
# plt.ylabel('Accuracy')
# plt.title('Word Embedding Model Comparison')
# plt.savefig('own_Corpus_Accuracies_Comparison.png')
# plt.show()

# todo: compare performance of models (graphs)
# todo: analyze data points & why some models perform better than others