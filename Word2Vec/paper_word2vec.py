# ======== PROJECT CONFIGURATION IMPORTS ========


from __future__ import print_function
import os
import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import csv


# ===============================================

# ============= ML IMPORT STATEMENTS ============


import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from Dev import DataTools
from Dev.DataTools.file_reader import Reader
from Dev.DataTools.paper_tokenizer import wait

from nltk.tokenize import sent_tokenize, word_tokenize

# This is so that word2vec produces nice output messages
import logging

# The word2vec algorithm itself
from gensim.models import word2vec


# ============================================

# ========== CONFIGURATION VARIABLES =========


DEBUG = False
BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"
DATA_SOURCE = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/sentence_issummary_listform.csv"


# ============================================

# ================= FUNCTIONS ================


def read_in_files():
    """Function which reads in all the scientific paper data, and parses it into a list of lists, where each item in
       the list is a sentence, in the form of a list of words. This is the form needed for the Word2Vec model."""

    num_files = len([name for name in os.listdir(DATA_SOURCE) if name.endswith(".txt")])
    loading_section_size = num_files / 30
    count = 0

    sentences_as_lists = []
    for filename in os.listdir(DATA_SOURCE):
        if filename.endswith(".txt"):

            # Pretty loading bar
            print("Processing Files: [", end="")
            for i in range(31, -1, -1):
                if count > i * loading_section_size:
                    for j in range(0, i):
                        print("-", end="")
                        sys.stdout.flush()
                    for j in range(i, 30):
                        print(" ", end="")
                        sys.stdout.flush()
                    break;
            if count == num_files:
                print("] ", count, end="\n")
            else:
                print("] ", count, end="\r")
            sys.stdout.flush()

            # Open the paper
            paper_to_open = DATA_SOURCE + filename
            paper = Reader().open_file_single_string(paper_to_open)
            udata = paper.decode("utf-8")
            paper = udata.encode("ascii", "ignore")

            # Split the data into a list of sentences, where each sentence is a list of words
            sentences = sent_tokenize(paper)

            for sentence in sentences:
                words = word_tokenize(sentence)
                sentences_as_lists.append(words)

            if DEBUG:
                print(sentences_as_lists)
                wait()

            count += 1

    return sentences_as_lists


def read_data(source):
    """
    Reads the sentence data from the csv file, which is of the form (sentence, is_summary_sentence).
    Args:
        source = the data file to read the data from
    Returns:
        A list of tuples where each tuple is of the form (sentence, is_summary_sentence).
    """

    sentences = []
    count = 0
    with open(source, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sentence = row[0]
            sentence = sentence.strip("\"")
            sentence = sentence.strip("[")
            sentence = sentence.strip("]")
            sentence = sentence.replace("'", "")
            sentence = sentence.replace(" ", "")
            sentence = sentence.split(",")
            sentences.append(sentence)
            count += 1

    return sentences


# ============================================

# ================ MAIN PROGRAM ==============


# Read in all of the papers into a list of lists. Each item in the list is a sentence, in the form of a list of words.
print("Reading data...")
sentences = read_data(DATA_SOURCE)
print("Done")

# Configure the logging module so that Word2Vec outputs pretty error messages
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Set the parameters for the Word2Vec model

#     The number of features for each word
num_features = 100

#     The number of times a word should occur in the data before it is counted as part of the model
min_word_count = 5

#     The number of threads to run in parallel
num_workers = 4

#     The context window size for each word
context = 20

#     Downsample setting for frequent words
downsampling = 1e-3

# Initialise and train the model
print("Beginning Word2Vec model training...")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
print("Done")

# Save the model
model_name = "summarisation_" + str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context"
model.save(model_name)


# ============================================
