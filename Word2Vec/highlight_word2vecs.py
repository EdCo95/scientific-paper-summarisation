# ======== PROJECT CONFIGURATION IMPORTS ========


from __future__ import print_function, division
import os
import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")


# ===============================================

# ============= ML IMPORT STATEMENTS ============


import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
from Dev import DataTools
from Dev.DataTools.file_reader import Reader
from Dev.DataTools.paper_tokenizer import wait, paper_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from operator import itemgetter


# ============================================

# ========== CONFIGURATION VARIABLES =========


DEBUG = True
REMOVE_STOPWORDS = False
BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"
DATA_SOURCE = BASE_DIR + "/Data/Papers/Full/Papers_With_Section_Titles/"
FILE_LIMIT = 1000

# ============================================

# ================= FUNCTIONS ================


def prepare_data(section_text):
    """ Prepares the raw text from each section of the data to take the average word vector for that section. Splits the   data into sentences and then to words, then removes characters such as newlines etc.
        Args:
            section_text = the raw text from each section of the data
        Returns:
            A list of lists, where each list is a list of words in the sentence
    """
    sentence_list = []
    sentences = sent_tokenize(section_text)
    for sentence in sentences:
        words_temp = word_tokenize(sentence)
        words = []
        for word in words_temp:
            words.append(word.strip(" \n\r\t"))
        sentence_list.append(words)
    return sentence_list


def calculate_average_vector(data, model, include_stopwords=True):
    """Calculates the average word vector for a section of data from the paper, using the provided Word2Vec model.
       Args:
           data = the data to calculate the average vector for, in the form of a list of sentences where each sentence  is a list of words
       Returns:
           The average word vector for the data section
    """

    # The shape of the model, used to get the number of features
    model_shape = model.syn0.shape

    # The words in the model
    model_vocabulary = set(model.index2word)

    # The array that will be used to calculate the average word vector
    average = np.zeros((1, model_shape[1]), dtype="float32")
    total_word_count = 0

    # Calculate the average vector
    for sentence in data:
        for word in sentence:
            if word in model_vocabulary:
                word_vec = model[word]
                average += word_vec
                total_word_count += 1

    average = np.divide(average, total_word_count)

    return average

def analyse_files(model):
    """Function which reads in all the scientific paper data, and parses it into a list of lists, where each item in
       the list is a sentence, in the form of a list of words. This is the form needed for the Word2Vec model."""

    num_files = len([name for name in os.listdir(DATA_SOURCE) if name.endswith(".txt")])
    loading_section_size = num_files / 30
    count = 0

    # Defaultdicts to store the average of word vectors for each section
    averages_per_section = defaultdict(float)
    section_counter = defaultdict(float)

    for filename in os.listdir(DATA_SOURCE):

        if count > FILE_LIMIT:
            break

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

            # Split the data into a dictionary where we can get each section of the paper by its name
            paper_data = paper_tokenize(paper)

            # Calculate the average vectors for each section
            for key, value in paper_data.iteritems():
                data = prepare_data(value)
                avg = calculate_average_vector(data, model)
                if not np.isnan(avg).any():
                    averages_per_section[key] = np.add(averages_per_section[key], avg)
                    section_counter[key] += 1

            count += 1

    final_averages = defaultdict(None)
    for key, value in averages_per_section.iteritems():
        final_averages[key] = np.divide(value, section_counter[key])

    if DEBUG:
        average_single_nums = defaultdict(float)

        for key, val in final_averages.iteritems():
            average_single_num = np.sum(val) / model.syn0.shape[1]
            average_single_nums[key] = average_single_num

        for key, val in sorted(average_single_nums.iteritems(), key=itemgetter(1)):
            print(key, " ", val)

        wait()

    return final_averages


# ============================================

# ================ MAIN PROGRAM ==============


# Load the Word2Vec model
model = Word2Vec.load("Word2Vec/summarisation_300features_20minwords_10context")

# Analyse the files
analyse_files(model)

# ============================================
