# ======== PROJECT CONFIGURATION IMPORTS ========


from __future__ import print_function, division
import os
import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import csv
from joblib import Parallel, delayed
import multiprocessing
import threading

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
DATA_SOURCE = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/sentence_issummary.csv"
FILE_LIMIT = 1000

# ============================================

# ================= FUNCTIONS ================


def sentence2list(data_item):
    """
    Takes a sentence and splits it into individual words, ready to be converted to their word representation from the
    Word2Vec model.
    Args:
        sentence = the sentence to split into words
    Returns:
        The sentence as a list of words
    """
    words = word_tokenize(data_item[0])
    new_item = (words, data_item[1])
    print("Processing item: ", data_item[2])
    return new_item


def read_data(source):
    """
    Reads the sentence data from the csv file, which is of the form (sentence, is_summary_sentence).
    Args:
        source = the data file to read the data from
    Returns:
        A list of tuples where each tuple is of the form (sentence, is_summary_sentence).
    """

    data = []
    count = 0
    with open(source, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sentence = row[0]
            sentence = sentence.strip("'")
            data.append((sentence, int(row[1].strip("'")), count))
            count += 1

    return data

def sentence_data_2_list(data):
    """
    Converts the data, which is of the form (sentence, is_summary) into the form ([sentence word list], is_summary)
    """

    num_cores = multiprocessing.cpu_count()

    new_data = Parallel(n_jobs=num_cores)(delayed(sentence2list)(item) for item in data)

    LOCATION_TO_WRITE = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/sentence_issummary_listform.csv"

    with open(LOCATION_TO_WRITE, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for item in new_data:
            csvwriter.writerow(item)

    print("Length of data is: ", len(new_data))

    for item in new_data:
        print(item)
        wait()

    return new_data

def csv2list():
    """Reads in the csv file of sentences and writes them as a series of lists where each list is a sentence."""

    data = read_data(DATA_SOURCE)
    data = sentence_data_2_list(data)

    return None


# ============================================

# ================ MAIN PROGRAM ==============

# Analyse the files
csv2list()

# ============================================
