# Code to take a directory of papers and transform them into the training data format needed for training many models.
# Data format is:
#
#     data_item = {
#         filename: filename of this paper - string
#         gold: gold summary of this paper - list of lists of words, each list of words is a highlight statement
#         title: title of this paper - list of list of words, each list of words is a sentence (only one sent for title)
#         abstract: abstract of this paper - list of lists of words, each list of words is a sentence in abstract
#         abstract_vec: averaged non-stopword word vectors of the abstract
#         sentences: list of sentences, section they came from and their classification
#         sentence_vecs: list of average sentence word vectors, section they came from and their classification
#         sentence_features: the set of 8 features for each sentence
#         description: string describing the data format
#    }
#
# FINDING THE TRAINING SENTENCES:
# ----> There are 20 summary sentences in each paper, including the highlights. These are found by computing ROUGE-L
#       between each sentence and the highlights and taking the sentences with the top scores.
# ----> There are an equal number of negative sentences to positive ones. These are found by sorting the sentences by
#       their ROUGE-L score into ascending order, and taking the same number of negative examples as there are positive
#       examples from the sentences with the lowest ROUGE-L scores.
# ----> The section which each sentence comes from also needs to be stored.
#
# COMPUTING THE FEATURES:
# ----> There are eight features, their computation is handled by functions in useful_functions.py
# ----> The items required to calculate features are:
#           => The sentence to calculate features for
#           => Bag of words representation of the paper in which the sentence occurs
#           => Count of how many different papers each word occurs in (for TF-IDF)
#           => Keyphrases of the paper in which the sentence occurs
#           => Abstract of the paper in which the sentence occurs
#           => Title of the paper in which the sentence occurs
#           => Section of the paper in which the sentence occurs
#
# ALGORITHM:
# ----> Can be done in parallel, so long as we have a count of the number of different papers each word occurs in (for
#       TF-IDF), should be stored in "Data/Utility_Data/Global_Counts/".
# (1)  Create list of filenames of papers in directory.
# (2)  Take filename and read in paper using function in useful_function.py. This returns the paper as a dictionary of
#      the form {PAPER_SECTION: SECTION_TEXT}. SECTION_TEXT should be a list of sentences.
# (3)  Extract the Highlights section from the paper. This is the gold summary. Extract Abstract, Keyphrases and Title.
# (4)  Compute ROUGE-L of each sentence and Highlights and store in a list (sentence section also included here).
# (5)  Sort the list of sentences and ROUGE-L scores into descending order.
# (6)  Take the top 20 sentences as the positive data for that paper.
# (7)  If there is not as many negative sentences as there are positive ones, stop.
# (8)  Take an equal number of negative sentences as there are positive sentences, worst scored first.
# (9)  Concatenate the positive and negative sentences and shuffle the list.
# (10) Copy the section and classification to a new list and average the word vectors of the sentence to create the
#      sentence_vecs list.
# (11) Calculate the features for each sentence and place in new list, this is sentence_features. This does NOT have a
#      classification and section associated with each entry, ONLY the 8 features. Will need to calculate paper bag of
#      words to do this step and extract keyphrases.
# (12) Extract the title and add it to the dictionary.
# (13) Extract the abstract and add it to the dictionary.
# (14) Calculate the average abstract vector, add to the dictionary.
# (15) Write the dictionary to disk.
#
# NOTES:
# Original training data had 9879 items, this code will a produce list with 10024 items - the original should have had
# 10024 as well but due to a well concealed bug only saved 9879 of them. This should not matter for comparision to our
# models as models should be compared on CSPUBSUM TEST which is 150 specified papers. The original training data is
# available on request if needed.

# ======== IMPORT STATEMENTS ========

from __future__ import print_function, division
import sys
import os
import pickle
import time
import random
import ujson as json
from operator import itemgetter
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import wait, BASE_DIR, PAPER_SOURCE, GLOBAL_WORDCOUNT_WRITE_LOC,\
    TRAINING_DATA_WRITE_LOC, Color, NUMBER_OF_PAPERS
from Dev.Evaluation.rouge import Rouge


# ===================================

# ======== CONFIG VARS ========

# Create a ROUGE evaluation object
rouge = Rouge()

# The number of summary sentences to find in each paper
num_summary = 20

# =============================

# ======== FUNCTIONS ========


def process_paper(filename):
    """
    The concurrent function which processes each paper into its training data format.
    :param filename: the filename of the paper to process.
    :return: none, but write the preprocessed file to "Data/Training_Data/"
    """

    # Start time
    start_time = time.time()

    # Read in the paper
    paper = useful_functions.read_in_paper(filename, sentences_as_lists=True)

    # Extract the gold summary
    gold = paper["HIGHLIGHTS"]
    gold_string_list = [" ".join(x) for x in gold]

    # Extract the title
    title = paper["MAIN-TITLE"][0]
    title_string = " ".join(title)

    # Extract the abstract
    abstract = paper["ABSTRACT"]
    abstract_string_list = [" ".join(x) for x in abstract]

    # Extract the keyphrases
    try:
        keyphrases = paper["KEYPHRASES"][0]
    except IndexError:
        keyphrases = []

    # Turn the paper into a single string and calculate the bag of words score
    paper_string = " ".join([" ".join(x) for key, val in paper.iteritems() for x in val])
    bag_of_words = useful_functions.calculate_bag_of_words(paper_string)

    # Get the paper as a list of sentences, associating each sentence with its section name - will be used by oracle
    # to find best summary sentences.
    paper_sentences = [(" ".join(x), key) for key, val in paper.iteritems() for x in val
                       if key != "ABSTRACT"]

    # Create a list of sentences, their ROUGE-L scores with the Highlights and the section they occur in
    # (as a string)
    sents_scores_secs = []

    for sentence, section in paper_sentences:
        # For some reason the candidate sentence needs to be the only item in a list
        r_score = rouge.calc_score([sentence], gold_string_list)

        sents_scores_secs.append((sentence.split(" "), r_score, section))

    # Sort the sentences, scores and sections into descending order
    sents_scores_secs = sorted(sents_scores_secs, key=itemgetter(1), reverse=True)

    pos_sents_scores_secs = sents_scores_secs[:num_summary]
    neg_sents_scores_secs = sents_scores_secs[num_summary:]

    if len(neg_sents_scores_secs) < len(pos_sents_scores_secs):
        print("{}**** NOT A SUFFICIENT AMOUNT OF DATA IN PAPER {}, IGNORING PAPER ****{}".format(
            Color.RED, filename, Color.END))
        return

    # Positive sentences
    positive_sents_secs_class = [(sent, sec, 1) for sent, _, sec in pos_sents_scores_secs]

    # Negative sentences

    # Take the sentences not used as positive and reverse it to have worst scores first then take an equal number
    neg_sents_scores_secs = [x for x in reversed(neg_sents_scores_secs)][:len(positive_sents_secs_class)]
    negative_sents_secs_class = [(sent, sec, 0) for sent, _, sec in neg_sents_scores_secs]

    # Don't create data from this paper if it's less than 40 sentences - i.e. there would be more positive than
    # negative data. The data needs to be balanced.
    #if len(positive_sents_secs_class) != len(negative_sents_secs_class):
    #    print("{}**** NOT A SUFFICIENT AMOUNT OF DATA IN PAPER {}, IGNORING PAPER ****{}".format(
    #        Color.RED, filename, Color.END))
    #    return

    # Concatenate the positive and negative sentences into a single data item and shuffle it
    data = positive_sents_secs_class + negative_sents_secs_class
    random.shuffle(data)

    # Average word vectors of each sentence and convert to list for JSON serialisation
    sentvecs_secs_class = [(useful_functions.sentence2vec(sent).tolist(), sec, y) for sent, sec, y in data]

    # Calculate features for each sentence
    features = [useful_functions.calculate_features(sent, bag_of_words, document_wordcount, keyphrases,
                                                    abstract_string_list, title_string, sec)
                for sent, sec, y in data]

    # Calculate abstract vector
    abs_vector = useful_functions.abstract2vector(abstract_string_list).tolist()

    # Description of the data
    description_text = "All text is of the form of a list of lists, where each sentence is a list of words. The" \
                       " sentences are of the form [(sentence (as a list of words), section in paper," \
                       " classification)]. The sentence vectors are of a similar form, except the sentence text is" \
                       " replaced with the vector representation of the sentence. The features are of the form " \
                       "[(AbstractROUGE, TF-IDF, Document_TF-IDF, keyphrase_score, title_score, numeric_score," \
                       " sentence_length, section)]. The dimensions of each sentence vector are [1x100]. The " \
                       "abstract vector is a single [1x100] vector also."

    # The data item that will be written for this paper
    data_item = {
        "filename": filename,
        "gold": gold,
        "title": paper["MAIN-TITLE"],
        "abstract": abstract,
        "abstract_vec": abs_vector,
        "sentences": data,
        "sentence_vecs": sentvecs_secs_class,
        "sentence_features": features,
        "description": description_text
    }

    # Write the data out
    with open(TRAINING_DATA_WRITE_LOC + filename.strip(".txt") + ".json", "wb") as f:
        json.dump(data_item, f)

    print("--> Finished processing {}, took {} seconds, data length: {}.".format(
        filename, (time.time() - start_time), len(data)))

# ===========================

# ======== MAIN ========

print()
print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")
print(">>>>>>>> COMMENCING PREPROCESSING <<<<<<<<")
print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")
print()

# The document wordcount - the count of how many different papers each word occurs in - is stored by default in
# PROJECT_BASE_DIR / Data / Utility_Data / Global_Counts / document_wordcount.pkl - a variable maintained in
# useful_functions.py

print("----> Checking folder {} {} {} for the existence of {} \"document_wordcount.pkl\" {}".format(
    Color.RED, GLOBAL_WORDCOUNT_WRITE_LOC, Color.END, Color.RED, Color.END))

# Check whether we have counted how many different papers each word has occured in and written it, and if not,
# do so.
if not os.path.exists(GLOBAL_WORDCOUNT_WRITE_LOC + "document_wordcount.pkl"):

    print("----> File does not exist, counting words across all papers...")

    useful_functions.count_words_across_all_papers(write_location=GLOBAL_WORDCOUNT_WRITE_LOC)

    print("\n----> Done.")

    # Sanity check that the file was actually written
    assert(os.path.exists(GLOBAL_WORDCOUNT_WRITE_LOC + "document_wordcount.pkl"))

# Load the count of how many different papers each word occurs in
document_wordcount = pickle.load(open(GLOBAL_WORDCOUNT_WRITE_LOC + "document_wordcount.pkl", "rb"))

print("----> Count of number of papers each word occurs in exists.")

# Preprocess the papers to turn them into suitable training data

print("----> Beginning preprocessing stage")

# Check the data save location exists
if not os.path.exists(TRAINING_DATA_WRITE_LOC):
    os.makedirs(TRAINING_DATA_WRITE_LOC)

# Sanity check
assert(os.path.exists(TRAINING_DATA_WRITE_LOC))

# Get the list of paper filenames to process
filenames = [x for x in os.listdir(PAPER_SOURCE) if x.endswith(".txt")]

print("----> Starting thread pool...")

# Create the thread pool
pool = Pool()

print("----> Done.")

print("----> Starting processes...")
start_time = time.time()

# Begin the processing - takes 134 minutes to run on late 2016 MacBook Pro, dual core i7
pool.map(process_paper, filenames)

print("----> Papers processed, took {} minutes.".format((time.time() - start_time)/60))

data_list = []
i = 0
for fname in [x for x in os.listdir(TRAINING_DATA_WRITE_LOC) if x.endswith(".json")]:
    print("Reading item {}".format(i), end="\r")
    sys.stdout.flush()
    data_list.append(json.load(open(TRAINING_DATA_WRITE_LOC + fname, "rb")))
    i += 1

with open(TRAINING_DATA_WRITE_LOC + "all_data.json", "wb") as f:
    json.dump(data_list, f)

print("----> Data written")

# ======================
