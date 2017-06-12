from __future__ import print_function, division
import os
import dill
import pickle
import sys
import random
import time
import csv
from multiprocessing.dummy import Pool as ThreadPool
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from operator import itemgetter
from collections import defaultdict
from Dev.DataTools.Reader import Reader
from Dev.DataTools.DataPreprocessing.DataPreprocessor import DataPreprocessor
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import wait, printlist, printdict, num2onehot, BASE_DIR, PAPER_SOURCE,\
    HIGHLIGHT_SOURCE, PAPER_BAG_OF_WORDS_LOC, KEYPHRASES_LOC, GLOBAL_COUNT_LOC, HIGHLIGHT, ABSTRACT, INTRODUCTION,\
    RESULT_DISCUSSION, METHOD, CONCLUSION, OTHER, STOPWORDS, NUMBER_OF_PAPERS
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.LSTM_preproc.vocab import Vocab
from Dev.DataTools.LSTM_preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from Dev.DataTools.LSTM_preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30
GRAPH_SAVE_DIR = BASE_DIR + "/Analysis/Graphs/"

# Count of how many papers have been processed.
count = 0

# Sentences as vectors with their classification
data = []

# Count of positive data
pos_count = 0

# Count of negative data
neg_count = 0

r = Rouge()

pool = ThreadPool(2)

# True if already processed
PREPROCESSED = False

# True if data is prepared for plotting
PREPARED = False

# To count number of highlights
num_highlights = 0

# Number of highlights in first 150 papers
num_highlights_150 = 0

if not PREPROCESSED:
    # Holds the ROUGE scores by section
    avg_section_lens = defaultdict(list)

    # Iterate over every file in the paper directory
    for filename in os.listdir(PAPER_SOURCE):

        # Ignores files which are not papers e.g. hidden files
        if filename.endswith(".txt"):

            # Opens the paper as a dictionary, with keys corresponding to section titles and values corresponding
            # to the text in that section. The text is given as a list of lists, each list being a list of words
            # corresponding to a sentence.
            paper = useful_functions.read_in_paper(filename, sentences_as_lists=True)

            # Get the highlights of the paper
            highlights = paper["HIGHLIGHTS"]
            hlen = len(highlights)

            if count < 150:
                num_highlights_150 += hlen

            num_highlights += hlen

            sentences = []

            # Iterate over the whole paper
            for section, sents in paper.iteritems():

                avg_section_lens[section].append(len(sents))

            if count % 1000 == 0:
                print("\nWriting data...")
                write_dir = BASE_DIR + "/Data/Generated_Data/Section_Length/"
                with open(write_dir + "section_lens.pkl", "wb") as f:
                    pickle.dump(avg_section_lens, f)
                print("Done")

            # Display a loading bar of progress
            useful_functions.loading_bar(LOADING_SECTION_SIZE, count, NUMBER_OF_PAPERS)
            count += 1

if not PREPARED:
    with open(BASE_DIR + "/Data/Generated_Data/Section_Length/section_lens.pkl", "rb") as f:
        avg_section_lens = pickle.load(f)

    section_lens_final = defaultdict(float)

    for section, sec_ls in avg_section_lens.iteritems():
        section_lens_final[section] = np.mean(sec_ls)

    for key, val in section_lens_final.iteritems():
        print(key, " ", val)

    print("\nWriting data...")
    write_dir = BASE_DIR + "/Data/Generated_Data/Section_Length/"
    with open(write_dir + "section_lens_final.pkl", "wb") as f:
        pickle.dump(section_lens_final, f)

    with open(write_dir + "section_lens_titles.csv", "wb") as f:
        writer = csv.writer(f)
        for key, val in section_lens_final.iteritems():
            writer.writerow((key,))

    with open(write_dir + "section_lens_values.csv", "wb") as f:
        writer = csv.writer(f)
        for key, val in section_lens_final.iteritems():
            writer.writerow((str(val),))

    print("Done")

    print("====> HIGHLIGHT COUNTS <====")
    print("----> The total number of highlights is: ", num_highlights)
    print("----> The number of highlights in the first 150 papers is: ", num_highlights_150)

if PREPARED:
    rouge_section_scores = []
    rouge_section_titles = []
    write_dir = BASE_DIR + "/Data/Generated_Data/Rouge_By_Section/"
    with open(write_dir + "ROUGE_by_section_processed.csv", "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            rouge_section_scores.append(float(row[1]))
            rouge_section_titles.append(row[0])

    copy_paste_scores = []
    copy_paste_titles = []

    with open(write_dir + "highlight_copy_and_paste.csv", "rb") as f:
        reader=csv.reader(f)
        for row in reader:
            copy_paste_scores.append(float(row[1]))
            copy_paste_titles.append(row[0])

    copy_paste_scores = np.divide(copy_paste_scores, sum(copy_paste_scores))

    copy_paste = zip(copy_paste_titles, copy_paste_scores)
    rouge_section = zip(rouge_section_titles, rouge_section_scores)

    rouge_score = []
    copy_paste_score = []
    titles = []

    for title, score in rouge_section:

        found = False

        for title2, score2 in copy_paste:

            if title2 == title:
                found = True
                rouge_score.append(score)
                copy_paste_score.append(score2)
                titles.append(title2)
                break

        if not found:
            rouge_score.append(score)
            copy_paste_score.append(0)
            titles.append(title)

    width=0.8

    rouge_section_scores = [x for x in reversed(rouge_section_scores)]
    rouge_section_titles = [x for x in reversed(rouge_section_titles)]
    indicies = np.arange(0, len(rouge_section_scores))
    fig, ax = plt.subplots()
    plt.tight_layout(pad=5)
    ax.barh(indicies, rouge_section_scores, 0.8, alpha=0.4, color="r")
    plt.yticks(indicies + (0.8 / 2), [n for n in rouge_section_titles])

    plt.xlabel("ROUGE Score")
    plt.title("ROUGE Score by Section of Paper\nin Relation to the Highlights")
    plt.savefig(GRAPH_SAVE_DIR + "rouge_by_section.png")
    plt.show()

    rouge_score = [x for x in reversed(rouge_score)]
    copy_paste_score = [x for x in reversed(copy_paste_score)]
    titles = [x for x in reversed(titles)]
    plt.barh(indicies, rouge_score, 0.8, alpha=0.4, color="r", label="ROUGE Score")
    plt.barh([i+0.25*width for i in indicies], copy_paste_score, 0.5*width, color="b", alpha=0.4, label="Copy/Paste Score")
    plt.yticks(indicies + width/2, [n for n in titles])
    plt.legend(loc=4)
    plt.tight_layout(pad=5)
    plt.xlabel(("ROUGE Score / Normalised Copy/Paste Count"))
    plt.title("Comparison of ROUGE Score by Section for the Highlights\nand Copy/Paste Count")
    plt.savefig(GRAPH_SAVE_DIR + "rouge_copy_paste_by_section.png")
    plt.show()