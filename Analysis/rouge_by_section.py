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
PREPROCESSED = True

# True if data is prepared for plotting
PREPARED = True

if not PREPROCESSED:
    # Holds the ROUGE scores by section
    rouge_by_section = defaultdict(list)

    # Iterate over every file in the paper directory
    for filename in os.listdir(PAPER_SOURCE):

        # Ignores files which are not papers e.g. hidden files
        if filename.endswith(".txt"):

            # Display a loading bar of progress
            useful_functions.loading_bar(LOADING_SECTION_SIZE, count, NUMBER_OF_PAPERS)
            count += 1

            # Opens the paper as a dictionary, with keys corresponding to section titles and values corresponding
            # to the text in that section. The text is given as a list of lists, each list being a list of words
            # corresponding to a sentence.
            paper = useful_functions.read_in_paper(filename, sentences_as_lists=True)

            # Get the highlights of the paper
            highlights = paper["HIGHLIGHTS"]
            highlights_join = [" ".join(x) for x in highlights]
            abstract = paper["ABSTRACT"]

            sentences = []

            # Iterate over the whole paper
            for section, sents in paper.iteritems():

                section_avg_score = 0
                i = 0

                # Iterate over each sentence in the section
                for sentence in sents:

                    # Calculate the ROUGE score and add it to the list
                    r_score = r.calc_score([" ".join(sentence)], highlights_join)
                    section_avg_score += r_score
                    i += 1

                if i > 0:
                    section_avg_score /= i
                else:
                    section_avg_score = 0

                rouge_by_section[section].append(section_avg_score)

            if count % 1000 == 0:
                print("\nWriting data...")
                write_dir = BASE_DIR + "/Data/Generated_Data/Rouge_By_Section/"
                with open(write_dir + "rouge_by_section_list.pkl", "wb") as f:
                    pickle.dump(rouge_by_section, f)
                print("Done")

if not PREPARED:
    with open(BASE_DIR + "/Data/Generated_Data/Rouge_By_Section/rouge_by_section_list.pkl", "rb") as f:
        rouge_by_section = pickle.load(f)

    rouge_by_section_final = defaultdict(float)

    for section, rouge_score_list in rouge_by_section.iteritems():
        rouge_by_section_final[section] = np.mean(rouge_score_list)

    for key, val in rouge_by_section_final.iteritems():
        print(key, " ", val)

    print("\nWriting data...")
    write_dir = BASE_DIR + "/Data/Generated_Data/Rouge_By_Section/"
    with open(write_dir + "rouge_by_section.pkl", "wb") as f:
        pickle.dump(rouge_by_section_final, f)

    with open(write_dir + "rouge_by_section_titles.csv", "wb") as f:
        writer = csv.writer(f)
        for key, val in rouge_by_section_final.iteritems():
            writer.writerow((key,))

    with open(write_dir + "rouge_by_section_values.csv", "wb") as f:
        writer = csv.writer(f)
        for key, val in rouge_by_section_final.iteritems():
            writer.writerow((str(val),))

    print("Done")

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

    for item in zip(titles, rouge_score, copy_paste_score):
        print(item)

    sys.exit()

    plt.barh(indicies, rouge_score, 0.8, alpha=0.4, color="r", label="ROUGE Score")
    plt.barh([i+0.25*width for i in indicies], copy_paste_score, 0.5*width, color="b", alpha=0.4, label="Copy/Paste Score")
    plt.yticks(indicies + width/2, [n for n in titles])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(pad=7)
    plt.xlabel(("ROUGE Score / Normalised Copy/Paste Count"))
    plt.title("Comparison of ROUGE Score by Section for the Highlights\nand Copy/Paste Count")
    plt.savefig(GRAPH_SAVE_DIR + "rouge_copy_paste_by_section.png")
    plt.show()