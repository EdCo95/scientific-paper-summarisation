# ================ IMPORTS ================
from __future__ import print_function, division
import sys
import os
import csv
import pickle
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.useful_functions import wait
from operator import itemgetter

# =========================================

# ================ CONFIG VARIABLES ================

BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"

# Word2Vec Classifier Trained with the Abstract as Positive Data
#GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/Word2Vec_AbstractPos/Gold/"
#REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/Word2Vec_AbstractPos/Text/"
#STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/Word2Vec_AbstractPos/scores.csv"

# Word2Vec Classifier Trained with the Abstract as Negative Data
#GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/Word2Vec_AbstractNeg/Gold/"
#REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/Word2Vec_AbstractNeg/Text/"
#STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/Word2Vec_AbstractNeg/scores.csv"

# DoubleDataset Classifier Trained with the Abstract as Negative Data
#GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/DoubleDataset_AbstractNeg/Gold/"
#REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/DoubleDataset_AbstractNeg/Text/"
#STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/DoubleDataset_AbstractNeg/scores.csv"

# LSTM Classifier Trained with the Abstract as Negative Data
#GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/LSTM_AbstractNeg/Gold/"
#REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/LSTM_AbstractNeg/Text/"
#STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/LSTM_AbstractNeg/scores.csv"

# TFIDF Unsupervised Classifier
GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/TFIDF/Gold/"
REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/TFIDF/Text/"
STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/TFIDF/scores.csv"
PICKLE_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/TFIDF/scores.pkl"

# Oracle ROUGE with the Abstract as Negative Data
GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/OracleROUGE/Gold/"
REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/OracleROUGE/Text/"
STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/OracleROUGE/scores.csv"
PICKLE_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/OracleROUGE/scores.pkl"

# Abstract ROUGE with the Abstract as Negative Data
GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/AbstractROUGE/Gold/"
REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/AbstractROUGE/Text/"
STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/AbstractROUGE/scores.csv"
PICKLE_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/AbstractROUGE/scores.pkl"

# Features summariser on standard dataset
GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/FeaturesSummariser/Gold/"
REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/FeaturesSummariser/Text/"
STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/FeaturesSummariser/scores.csv"
PICKLE_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/FeaturesSummariser/scores.pkl"

# Features summariser on standard dataset
SUMM_NAME = "KLSummariser"
GOLD_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/" + SUMM_NAME + "/Gold/"
REFERENCE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/" + SUMM_NAME + "/Text/"
STATS_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/" + SUMM_NAME + "/scores.csv"
PICKLE_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/" + SUMM_NAME + "/scores.pkl"

# ==================================================

golds = []
generateds = []
for filename in os.listdir(GOLD_DIR):
    gold = []
    generated = []
    if filename.endswith(".txt"):
        with open(GOLD_DIR + filename, "rb") as f:
            gold = f.readlines()
        with open(REFERENCE_DIR + filename, "rb") as f:
            generated = f.readlines()
        golds.append((gold, filename))
        generateds.append((generated, filename))

# Sanity checks
assert(type(golds) is list)
assert(type(generateds) is list)
assert(len(golds) > 0)
assert(len(generateds) > 0)
assert(type(golds[0]) is tuple)
assert(type(generateds[0]) is tuple)
assert(type(golds[0][0][0]) is str)
assert(type(generateds[0][0][0]) is str)

r = Rouge()
summaries_and_scores = []
for gold, ref in zip(golds, generateds):

    avg_score = 0
    count = 0

    for sent in ref[0]:
        score = r.calc_score([sent], gold[0])
        avg_score += score
        count += 1

    if count > 0:
        avg_score = avg_score / count
    else:
        avg_score = 0
    summaries_and_scores.append((gold[0], ref[0], avg_score, gold[1]))

summaries_and_scores = sorted(summaries_and_scores, key=itemgetter(2))

for item in summaries_and_scores:
    print("\n")
    print("SCORE: ", item[2])
    print()
    print("GOLD:")
    for sent in item[0]:
        print(sent)
    print()
    print("SUMMARY:")
    for sent in item[1]:
        print(sent)
    print("========")

avg_all = 0
i = 0
for item in summaries_and_scores:
    avg_all += item[2]
    i += 1
avg_all /= i

# Summaries and scores has form [(highlights, summary, r_score, filename)]
score_dict = {}
for _, _, r_score, fname in summaries_and_scores:
    score_dict[fname] = r_score

with open(PICKLE_WRITE_LOC, "wb") as f:
    pickle.dump(score_dict, f)

summaries_and_scores.append(("", "", avg_all, "OVERALL AVERAGE"))

to_write = [(score, filename) for _, _, score, filename in summaries_and_scores]
with open(STATS_WRITE_LOC, "wb") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)
    writer.writerows(to_write)