# ================ IMPORTS ================

from __future__ import print_function, division
import os
import dill
import pickle
import sys
import random
import time
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import matplotlib.pyplot as plt
from operator import itemgetter
from Dev.DataTools.Reader import Reader
from Dev.DataTools.DataPreprocessing.DataPreprocessor import DataPreprocessor
from multiprocessing.dummy import Pool as ThreadPool
from Dev.DataTools import useful_functions
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.useful_functions import Color, wait, printlist, num2onehot, BASE_DIR, PAPER_SOURCE, HIGHLIGHT_SOURCE,\
    PAPER_BAG_OF_WORDS_LOC, KEYPHRASES_LOC, GLOBAL_COUNT_LOC, WORD2VEC, weight_variable, bias_variable, conv2d, max_pool_2x2
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.LSTM_preproc.vocab import Vocab
from Dev.DataTools.LSTM_preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from Dev.DataTools.LSTM_preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import tensorflow as tf
import numpy as np

# =========================================

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

# The maximum length that a sentence could be (sentences longer than this are discarded)
MAX_SENT_LEN = 100

def get_data():
    """
    Loads the data from the data directory given above and puts it into the form required by the summarisers. In this
    summariser the data we require is: the raw sentences, the abstract and the features.
    :return: The data, but discarding the sentences longer than the maximum length.
    """

    print("Loading Data...")
    t = time.time()

    # The data is a pickled object
    data = useful_functions.load_cspubsumext()

    # Data list
    sents_absvec_feats_class = []

    for item in data:

        sentences = item["sentences"]
        abstract_vec = item["abstract_vec"]
        features = item["sentence_features"]

        for sentence, feat in zip(sentences, features):
            sent = sentence[0]
            sec = sentence[1]
            y = sentence[2]
            sents_absvec_feats_class.append((sent, abstract_vec, feat, y))

    data = sents_absvec_feats_class

    print("Done, took ", time.time() - t, " seconds")

    print("Processing Data...")

    new_data = []
    for sent, abs_vec, feat, y in data:
        if len(sent) > MAX_SENT_LEN:
            new_sent = sent[0:MAX_SENT_LEN]
        else:
            new_sent = sent
        new_data.append((feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], y))

    return new_data

# Load the data
data = get_data()

# Split into test and train sets
test_len = int(len(data) * (1/3))
test_data = data[0:test_len]

pos_test = len([x for x in test_data if x[-1] == 1])
neg_test = len([x for x in test_data if x[-1] == 0])

print(pos_test)
print(neg_test)

pred_vals_pos = [1] * pos_test
pred_vals_neg = [0] * neg_test


# ========> AbstractROUGE <========
abs_rouge_ranked = sorted(test_data, key=itemgetter(0), reverse=True)
abs_rouge_predict = zip(abs_rouge_ranked, pred_vals_pos + pred_vals_neg)
abs_rouge_predictions = [(ys[8], y_) for ys, y_ in abs_rouge_predict]

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support([y for y, _ in abs_rouge_predictions], [y_ for _, y_ in abs_rouge_predictions], average="binary")
acc = accuracy_score([y for y, _ in abs_rouge_predictions], [y_ for _, y_ in abs_rouge_predictions])

print("====> AbstractROUGE <====")
print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

# ========> TF-IDF <========
feat_ranked = sorted(test_data, key=itemgetter(1), reverse=True)
feat_predict = zip(feat_ranked, pred_vals_pos + pred_vals_neg)
feat_predictions = [(ys[8], y_) for ys, y_ in feat_predict]

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions], average="binary")
acc = accuracy_score([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions])

print("\n====> TFIDF <====")
print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

# ========> Keyphrase Score <========
feat_ranked = sorted(test_data, key=itemgetter(3), reverse=True)
feat_predict = zip(feat_ranked, pred_vals_pos + pred_vals_neg)
feat_predictions = [(ys[8], y_) for ys, y_ in feat_predict]

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions], average="binary")
acc = accuracy_score([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions])

print("\n====> Keyphrase Score <====")
print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

# ========> Title Score <========
feat_ranked = sorted(test_data, key=itemgetter(4), reverse=True)
feat_predict = zip(feat_ranked, pred_vals_pos + pred_vals_neg)
feat_predictions = [(ys[8], y_) for ys, y_ in feat_predict]

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions], average="binary")
acc = accuracy_score([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions])

print("\n====> Title Score <====")
print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

# ========> Document TF-IDF <========
feat_ranked = sorted(test_data, key=itemgetter(2), reverse=True)
feat_predict = zip(feat_ranked, pred_vals_pos + pred_vals_neg)
feat_predictions = [(ys[8], y_) for ys, y_ in feat_predict]

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions], average="binary")
acc = accuracy_score([y for y, _ in feat_predictions], [y_ for _, y_ in feat_predictions])

print("\n====> Document TF-IDF <====")
print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)





