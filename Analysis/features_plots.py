# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import pickle
import time
import numpy as np
from scipy import spatial
import scipy.stats as stats
import tensorflow as tf
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE
from Dev.DataTools.DataPreprocessing.AbstractNetPreprocessor import AbstractNetPreprocessor
from Dev.Models.SummariserNetClassifier.summariser_net_v2 import graph, sents2input, SAVE_PATH, NUM_FEATURES,\
    ABSTRACT_DIMENSION, WORD_DIMENSIONS, MAX_SENT_LEN
from operator import itemgetter
from sklearn import linear_model
from sklearn.decomposition import PCA
from Dev.Evaluation.rouge import Rouge
from collections import defaultdict

GRAPH_SAVE_DIR = BASE_DIR + "/Analysis/Graphs/"
SUMMARY_BASE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/"
SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/SummariserNetV2Summariser/"
# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================

def load_scores(model_name):
    """
    Loads the summary scores
    :param model_name: name of the summary model's scores to load
    :return: a list and dict of the scores
    """
    with open(SUMMARY_BASE_DIR + model_name + "/scores.pkl", "rb") as f:
        model_dict = pickle.load(f)
        model_list = []
        for filename, val in model_dict.iteritems():
            model_list.append((filename, val))

        model_list = [x for x in reversed(sorted(model_list, key=itemgetter(1)))]

    return model_dict, model_list

if __name__ == "__main__":

    print("Reading data...")
    with open(DATA_DIR, "rb") as f:
        data_list = pickle.load(f)
    print("Done")

    features_class = []
    for item in data_list:
        sentences = item["sentences"]
        features = item["sentence_features"]
        for sent, feats in zip(sentences, features):
            features_class.append((feats, sent[2]))

    classes = []

    abs_rouges_pos = []
    abs_rouges_neg = []

    tf_idfs_pos = []
    tf_idfs_neg = []

    document_tf_idfs_pos = []
    document_tf_idfs_neg = []

    keyphrase_pos = []
    keyphrase_neg = []

    title_pos = []
    title_neg = []

    numeric_pos = []
    numeric_neg = []

    len_pos = []
    len_neg = []

    secs_pos = []
    secs_neg = []

    for feats, y in features_class:

        if feats[1] > 1500:
            continue

        classes.append(y)

        if y == 1:
            abs_rouges_pos.append(feats[0])
            tf_idfs_pos.append(feats[1])
            document_tf_idfs_pos.append(feats[2])
            keyphrase_pos.append(feats[3])
            title_pos.append(feats[4])
            numeric_pos.append(feats[5])
            len_pos.append(feats[6])
            secs_pos.append(feats[7])
        else:
            abs_rouges_neg.append(feats[0])
            tf_idfs_neg.append(feats[1])
            document_tf_idfs_neg.append(feats[2])
            keyphrase_neg.append(feats[3])
            title_neg.append(feats[4])
            numeric_neg.append(feats[5])
            len_neg.append(feats[6])
            secs_neg.append(feats[7])

    max_abs_rouge = 1
    max_tf_idfs = max([max(tf_idfs_pos), max(tf_idfs_neg)])
    max_doc_tf_idfs = max([max(document_tf_idfs_pos), max(document_tf_idfs_neg)])
    max_keyphrase = max([max(keyphrase_pos), max(keyphrase_neg)])
    max_title = max([max(title_pos), max(title_neg)])
    max_numeric = max([max(numeric_pos), max(numeric_neg)])
    max_len = max([max(len_pos), max(len_neg)])
    max_secs = max([max(secs_pos, max(secs_neg))])

    # ====> PCA <====
    feats = np.asarray([feat for feat, _ in features_class])
    ys = [y for _, y in features_class]

    pca = PCA(n_components=2)
    pca.fit(feats)
    reduced_dim = pca.transform(feats)
    reduced_dim_and_class = []
    for i in range(np.shape(reduced_dim)[0]):
        label = ys[i]
        data = reduced_dim[i, :]
        reduced_dim_and_class.append((data, label))

    print(np.shape(reduced_dim_and_class))
    wait()

    pos = [x for x, y in reduced_dim_and_class if y == 1]
    neg = [x for x, y in reduced_dim_and_class if y == 0]

    fig, ax = plt.subplots()
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter([x1 for x1, _ in pos], [x2 for _, x2 in pos], c="r", label=y, alpha=0.3, marker="s")
        else:
            plt.scatter([x1 for x1, _ in neg], [x2 for _, x2 in neg], c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("Reduced Dimension Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(GRAPH_SAVE_DIR + "pca_features.png")
    plt.show()

    sys.exit()

    # ====> Sentence Length vs Document TF-IDF <====
    new_document_tf_idfs_pos = [x for x in document_tf_idfs_pos if x < 150]
    new_document_tf_idfs_neg = [x for x in document_tf_idfs_neg if x < 150]
    new_len_pos = [x for x in len_pos if x < 1500]
    new_len_neg = [x for x in len_neg if x < 1500]

    fig, ax = plt.subplots()
    plt.xlim([0, 150])
    plt.ylim([0, 1500])
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter(document_tf_idfs_pos, len_pos, c="r", label=y, alpha=0.3, marker="s")
        else:
            plt.scatter(document_tf_idfs_neg, len_neg, c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("Sentence Length vs. Document TF-IDF")
    plt.xlabel("Document TF-IDF")
    plt.ylabel("Sentence Length")
    plt.savefig(GRAPH_SAVE_DIR + "sent_len_vs_doc_tf_idf.png")
    plt.show()
    sys.exit()

    # ====> Section Bar Chart <====
    secs_pos_count = defaultdict(float)
    secs_neg_count = defaultdict(float)

    for item in secs_pos:
        secs_pos_count[item] += 1.0

    for item in secs_neg:
        secs_neg_count[item] += 1.0

    # Only 6 because no abstract
    secs_pos_final = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    secs_neg_final = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for key, val in secs_pos_count.iteritems():
        if key > 0:
            secs_pos_final[key - 1] = val
        if key == 0:
            secs_pos_final[key] = val

    for key, val in secs_neg_count.iteritems():
        if key > 0:
            secs_neg_final[key - 1] = val
        if key == 0:
            secs_neg_final[key] = val

    print(len(secs_pos))
    print(len(secs_neg))
    print(sum(secs_pos_final))
    print(sum(secs_neg_final))
    print(sum(secs_pos_final) + sum(secs_neg_final))

    print("=============")
    print(secs_pos_final)
    print(secs_neg_final)
    wait()

    secs_pos_normed = np.divide(secs_pos_final, sum(secs_pos_final) + sum(secs_neg_final))
    secs_neg_normed = np.divide(secs_neg_final, sum(secs_pos_final) + sum(secs_neg_final))

    print(secs_pos_normed)
    print(secs_neg_normed)

    wait()

    indicies = np.arange(len(secs_neg_final))
    bar_width = 0.35

    rects1 = plt.bar(indicies, secs_pos_final, bar_width, alpha=0.4, color="r", label="Positive")
    rects2 = plt.bar(indicies + bar_width, secs_neg_final, bar_width, alpha=0.4, color="b", label="Negative")

    plt.xticks(indicies + (bar_width),
               ("HIGHLIGHT", "INTRO", "RESULTS/\nDISCUSSION", "METHOD", "CONC.", "OTHER"))

    plt.tight_layout(pad=2.5)
    plt.title("Count of Positive and Negative Examples\nby Section of Paper")
    plt.ylabel("Number of Examples")
    plt.savefig(GRAPH_SAVE_DIR + "pos_neg_counts_per_section.png")
    plt.show()
    sys.exit()

    # ====> Distribution of Numeric Score <===
    to_use_neg = []
    to_use_pos = []
    for i in document_tf_idfs_neg:
        if i < 50:
            to_use_neg.append(i)

    for i in document_tf_idfs_pos:
        if i < 50:
            to_use_pos.append(i)

    l_neg = sorted(numeric_neg)
    l_pos = sorted(numeric_pos)

    neg_mean = np.mean(l_neg)
    neg_std = np.std(l_neg)

    pos_mean = np.mean(l_pos)
    pos_std = np.std(l_pos)

    n, bins, patches = plt.hist(l_neg, normed=True, bins=50, alpha=0.4, facecolor="blue", label="Negative")
    n2, bins2, patches2 = plt.hist(l_pos, normed=True, bins=50, alpha=0.4, facecolor="red", label="Positive")

    y_neg = mlab.normpdf(bins, neg_mean, neg_std)
    l_neg = plt.plot(bins, y_neg, "b", linewidth=2)

    y_pos = mlab.normpdf(bins2, pos_mean, pos_std)
    l_pos = plt.plot(bins2, y_pos, "r", linewidth=2)

    plt.plot([neg_mean, neg_mean], [0, max(y_neg)], color="b", linestyle="--", linewidth=2)
    plt.plot([pos_mean, pos_mean], [0, max(y_pos)], color="r", linestyle="--", linewidth=2)

    plt.grid(True)
    #plt.xlim([0, 20])
    # plt.ylim([0, 0.02])
    plt.xlabel("Numeric Score")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of Numeric Scores")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "numeric_dist.png")
    plt.show()

    sys.exit()

    # ====> Distribution of Key Phrase Score <===
    to_use_neg = []
    to_use_pos = []
    for i in document_tf_idfs_neg:
        if i < 50:
            to_use_neg.append(i)

    for i in document_tf_idfs_pos:
        if i < 50:
            to_use_pos.append(i)

    l_neg = sorted(keyphrase_neg)
    l_pos = sorted(keyphrase_pos)

    neg_mean = np.mean(l_neg)
    neg_std = np.std(l_neg)

    pos_mean = np.mean(l_pos)
    pos_std = np.std(l_pos)

    n, bins, patches = plt.hist(l_neg, normed=True, bins=50, alpha=0.4, facecolor="blue", label="Negative")
    n2, bins2, patches2 = plt.hist(l_pos, normed=True, bins=50, alpha=0.4, facecolor="red", label="Positive")

    y_neg = mlab.normpdf(bins, neg_mean, neg_std)
    l_neg = plt.plot(bins, y_neg, "b", linewidth=2)

    y_pos = mlab.normpdf(bins2, pos_mean, pos_std)
    l_pos = plt.plot(bins2, y_pos, "r", linewidth=2)

    plt.plot([neg_mean, neg_mean], [0, max(y_neg)], color="b", linestyle="--", linewidth=2)
    plt.plot([pos_mean, pos_mean], [0, max(y_pos)], color="r", linestyle="--", linewidth=2)

    plt.grid(True)
    plt.xlim([0, 20])
    # plt.ylim([0, 0.02])
    plt.xlabel("Keyphrase Score")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of Keyphrase Scores")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "keyphrase_dist.png")
    plt.show()

    # ====> Normal Distribution of Document TF-IDF <====
    to_use_neg = []
    to_use_pos = []
    for i in document_tf_idfs_neg:
        if i < 50:
            to_use_neg.append(i)

    for i in document_tf_idfs_pos:
        if i < 50:
            to_use_pos.append(i)

    l_neg = sorted(to_use_neg)
    l_pos = sorted(to_use_pos)

    neg_mean = np.mean(l_neg)
    neg_std = np.std(l_neg)

    pos_mean = np.mean(l_pos)
    pos_std = np.std(l_pos)

    n, bins, patches = plt.hist(l_neg, normed=True, bins=50, alpha=0.4, facecolor="blue", label="Negative")
    n2, bins2, patches2 = plt.hist(l_pos, normed=True, bins=50, alpha=0.4, facecolor="red", label="Positive")

    y_neg = mlab.normpdf(bins, neg_mean, neg_std)
    l_neg = plt.plot(bins, y_neg, "b", linewidth=2)

    y_pos = mlab.normpdf(bins2, pos_mean, pos_std)
    l_pos = plt.plot(bins2, y_pos, "r", linewidth=2)

    plt.plot([neg_mean, neg_mean], [0, max(y_neg)], color="b", linestyle="--", linewidth=2)
    plt.plot([pos_mean, pos_mean], [0, max(y_pos)], color="r", linestyle="--", linewidth=2)

    plt.grid(True)
    plt.xlim([0, 30])
    # plt.ylim([0, 0.02])
    plt.xlabel("Document TF-IDF")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of Document TF-IDF Scores")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "document_tf_idf_dist.png")
    plt.show()

    # ====> Normal Distribution of Sentence Length <====
    to_use_neg = []
    to_use_pos = []
    for i in len_neg:
        if i < 200:
            to_use_neg.append(i)

    for i in len_pos:
        if i < 200:
            to_use_pos.append(i)

    l_neg = sorted(to_use_neg)
    l_pos = sorted(to_use_pos)

    neg_mean = np.mean(l_neg)
    neg_std = np.std(l_neg)

    pos_mean = np.mean(l_pos)
    pos_std = np.std(l_pos)

    n, bins, patches = plt.hist(l_neg, normed=True, bins=50, alpha=0.4, facecolor="blue", label="Negative")
    n2, bins2, patches2 = plt.hist(l_pos, normed=True, bins=50, alpha=0.4, facecolor="red", label="Positive")

    y_neg = mlab.normpdf(bins, neg_mean, neg_std)
    l_neg = plt.plot(bins, y_neg, "b", linewidth=2)

    y_pos = mlab.normpdf(bins2, pos_mean, pos_std)
    l_pos = plt.plot(bins2, y_pos, "r", linewidth=2)

    plt.plot([neg_mean, neg_mean], [0, max(y_neg)], color="b", linestyle="--", linewidth=2)
    plt.plot([pos_mean, pos_mean], [0, max(y_pos)], color="r", linestyle="--", linewidth=2)

    plt.grid(True)
    plt.xlim([0, 200])
    #plt.ylim([0, 0.02])
    plt.xlabel("Sentence Length")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of Sentence Lengths")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "sent_len_dist.png")
    plt.show()

    # ====> Normal Distribution of TF IDF <====
    to_use_neg = []
    to_use_pos = []
    for i in tf_idfs_neg:
        if i < 300:
            to_use_neg.append(i)

    for i in tf_idfs_pos:
        if i < 300:
            to_use_pos.append(i)

    l_neg = sorted(to_use_neg)
    l_pos = sorted(to_use_pos)

    neg_mean = np.mean(l_neg)
    neg_std = np.std(l_neg)

    pos_mean = np.mean(l_pos)
    pos_std = np.std(l_pos)

    n, bins, patches = plt.hist(l_neg, normed=True, bins=50, alpha=0.4, facecolor="blue", label="Negative")
    n2, bins2, patches2 = plt.hist(l_pos, normed=True, bins=50, alpha=0.4, facecolor="red", label="Positive")

    y_neg = mlab.normpdf(bins, neg_mean, neg_std)
    l_neg = plt.plot(bins, y_neg, "b", linewidth=2)

    y_pos = mlab.normpdf(bins2, pos_mean, pos_std)
    l_pos = plt.plot(bins2, y_pos, "r", linewidth=2)

    plt.plot([neg_mean, neg_mean], [0, max(y_neg)], color="b", linestyle="--", linewidth=2)
    plt.plot([pos_mean, pos_mean], [0, max(y_pos)], color="r", linestyle="--", linewidth=2)

    plt.grid(True)
    plt.xlim([0, 300])
    plt.ylim([0, 0.02])
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of TF-IDF Scores")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "tf_idf_dist.png")
    plt.show()

    # ====> Normal Distribution of Abstract ROUGE <====
    abs_rouges_neg = sorted(abs_rouges_neg)
    abs_rouges_pos = sorted(abs_rouges_pos)

    neg_mean = np.mean(abs_rouges_neg)
    neg_std = np.std(abs_rouges_neg)

    pos_mean = np.mean(abs_rouges_pos)
    pos_std = np.std(abs_rouges_pos)

    n, bins, patches = plt.hist(abs_rouges_neg, normed=True, bins=50, alpha=0.4, facecolor="blue", label="Negative")
    n2, bins2, patches2 = plt.hist(abs_rouges_pos, normed=True, bins=50, alpha=0.4, facecolor="red", label="Positive")

    y_neg = mlab.normpdf(bins, neg_mean, neg_std)
    l_neg = plt.plot(bins, y_neg, "b", linewidth=2)

    y_pos = mlab.normpdf(bins2, pos_mean, pos_std)
    l_pos = plt.plot(bins2, y_pos, "r", linewidth=2)

    plt.plot([neg_mean, neg_mean], [0, max(y_neg)], color="b", linestyle="--", linewidth=2)
    plt.plot([pos_mean, pos_mean], [0, max(y_pos)], color="r", linestyle="--", linewidth=2)

    plt.grid(True)
    plt.xlabel("AbstractROUGE Score")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of AbstractROUGE Scores")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "abs_rouge_dist.png")
    plt.show()

    # ====> Sentence Length vs TF-IDF <====
    fig, ax = plt.subplots()
    plt.xlim([0, max_tf_idfs])
    plt.ylim([0, max_len])
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter(tf_idfs_pos, len_pos, c="r", label=y, alpha=0.3)
        else:
            plt.scatter(tf_idfs_neg, len_neg, c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("Sentence Length vs. TF-IDF")
    plt.xlabel("TF-IDF")
    plt.ylabel("Sentence Length")
    plt.savefig(GRAPH_SAVE_DIR + "sent_len_vs_tf_idf.png")
    plt.show()

    # ====> Sentence Length vs AbstractROUGE <====
    fig, ax = plt.subplots()
    plt.xlim([0, max_abs_rouge])
    plt.ylim([0, max_len])
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter(abs_rouges_pos, len_pos, c="r", label=y, alpha=0.3)
        else:
            plt.scatter(abs_rouges_neg, len_neg, c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("Sentence Length vs. AbstractRouge Metrics")
    plt.xlabel("AbstractROUGE")
    plt.ylabel("Sentence Length")
    plt.savefig(GRAPH_SAVE_DIR + "sent_len_vs_abs_rouge.png")
    plt.show()

    # ====> Keyphrase Score vs AbstractROUGE <====
    fig, ax = plt.subplots()
    plt.xlim([0, max_abs_rouge])
    plt.ylim([0, max_keyphrase])
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter(abs_rouges_pos, keyphrase_pos, c="r", label=y, alpha=0.3)
        else:
            plt.scatter(abs_rouges_neg, keyphrase_neg, c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("Keyphrase Score vs. AbstractRouge Metrics")
    plt.xlabel("AbstractROUGE")
    plt.ylabel("Keyphrase Score")
    plt.savefig(GRAPH_SAVE_DIR + "keyphrase_score_vs_abs_rouge.png")
    plt.show()

    # ====> Doc TF-IDF vs AbstractROUGE <====
    fig, ax = plt.subplots()
    plt.xlim([0, max_abs_rouge])
    plt.ylim([0, max_doc_tf_idfs])
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter(abs_rouges_pos, document_tf_idfs_pos, c="r", label=y, alpha=0.3)
        else:
            plt.scatter(abs_rouges_neg, document_tf_idfs_neg, c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("Document TF-IDF vs. AbstractRouge Metrics")
    plt.xlabel("AbstractROUGE")
    plt.ylabel("Document TF-IDF")
    plt.savefig(GRAPH_SAVE_DIR + "doc_tf_idf_vs_abs_rouge.png")
    plt.show()

    # ====> TF-IDF vs AbstractROUGE <====
    fig, ax = plt.subplots()
    plt.xlim([0, max_abs_rouge])
    plt.ylim([0, max_tf_idfs])
    for y in ["Negative", "Positive"]:
        if y == "Positive":
            plt.scatter(abs_rouges_pos, tf_idfs_pos, c="r", label=y, alpha=0.3)
        else:
            plt.scatter(abs_rouges_neg, tf_idfs_neg, c="b", label=y, alpha=0.3)

    ax.legend()
    ax.grid(True)
    plt.title("TF-IDF vs. AbstractRouge Metrics")
    plt.xlabel("AbstractROUGE")
    plt.ylabel("TF-IDF")
    plt.savefig(GRAPH_SAVE_DIR + "tf_idf_vs_abs_rouge.png")
    plt.show()






