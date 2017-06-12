# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import pickle
import time
import numpy as np
from scipy import spatial
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE
from Dev.DataTools.DataPreprocessing.AbstractNetPreprocessor import AbstractNetPreprocessor
from Dev.Models.SummariserNetClassifier.summariser_net_v2 import graph, sents2input, SAVE_PATH, NUM_FEATURES,\
    ABSTRACT_DIMENSION, WORD_DIMENSIONS, MAX_SENT_LEN
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

GRAPH_SAVE_DIR = BASE_DIR + "/Analysis/Graphs/"
SUMMARY_BASE_DIR = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/"
SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/SummariserNetV2Summariser/"
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

    # ====> Combined Weights <====
    with open(BASE_DIR + "/Data/Generated_Data/Weights/feature_weights_no_abs_rouge.npy", "rb") as f:
        feat_weights_no_abs_rouge = np.load(f)

    with open(BASE_DIR + "/Data/Generated_Data/Weights/feature_weights.npy", "rb") as f:
        feat_weights = np.load(f)

    weights_pos = list(feat_weights[:, 1])
    weights_pos_norouge = list(feat_weights_no_abs_rouge[:, 1])
    weights_pos_norouge.insert(0, 0)
    names = ["AbstractROUGE", "TF-IDF", "Document TF-IDF", "Keyphrase Score", "Title Score", "Numeric Count",
             "Sent Length", "Section"]

    weights_and_names = zip(weights_pos, weights_pos_norouge, names)
    weights_and_names = [x for x in reversed(sorted(weights_and_names, key=itemgetter(0)))]
    print(weights_and_names)
    wait()

    # Plot the result
    width = 0.4
    indicies = np.arange(0, len(weights_pos) * 1, 1)
    fig, ax = plt.subplots()
    plt.tight_layout(pad=7)

    for item in zip([n for _, _, n in weights_and_names], [x for x, _, _ in weights_and_names], [x for _, x, _ in weights_and_names]):
        print(item)
    exit()

    ax.barh(indicies, [x for x, _, _ in weights_and_names], height=width, alpha=0.4, color="b", label="With AbsROUGE")

    ax.barh(indicies + width, [x for _, x, _ in weights_and_names], height=width, alpha=0.4, color="r", label="Without AbsROUGE")

    plt.yticks(indicies + width, [n for _, _, n in weights_and_names])
    # ax.plot([0, len(performance_list)*1], [oracle[1], oracle[1]], "k--")

    plt.ylabel("Feature")
    plt.xlabel("Weighting")
    plt.title(
        "Comparison of the Weights for the Positive Class Given to Each Feature\nBy a Linear Classifier With and Without AbstractROUGE")
    plt.legend()
    plt.savefig(GRAPH_SAVE_DIR + "feature_weight_comparison_both.png")
    plt.show()
    sys.exit()

    # ====> Weights With AbstractROUGE <====
    with open(BASE_DIR + "/Data/Generated_Data/Weights/feature_weights.npy", "rb") as f:
        feat_weights = np.load(f)

    weights_pos = feat_weights[:, 1]
    names = ["AbstractROUGE", "TF-IDF", "Document TF-IDF", "Keyphrase Score", "Title Score", "Numeric Count", "Sent Length", "Section"]

    weights_and_names = zip(weights_pos, names)
    weights_and_names = [x for x in reversed(sorted(weights_and_names, key=itemgetter(0)))]

    # Plot the result
    indicies = np.arange(0, len(weights_pos) * 1, 1)
    fig, ax = plt.subplots()
    plt.tight_layout(pad=7)
    ax.barh(indicies, [x for x, _ in weights_and_names], 0.8, alpha=0.4, color="b")
    plt.yticks(indicies + (0.8 / 2), [n for _, n in weights_and_names])
    # ax.plot([0, len(performance_list)*1], [oracle[1], oracle[1]], "k--")

    plt.ylabel("Feature")
    plt.xlabel("Weighting")
    plt.title("Comparison of the Weights for the Positive Class\nGiven to Each FeatureBy a Linear Classifier")
    plt.savefig(GRAPH_SAVE_DIR + "feature_weight_comparison.png")
    plt.show()

    # ====> Weights Without AbstractROUGE <====
    with open(BASE_DIR + "/Data/Generated_Data/Weights/feature_weights_no_abs_rouge.npy", "rb") as f:
        feat_weights = np.load(f)

    weights_pos = feat_weights[:, 1]
    names = ["TF-IDF", "Document TF-IDF", "Keyphrase Score", "Title Score", "Numeric Count",
             "Sent Length", "Section"]

    weights_and_names = zip(weights_pos, names)
    print(weights_and_names)
    wait()
    weights_and_names = [x for x in reversed(sorted(weights_and_names, key=itemgetter(0)))]

    # Plot the result
    indicies = np.arange(0, len(weights_pos) * 1, 1)
    fig, ax = plt.subplots()
    plt.tight_layout(pad=7)
    ax.barh(indicies, [x for x, _ in weights_and_names], 0.8, alpha=0.4, color="b")
    plt.yticks(indicies + (0.8 / 2), [n for _, n in weights_and_names])
    # ax.plot([0, len(performance_list)*1], [oracle[1], oracle[1]], "k--")

    plt.ylabel("Feature")
    plt.xlabel("Weighting")
    plt.title("Comparison of the Weights for the Positive Class\nGiven to Each Feature By a Linear Classifier Without AbstractROUGE")
    plt.savefig(GRAPH_SAVE_DIR + "feature_weight_comparison_no_abs_rouge.png")
    plt.show()

    sys.exit()






