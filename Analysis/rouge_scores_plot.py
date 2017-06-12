# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import pickle
import csv
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
from sys import exit

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

    # Models and performance list
    performance_list = []

    # Standard Deviation List
    sd_list = []

    # Load OracleROUGE ROUGE data
    name = "OracleROUGE"
    oracle_dict, oracle_list = load_scores(name)
    avg_oracle_score = np.mean([x for _, x in oracle_list])
    performance_list.append(("Oracle", avg_oracle_score))

    # Load features ROUGE data
    name = "FeaturesSummariser"
    feats_dict, feats_list = load_scores(name)
    avg_feats_score = np.mean([x for _, x in feats_list])
    std_feats_score = np.std([x for _, x in feats_list])
    performance_list.append(("FNet", (avg_feats_score / avg_oracle_score) * 100))

    # Load SummariserNet ROUGE data
    name = "SummariserNetSummariser"
    summNet_dict, summNet_list = load_scores(name)
    avg_summNet_score = np.mean([x for _, x in summNet_list])
    performance_list.append(("SAFNet", (avg_summNet_score / avg_oracle_score) * 100))

    # Load SummariserNet V2 ROUGE data
    name = "SummariserNetV2Summariser"
    summNetV2_dict, summNetV2_list = load_scores(name)
    avg_summNetV2_score = np.mean([x for _, x in summNetV2_list])
    std_summNetV2_score = np.std([x for _, x in summNetV2_list])
    performance_list.append(("SFNet", (avg_summNetV2_score / avg_oracle_score) * 100))

    # Load LSTM ROUGE data
    name = "LSTMSummariser"
    lstm_dict, lstm_list = load_scores(name)
    avg_lstm_score = np.mean([x for _, x in lstm_list])
    std_lstm_score = np.std([x for _, x in lstm_list])
    performance_list.append(("LSTM", (avg_lstm_score / avg_oracle_score) * 100))

    # Load AbstractROUGE ROUGE data
    name = "AbstractROUGE"
    abs_rouge_dict, abs_rouge_list = load_scores(name)
    avg_abs_rouge_score = np.mean([x for _, x in abs_rouge_list])
    std_abs_rouge_score = np.std([x for _, x in abs_rouge_list])
    performance_list.append(("Abs Rouge", (avg_abs_rouge_score / avg_oracle_score) * 100))

    # Load TFIDF ROUGE data
    name = "TFIDF"
    tf_idf_dict, tf_idf_list = load_scores(name)
    avg_tf_idf_score = np.mean([x for _, x in tf_idf_list])
    performance_list.append(("TFIDF", (avg_tf_idf_score / avg_oracle_score) * 100))

    # Load Ensemble ROUGE data
    name = "EnsembleSummariser"
    ensemble_dict, ensemble_list = load_scores(name)
    avg_ensemble_score = np.mean([x for _, x in ensemble_list])
    std_ensemble = np.std([x for _, x in ensemble_list])
    performance_list.append(("SAF+F Ens", (avg_ensemble_score / avg_oracle_score) * 100))

    # Load SummariserNet No Abstract Rouge ROUGE data
    name = "SummariserNetNoAbsRouge"
    snet_no_absrouge_dict, snet_no_absrouge_list = load_scores(name)
    avg_snet_no_absrouge_score = np.mean([x for _, x in snet_no_absrouge_list])
    #performance_list.append(("SNet No AbsRouge", (avg_snet_no_absrouge_score / avg_oracle_score) * 100))

    # Load SummariserNetV2 No Abstract Rouge ROUGE data
    name = "SummariserNetV2NoAbsRouge"
    snet2_no_absrouge_dict, snet2_no_absrouge_list = load_scores(name)
    avg_snet2_no_absrouge_score = np.mean([x for _, x in snet2_no_absrouge_list])
    # performance_list.append(("SNetV2 No AbsRouge", (avg_snet2_no_absrouge_score / avg_oracle_score) * 100))

    # Load Features No Abs Rouge ROUGE data
    name = "FeaturesNoAbsRougeSummariser"
    feats_no_absrouge_dict, feats_no_absrouge_list = load_scores(name)
    avg_feats_no_absrouge_score = np.mean([x for _, x in feats_no_absrouge_list])
    std_feats_no_absrouge_score = np.std([x for _, x in feats_no_absrouge_list])
    #performance_list.append(("Features No AbsRouge", (avg_feats_no_absrouge_score / avg_oracle_score) * 100))

    # Load TitleScore ROUGE data
    name = "TitleScoreSummariser"
    title_score_dict, title_score_list = load_scores(name)
    avg_title_score = np.mean([x for _, x in title_score_list])
    performance_list.append(("Title Score", (avg_title_score / avg_oracle_score) * 100))

    # Load KeyphraseScore ROUGE data
    name = "KeyphraseScoreSummariser"
    keyphrase_score_dict, keyphrase_score_list = load_scores(name)
    avg_keyphrase_score = np.mean([x for _, x in keyphrase_score_list])
    performance_list.append(("Keyphrase Score", (avg_keyphrase_score / avg_oracle_score) * 100))

    # Load DocTFIDF ROUGE data
    name = "DocTFIDF"
    doctfidf_dict, doctfidf_list = load_scores(name)
    avg_doctfidf_score = np.mean([x for _, x in doctfidf_list])
    performance_list.append(("DocTFIDF", (avg_doctfidf_score / avg_oracle_score) * 100))

    # Load Low Data Features ROUGE data
    name = "LowDataFeaturesSummariser"
    low_data_feats_dict, low_data_feats_list = load_scores(name)
    avg_low_data_feats_score = np.mean([x for _, x in low_data_feats_list])
    std_low_data_feats_score = np.std([x for _, x in low_data_feats_list])
    #performance_list.append(("Low Data Features", (avg_low_data_feats_score / avg_oracle_score) * 100))

    # Load Low Data SummNet ROUGE data
    name = "LowDataSummariserNetSummariser"
    low_data_summNet_dict, low_data_summNet_list = load_scores(name)
    avg_low_data_summNet_score = np.mean([x for _, x in low_data_summNet_list])
    #performance_list.append(("Low Data SummNet", (avg_low_data_summNet_score / avg_oracle_score) * 100))

    # Load Low Data SummNetV2 ROUGE data
    name = "LowDataSummariserNetV2Summariser"
    low_data_summNet2_dict, low_data_summNet2_list = load_scores(name)
    avg_low_data_summNet2_score = np.mean([x for _, x in low_data_summNet2_list])
    std_low_data_summNet2_score = np.std([x for _, x in low_data_summNet2_list])
    # performance_list.append(("Low Data SummNet`v2", (avg_low_data_summNet2_score / avg_oracle_score) * 100))

    # Load Word2Vec Summariser ROUGE data
    name = "Word2VecSummariser"
    word2vec_dict, word2vec_list = load_scores(name)
    avg_word2vec_score = np.mean([x for _, x in word2vec_list])
    performance_list.append(("Word2Vec", (avg_word2vec_score / avg_oracle_score) * 100))

    # Load Combined Summariser ROUGE data
    name = "CombinedSummariser"
    combined_dict, combined_list = load_scores(name)
    avg_combined_score = np.mean([x for _, x in combined_list])
    std_combined = np.std([x for _, x in combined_list])
    performance_list.append(("Word2VecAFNet", (avg_combined_score / avg_oracle_score) * 100))

    # Load EnsembleV2 ROUGE data
    name = "EnsembleV2Summariser"
    ensemble_v2_dict, ensemble_v2_list = load_scores(name)
    avg_ensemble_v2_score = np.mean([x for _, x in ensemble_v2_list])
    performance_list.append(("S+F Ens", (avg_ensemble_v2_score / avg_oracle_score) * 100))

    # ==================================================================================================================

    # BASELINE COMPARISONS

    # Load KLSummariser ROUGE data
    name = "KLSummariser"
    kl_dict, kl_list = load_scores(name)
    avg_kl_score = np.mean([x for _, x in kl_list])
    performance_list.append(("KLSumm", (avg_kl_score / avg_oracle_score) * 100))

    # Load LSASumm ROUGE data
    name = "LSASummariser"
    lsa_dict, lsa_list = load_scores(name)
    avg_lsa_score = np.mean([x for _, x in lsa_list])
    performance_list.append(("LSA", (avg_lsa_score / avg_oracle_score) * 100))

    # Load LexRank ROUGE data
    name = "LexRank"
    lexrank_dict, lexrank_list = load_scores(name)
    avg_lexrank_score = np.mean([x for _, x in lexrank_list])
    performance_list.append(("LexRank", (avg_lexrank_score / avg_oracle_score) * 100))

    # Load SumBasic ROUGE data
    name = "SumBasic"
    sumbasic_dict, sumbasic_list = load_scores(name)
    avg_sumbasic_score = np.mean([x for _, x in sumbasic_list])
    performance_list.append(("SumBasic", (avg_sumbasic_score / avg_oracle_score) * 100))

    # Load TextRank ROUGE data
    name = "TextRank"
    textrank_dict, textrank_list = load_scores(name)
    avg_textrank_score = np.mean([x for _, x in textrank_list])
    performance_list.append(("TextRank", (avg_textrank_score / avg_oracle_score) * 100))

    # ==================================================================================================================

    # Load test data performance stats
    test_data_performance = []
    with open(BASE_DIR + "/Data/Generated_Data/Test_Data_Performance/test_data_performance.csv", "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            test_data_performance.append((row[0], float(row[1])))

    test_data_performance = sorted(test_data_performance, key=itemgetter(1))

    # ====> Mean and SD comparisons <====
    print("Mean of Combined: ", avg_combined_score)
    print("Std of Combined: ", std_combined)
    print("\nMean of Ensemble: ", avg_ensemble_score)
    print("Std of Ensemble: ", std_ensemble)
    print("\nMean of SNetv2: ", avg_summNetV2_score)
    print("Std of SNetv2: ", std_summNetV2_score)

    print("\nMean of SNetv2 Low Data: ", avg_low_data_summNet2_score)
    print("Std of SNetv2 Low Data: ", std_low_data_summNet2_score)

    print("\nMean of LSTM: ", avg_lstm_score)
    print("Std of LSTM: ", std_lstm_score)
    print("\nMean of AbsRouge: ", avg_abs_rouge_score)
    print("Std of AbsRouge: ", std_abs_rouge_score)
    print("\nMean of Features Only: ", avg_feats_score)
    print("Std of Features Only: ", std_feats_score)
    print("\nMean of Low Data Features Only: ", avg_low_data_feats_score)
    print("Std of Low Data Features Only: ", std_low_data_feats_score)
    print("\nMean of Features No Abs Rouge: ", avg_feats_no_absrouge_score)
    print("Std of Features No Abs Rouge: ", std_feats_no_absrouge_score)

    wait()

    # ====> Comparison to Baseline <====
    full_data = [avg_ensemble_score, avg_kl_score, avg_lsa_score, avg_lexrank_score, avg_sumbasic_score,
                 avg_textrank_score]
    names = ["SAF+F Ens", "KLSum", "LSA", "LexRank", "SumBasic", "TextRank"]
    full = sorted(zip(full_data, names), key=itemgetter(0))
    indicies = np.arange(len(full_data))
    bar_width = 0.8

    fig, ax = plt.subplots()
    plt.tight_layout(pad=5)
    ax.barh(indicies, [x for x, _ in full], 0.8, alpha=0.4, color="r")
    plt.yticks(indicies + (0.8 / 2), [n for n in [x for _, x in full]])

    plt.xlabel("ROUGE-L Score")
    plt.title("Comparison of the ROUGE Scores of Different Models\nAnd Baseline Methods")
    plt.savefig(GRAPH_SAVE_DIR + "model_comparison_baselines.png")
    plt.show()

    exit()

    # ====> No AbstractROUGE Comparison <===
    full_data = [avg_feats_score, avg_summNet_score, avg_summNetV2_score]
    no_abs_rouge = [avg_feats_no_absrouge_score, avg_snet_no_absrouge_score, avg_snet2_no_absrouge_score]

    indicies = np.arange(len(full_data))
    bar_width = 0.35

    for item in zip(["FNet", "SAFNet", "SFNet"], full_data, no_abs_rouge):
        print(item)
    sys.exit()

    full_plot = plt.bar(indicies, full_data, bar_width, alpha=0.4, color="r", label="With AbstractROUGE")
    low_data_plot = plt.bar(indicies + bar_width, no_abs_rouge, bar_width, alpha=0.4, color="b",
                            label="Without AbstractROUGE")
    plt.xticks(indicies + bar_width, ["FNet", "SAFNet", "SFNet"])
    plt.ylabel("ROUGE Score (Max. = 1)")
    plt.xlabel("Model")
    plt.legend()
    plt.title("Comparison of Models With and Without\nthe AbstractROUGE Feature")
    plt.savefig(GRAPH_SAVE_DIR + "model_comparison_no_absrouge.png")
    plt.show()

    exit()

    # ====> Low Data Comparison <===
    full_data = [avg_feats_score, avg_summNet_score, avg_summNetV2_score]
    low_data = [avg_low_data_feats_score, avg_low_data_summNet_score, avg_low_data_summNet2_score]

    indicies = np.arange(len(full_data))
    bar_width = 0.4

    for item in zip(["FNet", "SAFNet", "SFNet"], full_data, low_data):
        print(item)
    sys.exit()

    full_plot = plt.bar(indicies, full_data, bar_width, alpha=0.4, color="r", label="Full Data")
    low_data_plot = plt.bar(indicies + bar_width, low_data, bar_width, alpha=0.4, color="b", label="Low Data")

    plt.xticks(indicies + bar_width, ["FNet", "SAFNet", "SFNet"])
    plt.ylabel("ROUGE Score (Max. = 1)")
    plt.xlabel("Model")
    plt.legend()
    plt.title("Comparison of Models when Trained on Full Dataset\nand Reduced Dataset")
    plt.savefig(GRAPH_SAVE_DIR + "model_comparison_low_data.png")
    plt.show()

    # ====> Main Comparison and Training Data <====

    new_performances = []

    for item in performance_list:
        if item[0] == "Oracle":
            continue
        else:
            new_performances.append((item[0], item[1] / 100))

    all_zipped = []
    for rouge_item in new_performances:
        rname = rouge_item[0]
        for test_item in test_data_performance:
            tname = test_item[0]
            if rname == tname:
                all_zipped.append((rname, rouge_item[1], test_item[1]))

    # Sort the performance list
    plot_list = [x for x in sorted(all_zipped, key=itemgetter(1))]

    for item in plot_list:
        print(item)
    wait()

    # Split performance list in two
    names = [x for x, _, _ in plot_list]
    rouges = [x for _, x, _ in plot_list]
    test_accs = [x for _, _, x in plot_list]

    width = 0.8

    # Plot the result
    indicies = np.arange(0, len(names) * 1, 1)
    fig, ax = plt.subplots()
    plt.tight_layout(pad=7)
    ax.barh(indicies, rouges, width, alpha=0.4, color="r", label="ROUGE Score as % of Oracle on CSPubSum Test")
    ax.barh([i+0.25*width for i in indicies], test_accs, (width * 0.5), alpha=0.4, color="b", label="CSPubSumExt Test Accuracy")
    plt.yticks(indicies + width/2, [n for n in names])
    # ax.plot([0, len(performance_list)*1], [oracle[1], oracle[1]], "k--")

    # plt.ylabel("Model")
    plt.xlabel("% of Oracle Score / Accuracy")
    plt.title("Comparison of the ROUGE Scores and SL Test Set Accuracy\nof Different Models As % of Oracle Score / Accuracy")
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15))
    plt.savefig(GRAPH_SAVE_DIR + "model_comparison_percent_oracle_and_accuracy.png")
    plt.show()

    exit()

    # ====> Test Set Performance <====
    accs = [x for _, x in test_data_performance]
    model_names = [x for x, _ in test_data_performance]
    indicies = np.arange(0, len(accs))

    fig, ax = plt.subplots()
    plt.tight_layout(pad=8)
    ax.barh(indicies, accs, 0.8, alpha=0.4, color="b")
    plt.yticks(indicies + (0.8 / 2), [n for n in model_names])

    plt.xlabel("Test Set Accuracy")
    plt.title("Comparison of Test Set Performance of Trainable Models")
    plt.savefig(GRAPH_SAVE_DIR + "test_performance_comparison.png")
    plt.show()

    # ====> Main Comparison <====

    oracle = performance_list.pop(0)

    # Sort the performance list
    performance_list = [x for x in sorted(performance_list, key=itemgetter(1))]

    # Split performance list in two
    names = [x for x, _ in performance_list]
    performances = [x for _, x in performance_list]

    # Plot the result
    indicies = np.arange(0, len(performance_list) * 1, 1)
    fig, ax = plt.subplots()
    plt.tight_layout(pad=7)
    ax.barh(indicies, performances, 0.8, alpha=0.4, color="b")
    plt.yticks(indicies + (0.8 / 2), [n for n in names])
    # ax.plot([0, len(performance_list)*1], [oracle[1], oracle[1]], "k--")

    # plt.ylabel("Model")
    plt.xlabel("% of Oracle Score")
    plt.title("Comparison of the ROUGE Scores of Different Models\nAs % of Oracle Score")
    plt.savefig(GRAPH_SAVE_DIR + "model_comparison_percent_oracle.png")
    plt.show()

    exit()

    # ====> LSTM vs Word2Vec Test Set Performance <====
    t = [x for x in reversed(test_data_performance) if x[0] == "LSTM" or x[0] == "Word2Vec"]
    accs = [x for _, x in t]
    model_names = [x for x, _ in t]
    indicies = np.arange(0, len(accs))

    fig, ax = plt.subplots()
    plt.tight_layout(pad=7)
    ax.bar(indicies, accs, 0.8, alpha=0.4, color="b")
    plt.xticks(indicies + (0.8 / 2), [n for n in model_names])

    plt.ylabel("Test Set Accuracy")
    plt.title("Comparison of Test Set Performance of LSTM and Word2Vec")
    plt.savefig(GRAPH_SAVE_DIR + "test_performance_LSTM_word2vec_comparison.png")
    plt.show()

    # ====> LSTM & Word2Vec Comparison <===
    data = [avg_lstm_score, avg_word2vec_score]

    print(data)
    wait()

    indicies = np.arange(len(data))
    bar_width = 0.8

    full_plot = plt.bar(indicies, data, bar_width, alpha=0.4, color="r")

    plt.xticks(indicies + bar_width / 2, ["LSTM", "Word2Vec MLP"])
    plt.ylabel("ROUGE Score (Max. = 1)")
    plt.xlabel("Model")
    plt.title("Comparison of an LSTM and Average Word Vector\nSummarisers")
    plt.savefig(GRAPH_SAVE_DIR + "model_comparison_lstm_word2vec.png")
    plt.show()







