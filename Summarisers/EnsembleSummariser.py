# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from scipy import spatial
import tensorflow as tf
from Summariser import Summariser
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE
from Dev.DataTools.DataPreprocessing.AbstractNetPreprocessor import AbstractNetPreprocessor
from Dev.Models.SummariserNetClassifier.summariser_net import graph, sents2input, SAVE_PATH, NUM_FEATURES,\
    ABSTRACT_DIMENSION, WORD_DIMENSIONS, MAX_SENT_LEN
from Dev.Models.FeaturesClassifier import features_mlp
from Dev.Models.SummariserNetClassifier import summariser_net
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/EnsembleSummariser/"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================

class EnsembleSummariser(Summariser):
    """
    Implements a logistic regression summariser that used a logistic regression classifier to tell if sentences are
    summary sentences or not.
    """

    def __init__(self):
        """
        ROUGE based summariser. This compares each sentence in the paper to the abstract to see which ones make the best
        summaries for the abstract. It is assumed that these sentences will then also be good highlights for the paper.
        """
        self.summary_length = 10
        self.min_sent_len = 10
        self.r = Rouge()
        self.preprocessor = AbstractNetPreprocessor()

        # Hyperparameter to tune weight given to feature probability
        self.C = 0.40


    def summarise(self, filename, visualise=False):
        """
        Generates a summary of the paper.
        :param filename: the name of the file to summaries
        :param visualise: true if visualising output
        :return: a sumamry of the paper.
        """

        # Each item has form (sentence, sentence_vector, abstract_vector, features)
        paper = self.prepare_paper(filename, visualise=visualise)

        # ========> Code from here on is summariser specific <========

        # Stores sentences, the probability of them being good summaries and their position in the paper
        sentences_and_summary_probs = []

        # Summary according to features
        sentences_feat_summary_probs = []

        tf.reset_default_graph()
        computation_graph = summariser_net.graph()
        sentence_input = computation_graph["sentence_input"]
        abstract_input = computation_graph["abstract_input"]
        features_input = computation_graph["features_input"]
        seq_lens = computation_graph["sequence_lengths"]
        prediction_probs = computation_graph["raw_predictions"]
        keep_prob = computation_graph["keep_prob"]

        with tf.Session() as sess:

            # Initialise all variables
            sess.run(tf.global_variables_initializer())

            # Saving object
            saver = tf.train.Saver()

            # Restore the saved model
            saver.restore(sess, summariser_net.SAVE_PATH)

            # ----> Create the matrix for sentences for the LSTM <----
            sentence_list = []

            for sent, sent_vec, abs_vec, feats in paper:
                if len(sent) < MAX_SENT_LEN:
                    sentence_list.append(sent)
                else:
                    sentence_list.append(sent[0:MAX_SENT_LEN])


            # Number of sentences in the paper
            num_sents = len(sentence_list)

            # Get the matrix representation of the sentences
            sentence_matrix, sent_lens = sents2input(sentence_list, num_sents)

            # ----> Create the matrix for abstracts for the LSTM <----
            abstract_matrix = np.zeros((num_sents, ABSTRACT_DIMENSION), dtype=np.float32)

            i = 0
            for _, _, abs_vec, _ in paper:
                abstract_matrix[i, :] = abs_vec
                i += 1

            # ----> Create the matrix of features for the LSTM <----
            feature_matrix = np.zeros((num_sents, NUM_FEATURES), dtype=np.float32)

            i = 0
            for _, _, _, feat in paper:
                feature_matrix[i, :] = feat
                i += 1

            # Create the feed_dict
            feed_dict = {
                sentence_input: sentence_matrix,
                abstract_input: abstract_matrix,
                features_input: feature_matrix,
                seq_lens: sent_lens,
                keep_prob: 1
            }

            # Predict how good a summary each sentence is using the computation graph
            probs = sess.run(prediction_probs, feed_dict=feed_dict)

            # Store the sentences and probabilities in a list to be sorted
            for i in range(num_sents):
                sentence = paper[i][0]
                sentence_vec = paper[i][1]
                prob = probs[i][1]
                sentences_and_summary_probs.append((sentence, sentence_vec, prob, i))

        tf.reset_default_graph()
        features_graph = features_mlp.graph()
        features_classifier_input = features_graph["features_input"]
        features_prediction_probs = features_graph["prediction_probs"]
        with tf.Session() as sess:

            # Initialise all variables
            sess.run(tf.global_variables_initializer())

            # Saving object
            saver = tf.train.Saver()

            # ====> Run the second graph <====
            saver.restore(sess, features_mlp.SAVE_PATH)

            # Predict how good a summary each sentence is using the computation graph
            probs = sess.run(features_prediction_probs, feed_dict={features_classifier_input: feature_matrix})

            # Store the sentences and probabilities in a list to be sorted
            for i in range(num_sents):
                sentence = paper[i][0]
                sentence_vec = paper[i][1]
                prob = probs[i][1]
                sentences_feat_summary_probs.append((sentence, sentence_vec, prob, i))

        # ====> Combine the results <====

        # This list is now sorted by the probability of the sentence being a good summary sentence
        #sentences_and_summary_probs = [x for x in reversed(sorted(sentences_and_summary_probs, key=itemgetter(2)))]

        # Sort features list in probability order
        #sentences_feat_summary_probs = [x for x in reversed(sorted(sentences_feat_summary_probs, key=itemgetter(2)))]

        summary = []
        sents_already_added = set()

        # ====> Attempt Four <====
        final_sents_probs = []

        for item in zip(sentences_feat_summary_probs, sentences_and_summary_probs):
            prob_summNet = item[1][2] * (1 - self.C)
            prob_Features = item[0][2] * (1 + self.C)
            avg_prob = (prob_summNet + prob_Features) / 2
            final_sents_probs.append((item[0][0], item[0][1], avg_prob, item[0][3]))

        final_sents_probs = [x for x in reversed(sorted(final_sents_probs, key=itemgetter(2)))]
        final_sents_probs = sorted(final_sents_probs, key=itemgetter(-1))

        if visualise:
            return final_sents_probs

        #summary = final_sents_probs[0:self.summary_length]

        """
        # ====> Attempt Three <====
        # Take summary sentences from features
        summary = sentences_feat_summary_probs[0:self.summary_length]
        for item in summary:
            sents_already_added.add(item[3])

        # Add ones from summary net if it's sure of them and they aren't there already
        max_additional = 5
        count_additional = 0
        for item in sentences_and_summary_probs:
            if count_additional > max_additional:
                break
            if item[3] not in sents_already_added and item[2] > 0.95:
                summary.append(item)
                sents_already_added.add(item[3])
                count_additional += 1
        """
        """
        # ====> Attempt Two <====
        i = 0
        while len(summary) < self.summary_length:

            if i >= len(sentences_feat_summary_probs) and i >= len(sentences_and_summary_probs):
                break

            feats = sentences_feat_summary_probs[i]
            summNet = sentences_and_summary_probs[i]

            feats_prob = feats[2]
            summNet_prob = summNet[2]

            if feats_prob >= summNet_prob and feats[3] not in sents_already_added:
                summary.append(feats)
                sents_already_added.add(feats[3])
            elif summNet_prob > feats_prob and summNet[3] not in sents_already_added:
                summary.append(summNet)
                sents_already_added.add(summNet[3])

            i += 1
        """
        """
        # ====> Attempt One <====
        # True to select a summary sentence from summ_net, false to select from features
        summ_net = True
        for i in range(num_sents):

            if len(summary) >= self.summary_length \
                    or len(sentences_and_summary_probs) <= 0 \
                    or len(sentences_feat_summary_probs) <= 0:
                break

            added = False

            if summ_net:

                while not added:

                    if len(sentences_and_summary_probs) <= 0:
                        break

                    highest_prob = sentences_and_summary_probs.pop(0)
                    if highest_prob[3] in sents_already_added or len(highest_prob[0]) < self.min_sent_len:
                        continue
                    else:
                        summary.append(highest_prob)
                        sents_already_added.add(highest_prob[3])
                        added = True

                summ_net = False

            else:

                while not added:

                    if len(sentences_feat_summary_probs) <= 0:
                        break

                    highest_prob = sentences_feat_summary_probs.pop(0)
                    if highest_prob[3] in sents_already_added or len(highest_prob[0]) < self.min_sent_len:
                        continue
                    else:
                        summary.append(highest_prob)
                        sents_already_added.add(highest_prob[3])
                        added = True

                summ_net = True
        """

        # Order sumamry sentences according to the order they appear in the paper
        ordered_summary = sorted(summary, key=itemgetter(-1))

        # Print the summary
        summary = []

        for sentence, sentence_vec, prob, pos in ordered_summary:
            sentence = " ".join(sentence)
            summary.append((sentence, pos))

        useful_functions.write_summary(SUMMARY_WRITE_LOC, summary, filename.strip(".txt"))

        for sentence in summary:
            print(sentence)
            print()


    def load_model(self):
        """
        Loads the classification model
        :return: the classification model
        """
        pass

    def prepare_paper(self, filename, visualise=False):
        """
        Prepares the paper for summarisation.
        :return: The paper in a form suitable for summarisation
        """
        paper = self.preprocessor.prepare_for_summarisation(filename, visualise=visualise)
        return paper


if __name__ == "__main__":
    # Paper One: S0168874X14001395.txt
    # Paper Two: S0141938215300044.txt
    # Paper Three: S0142694X15000423.txt
    summ = EnsembleSummariser()

    summ.summarise("our_paper.txt")

    wait()

    #summ.summarise("S0142694X15000423.txt")
    #summ.summarise("S0142694X15000423.txt")
    #sys.exit()
    count = 0
    for filename in os.listdir(PAPER_SOURCE):
        if count > 150:
            break
        if filename.endswith(".txt"):

            # We need to write the highlights as a gold summary with the same name as the generated summary.
            highlights = useful_functions.read_in_paper(filename, True)["HIGHLIGHTS"]
            useful_functions.write_gold(SUMMARY_WRITE_LOC, highlights, filename)

            # Display a loading bar of progress
            useful_functions.loading_bar(LOADING_SECTION_SIZE, count, NUMBER_OF_PAPERS)

            # Generate and write a summary
            summ.summarise(filename)

        count += 1
