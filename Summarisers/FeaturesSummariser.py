# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
import tensorflow as tf
from scipy import spatial
from Summariser import Summariser
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE
from Dev.DataTools.DataPreprocessing.AbstractNetPreprocessor import AbstractNetPreprocessor
from Dev.Models.FeaturesClassifier.features_mlp import graph, SAVE_PATH, NUM_FEATURES
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/FeaturesSummariser/"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================

class FeaturesSummariser(Summariser):
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
        self.r = Rouge()
        self.preprocessor = AbstractNetPreprocessor()
        self.computation_graph = graph()
        self.features_input = self.computation_graph["features_input"]
        self.prediction_probs = self.computation_graph["prediction_probs"]
        self.similarity_threshold = 0.75


    def summarise(self, filename):
        """
        Generates a summary of the paper.
        :param filename: the name of the file to summaries
        :return: a sumamry of the paper.
        """

        # Each item has form (sentence, sentence_vector, abstract_vector, features)
        paper = self.prepare_paper(filename)

        # ========> Code from here on is summariser specific <========

        with tf.Session() as sess:

            # Initialise all variables
            sess.run(tf.global_variables_initializer())

            # Saving object
            saver = tf.train.Saver()

            # Restore the saved model
            saver.restore(sess, SAVE_PATH)

            # Stores sentences, the probability of them being good summaries and their position in the paper
            sentences_and_summary_probs = []

            # Create a matrix of all of the features in the paper so that we can predict summary probabilities for
            # the whole paper at once
            num_sents = len(paper)
            feed_feats = np.zeros((num_sents, NUM_FEATURES), dtype=np.float32)
            for i, item in enumerate(paper):
                feed_feats[i, :] = item[3]

            # Predict how good a summary each sentence is using the computation graph
            probs = sess.run(self.prediction_probs, feed_dict={self.features_input: feed_feats})

            # Store the sentences and probabilities in a list to be sorted
            for i in range(num_sents):
                sentence = paper[i][0]
                sentence_vec = paper[i][1]
                prob = probs[i][1]
                sentences_and_summary_probs.append((sentence, sentence_vec, prob, i))

            # This list is now sorted by the probability of the sentence being a good summary sentence
            sentences_and_summary_probs = [x for x in reversed(sorted(sentences_and_summary_probs, key=itemgetter(2)))]

            summary = sentences_and_summary_probs[0:self.summary_length]

            # Order summary sentences according to the order they appear in the paper
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

    def prepare_paper(self, filename):
        """
        Prepares the paper for summarisation.
        :return: The paper in a form suitable for summarisation
        """
        paper = self.preprocessor.prepare_for_summarisation(filename)
        return paper


if __name__ == "__main__":
    # Paper One: S0168874X14001395.txt
    # Paper Two: S0141938215300044.txt
    # Paper Three: S0142694X15000423.txt
    summ = FeaturesSummariser()
    #summ.summarise("S0142694X15000423.txt")

    count = 0
    for filename in os.listdir(PAPER_SOURCE):
        if count > 175:
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
