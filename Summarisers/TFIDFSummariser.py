# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from Summariser import Summariser
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE, PAPER_BAG_OF_WORDS_LOC, \
    KEYPHRASES_LOC, GLOBAL_COUNT_LOC
from operator import itemgetter
from sklearn import linear_model

SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/TFIDF/"
WORD2VEC_FEATURE_NUMS = 100
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================


class TFIDFSummariser(Summariser):
    """
    Implements a logistic regression summariser that used a logistic regression classifier to tell if sentences are
    summary sentences or not.
    """

    def __init__(self):
        """
        Double Dataset based Summariser.
        """
        self.summary_length = 10
        print("Reading bags of words...")
        t = time.time()
        self.paper_bags_of_words = useful_functions.load_pickled_object(PAPER_BAG_OF_WORDS_LOC)
        print("Done, took ", time.time() - t, " seconds.")

        # Dictionary holding the keyphrases for each paper
        print("Reading keyphrases...")
        t = time.time()
        self.keyphrases = useful_functions.load_pickled_object(KEYPHRASES_LOC)
        print("Done, took ", time.time() - t, " seconds.")

        # Dictionary which contains the counts of the number of different papers that a word occurs in
        print("Reading global count...")
        t = time.time()
        self.global_paper_count = useful_functions.load_pickled_object(GLOBAL_COUNT_LOC)
        print("Done, took ", time.time() - t, " seconds.")

        # The word2vec model to represent the first word and first and second words of the sentence
        print("Loading word2vec...")
        t = time.time()
        self.word2vec_model = useful_functions.load_word2vec()
        self.vocab = set(self.word2vec_model.index2word)
        print("Done, took ", time.time() - t, " seconds.")

    def summarise(self, filename):
        """
        Generates a summary of the paper.
        :param filename: the name of the file to summaries
        :param name: the name of the file that will be written
        :return: a sumamry of the paper.
        """

        paper = self.prepare_paper(filename)

        bag_of_words = self.paper_bags_of_words[filename]
        paper_keyphrases = self.keyphrases[filename]

        # We don't want to make any predictions for the Abstract or Highlights as these are already summaries.
        sections_to_predict_for = []
        for section, text in paper.iteritems():
            if section != "ABSTRACT" and section != "HIGHLIGHTS":
                sections_to_predict_for.append(text)

        # Sorts the sections according to the order in which they appear in the paper.
        sorted_sections_to_predict_for = sorted(sections_to_predict_for, key=itemgetter(1))

        # Creates a list of the sentences in the paper in the correct order. Each item in the list is formed of
        # a list of words making up the sentence.
        sentence_list = []
        for sentence_text, section_position_in_paper in sorted_sections_to_predict_for:
            section_sentences = sentence_text
            for sentence in section_sentences:
                sentence_list.append(sentence)

        # Use the model to predict if each sentence is a summary sentence or not.
        predictions = []
        for sentence_text in sentence_list:

            tf_idf = useful_functions.calculate_tf_idf(sentence_text, self.global_paper_count, bag_of_words)
            predictions.append(tf_idf)

        # Produces a list of the form [sentence_text, sentence_index_in_paper, sentence tf_idf score]
        sentence_list_with_predictions = zip(sentence_list, range(len(sentence_list)), predictions)

        # Sort according to likelihood of being a summary
        sorted_predictions = reversed(sorted(sentence_list_with_predictions, key=itemgetter(-1)))
        sorted_predictions = [x for x in sorted_predictions]

        # Slice the top few sentences to form the summary sentences
        summary_sents = sorted_predictions[0:self.summary_length]

        # Order sumamry sentences according to the order they appear in the paper
        ordered_summary = sorted(summary_sents, key=itemgetter(-2))

        # Print the summary
        summary = []

        for item in ordered_summary:
            sentence_position = item[1]
            sentence = " ".join(item[0])
            summary.append((sentence, sentence_position))

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
        word2vec_model = useful_functions.load_word2vec()
        paper = useful_functions.read_in_paper(filename, sentences_as_lists=True, preserve_order=True)
        return paper


if __name__ == "__main__":
    # Paper One: S0168874X14001395.txt
    # Paper Two: S0141938215300044.txt
    # Paper Three: S0142694X15000423.txt
    summ = TFIDFSummariser()
    #summ.summarise("S0142694X15000423.txt")

    count = 0
    for filename in os.listdir(PAPER_SOURCE):
        if count > 150:
            break
        if filename.endswith(".txt") and count > 0:

            # We need to write the highlights as a gold summary with the same name as the generated summary.
            highlights = useful_functions.read_in_paper(filename, True)["HIGHLIGHTS"]
            useful_functions.write_gold(SUMMARY_WRITE_LOC, highlights, filename)

            # Display a loading bar of progress
            useful_functions.loading_bar(LOADING_SECTION_SIZE, count, NUMBER_OF_PAPERS)

            # Generate and write a summary
            summ.summarise(filename)

        count += 1