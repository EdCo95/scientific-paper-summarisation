# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from Summariser import Summariser
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE, SECTION_TITLES
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/OracleROUGE/"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================

class OracleSummariser(Summariser):

    def __init__(self, visualise):
        """
        Oracle summariser is not an actual, usable summariser. It extracts the best sentences from the paper possible
        by comparing them to the gold summaries. It represents the high-water mark in what ROUGE score it is possible
        for a summariser to achieve.
        """
        self.summary_length = 10
        self.r = Rouge()
        self.visualise = visualise


    def summarise(self, filename, name):
        """
        Generates a summary of the paper.
        :param filename: the name of the file to summaries
        :param name: the name of the file that will be written
        :return: a sumamry of the paper.
        """

        paper = self.prepare_paper(filename)

        highlights = paper["HIGHLIGHTS"][0]
        highlights = [" ".join(x) for x in highlights]

        # We don't want to make any predictions for the Abstract or Highlights as these are already summaries.
        sections_to_predict_for = []
        for section, text in paper.iteritems():
            if section != "ABSTRACT" and section != "HIGHLIGHTS":
                sections_to_predict_for.append((section, text[0], text[1]))

        # Sorts the sections according to the order in which they appear in the paper.
        sorted_sections_to_predict_for = sorted(sections_to_predict_for, key=itemgetter(-1))

        # Creates a list of the sentences in the paper in the correct order. Each item in the list is formed of
        # a list of words making up the sentence.
        sentence_list = []
        for sentence_section, sentence_text, section_position_in_paper in sorted_sections_to_predict_for:
            section_sentences = sentence_text
            sentence_list.append([sentence_section])
            for sentence in section_sentences:
                sentence_list.append(sentence)

        # Use the model to predict if each sentence is a summary sentence or not.
        predictions = []
        for sentence_text in sentence_list:
            if sentence_text[0].lower() in SECTION_TITLES and sentence_text[0].isupper():
                predictions.append(0)
            else:
                oracle_rouge = useful_functions.compute_rouge_abstract_score(sentence_text, highlights)
                predictions.append(oracle_rouge)

        # Produces a list of the form [sentence_text, sentence_index_in_paper, sentence tf_idf score]
        sentence_list_with_predictions = zip(sentence_list, range(len(sentence_list)), predictions)

        # Sort according to likelihood of being a summary
        sorted_predictions = reversed(sorted(sentence_list_with_predictions, key=itemgetter(-1)))
        sorted_predictions = [x for x in sorted_predictions]

        if not self.visualise:
            # Slice the top few sentences to form the summary sentences
            summary_sents = sorted_predictions[0:self.summary_length]
        else:
            summary_sents = sorted_predictions

        # Order sumamry sentences according to the order they appear in the paper
        ordered_summary = sorted(summary_sents, key=itemgetter(-2))

        if self.visualise:
            return ordered_summary

        # Print the summary
        summary = []

        for item in ordered_summary:
            sentence_position = item[1]
            sentence = " ".join(item[0])
            summary.append((sentence, sentence_position))

        useful_functions.write_summary(SUMMARY_WRITE_LOC, summary, name)

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
        paper = useful_functions.read_in_paper(filename, sentences_as_lists=True, preserve_order=True)
        return paper


if __name__ == "__main__":
    # Paper One: S0168874X14001395.txt
    # Paper Two: S0141938215300044.txt
    # Paper Three: S0142694X15000423.txt
    summ = OracleSummariser(False)
    summ.summarise("S0142694X15000423.txt", "1.txt")
    wait()
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
            summ.summarise(filename, filename.strip(".txt"))

        count += 1