# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division, absolute_import
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
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.sum_basic import SumBasicSummarizer

SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/SumBasic/"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================

class SumBasicWrapperSummariser(Summariser):

    def __init__(self):
        """
        Oracle summariser is not an actual, usable summariser. It extracts the best sentences from the paper possible
        by comparing them to the gold summaries. It represents the high-water mark in what ROUGE score it is possible
        for a summariser to achieve.
        """
        self.summary_length = 10
        self.summariser = SumBasicSummarizer()


    def summarise(self, filename):
        """
        Generates a summary of the paper.
        :param filename: the name of the file to summaries
        :param name: the name of the file that will be written
        :return: a sumamry of the paper.
        """

        paper = self.prepare_paper(filename)

        parser = PlaintextParser.from_string(paper, Tokenizer("english"))

        summary = self.summariser(parser.document, self.summary_length)

        # The "1" is only added her to stop the summary breaking the save function - it's a bit of an ungainly hack
        summary = [(unicode(x), 1) for x in summary]

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
        paper_loc = PAPER_SOURCE + filename
        with open(paper_loc, "rb") as f:

            plaintext_paper = []

            for line in f.readlines():
                udata = line.decode("utf-8")
                new_line = udata.encode("ascii", "ignore")
                plaintext_paper.append(new_line)

        plaintext_paper = "".join(plaintext_paper)
        return plaintext_paper


if __name__ == "__main__":
    # Paper One: S0168874X14001395.txt
    # Paper Two: S0141938215300044.txt
    # Paper Three: S0142694X15000423.txt
    summ = SumBasicWrapperSummariser()
    #summ.summarise("S0142694X15000423.txt")
    #wait()
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