# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
from collections import defaultdict
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE, STOPWORDS, NUMBER_OF_PAPERS,\
    PAPER_BAG_OF_WORDS_LOC, GLOBAL_COUNT_LOC
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

NAME = "OracleVis"

def heatmap(value):
    h = (1.0 - value) * 240
    return "hsla(" + str(h) + ", 100%, 50%, 0.5)"


def calculate_tf_idf(sentence, global_count_of_papers_words_occur_in, paper_bag_of_words):
    """
    Calculates the tf-idf score for a sentence based on all of the papers.
    :param sentence: the sentence to calculate the score for, as a list of words
    :param global_count_of_papers_words_occur_in: a dictionary of the form (word: number of papers the word occurs in)
    :param paper_bag_of_words: the bag of words representation for a paper
    :return: the tf-idf score of the sentence
    """
    bag_of_words = paper_bag_of_words

    sentence_tf_idf = 0

    length = 0

    tf_idfs = []

    for word in sentence:

        # Get the number of documents containing this word - the idf denominator (1 is added to prevent division by 0)
        docs_containing_word = global_count_of_papers_words_occur_in[word] + 1

        # Count of word in this paper - the tf score
        count_word = bag_of_words[word]

        idf = np.log(NUMBER_OF_PAPERS / docs_containing_word)

        word_tf_idf = count_word * idf

        tf_idfs.append(word_tf_idf)

    return [x for x in zip(sentence, tf_idfs)]

with open(BASE_DIR + "/Visualisations/base_html.txt", "rb") as f:
    html = f.readlines()

html.append("<body>\n")
html.append("<div class=\"container\">\n")
html.append("<div id=\"title\" class=\"text\">")

filename = "S0920548915000744.txt"
paper = useful_functions.read_in_paper(filename, sentences_as_lists=True, preserve_order=True)

html.append("<h1>" + " ".join(paper["MAIN-TITLE"][0][0]) + "</h1>")
html.append("</div>")
html.append("<div id=\"gold\" class=\"text\">")
html.append("<h2>Human Written Summary</h2>")
html.append("<hr>")
html.append("<br>")
html.append("<p>")

highlights = paper["HIGHLIGHTS"]
print("Reading stuff...")
bag_of_words = defaultdict(float)
for key, val in paper.iteritems():
    sents = val[0]
    for sent in sents:
        for word in sent:
            bag_of_words[word] += 1.0
global_paper_count = useful_functions.load_pickled_object(GLOBAL_COUNT_LOC)
print("Done")

sents_and_scores = []
for sentence in highlights[0]:
    sents_and_scores.append(calculate_tf_idf(sentence, global_paper_count, bag_of_words))

max_tf_idf = -1
for sentence in sents_and_scores:
    for word, score in sentence:
        if score > max_tf_idf:
            max_tf_idf = score

highlights_and_scores = []
for sentence in sents_and_scores:
    new_sent = []
    for word, score in sentence:
        new_score = score / max_tf_idf
        new_sent.append((word, new_score))
    highlights_and_scores.append(new_sent)

for sent in highlights_and_scores:
    for word, score in sent:
        html.append("<span style=\"background-color:" + heatmap(score) + "\">&nbsp" + word + " </span>")
    html.append("<br><br>")

html.append("</p>")

html.append("</div>")

html.append("<div id=\"paper\" class=\"text\">")
html.append("<h2>Full Paper</h2>")
html.append("<hr>")
html.append("<br>")

section_titles = useful_functions.read_section_titles()

paper_as_list = []
for title, section in paper.iteritems():
    paper_as_list.append((title, section[0], section[1]))

paper_as_list = sorted(paper_as_list, key=itemgetter(2))

max_tf_idf = -1

for _, sentences, _ in paper_as_list:
    for sentence in sentences:
        sent_and_scores = calculate_tf_idf(sentence, global_paper_count, bag_of_words)
        for word, score in sent_and_scores:
            if score > max_tf_idf:
                max_tf_idf = score

for section_title, sentences, _ in paper_as_list:
    if section_title != "HIGHLIGHTS" and section_title != "MAIN-TITLE":
        html.append("<h3>" + section_title + "</h3>")
        html.append("<p>")
        for sentence in sentences:
            sent_and_scores = calculate_tf_idf(sentence, global_paper_count, bag_of_words)
            for word, score in sent_and_scores:
                score = score / max_tf_idf
                html.append("<span style=\"background-color:" + heatmap(score) + "\">&nbsp" + word + " </span>")
        html.append("</p>")
        html.append("<br><br>")

html.append("</div>")
html.append("</body>")
html.append("</html>")

with open(BASE_DIR + "/Visualisations/" + NAME + "_index.html", "wb") as f:
    for item in html:
        f.write(item)