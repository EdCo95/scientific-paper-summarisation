# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
from collections import defaultdict
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from Dev.Summarisers.OracleSummariser import OracleSummariser
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE, STOPWORDS, NUMBER_OF_PAPERS,\
    PAPER_BAG_OF_WORDS_LOC, GLOBAL_COUNT_LOC, SECTION_TITLES
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

NAME = "OracleVis"

def heatmap(value):
    h = (1.0 - value) * 240
    return "hsla(" + str(h) + ", 100%, 50%, 0.5)"

with open(BASE_DIR + "/Visualisations/base_html.txt", "rb") as f:
    html = f.readlines()

html.append("<body>\n")
html.append("<div class=\"container\">\n")
html.append("<div id=\"title\" class=\"text\">")

filename = "S0920548915000744.txt"
filename = "S1568494614006000.txt"
filename = "S0140366416300068.txt"
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

for item in highlights[0]:
    html.append(" ".join(item))
    html.append("<br><br>")

html.append("</p>")

html.append("</div>")

html.append("<div id=\"paper\" class=\"text\">")
html.append("<h2>Full Paper</h2>")
html.append("<hr>")
html.append("<br>")

summ = OracleSummariser(True)
sents_and_scores = summ.summarise(filename, filename)

p_open = False

for sentence, pos, prob in sents_and_scores:

    if sentence[0] == "":
        continue

    if sentence[0].lower() in SECTION_TITLES and sentence[0].isupper():

        if p_open:
            html.append("</p>")
            p_open = False

        html.append("<br><br>")
        html.append("<h3>" + " ".join(sentence) + "</h3>")

    else:

        if not p_open:
            html.append("<p>")
            p_open = True

        if prob > 0.8:
            html.append("<span style=\"background-color:" + heatmap(prob) + "\">&nbsp" + " ".join(sentence) + " </span>")
        else:
            html.append(" ".join(sentence))

        html.append("<br><br>")

if p_open:
   html.append("</p>")
   p_open = False

html.append("</div>")
html.append("</body>")
html.append("</html>")

with open(BASE_DIR + "/Visualisations/" + NAME + "_index.html", "wb") as f:
    for item in html:
        f.write(item)