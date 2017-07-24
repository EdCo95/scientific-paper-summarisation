# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
from collections import defaultdict
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from Dev.Summarisers.EnsembleSummariser import EnsembleSummariser
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, BASE_DIR, PAPER_SOURCE, STOPWORDS, NUMBER_OF_PAPERS,\
    PAPER_BAG_OF_WORDS_LOC, GLOBAL_COUNT_LOC, SECTION_TITLES
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

def heatmap(value):
    h = (1.0 - value) * 240
    return "hsla(" + str(h) + ", 100%, 50%, 0.5)"

def heatmap_simple(value):
    h = (1.0 - value) * 240
    return "hsla(0, 100%, 50%, 0.5)"

def opacitymap(value):
    a = value
    return "hsla(0, 100%, 50%, " + str(a) + ")"

with open(BASE_DIR + "/Visualisations/base_html.txt", "rb") as f:
    html = f.readlines()

html.append("<body>\n")
html.append("<div class=\"container\">\n")
html.append("<div id=\"title\" class=\"text\">")

# Ensemble scored highest on this paper

# Numbers in the summary for this paper
filename = "S0045790615002785.txt"

# The one the Oracle visualiser uses
filename = "S0140366416300068.txt"

# About web crawlers
filename = "S1568494613002974.txt"

#filename = "S1568494615007012.txt"

# Our paper - 1
filename1 = "our_paper.txt"

# Top ROUGE Score - 2
filename2 = "S0003687013000562.txt"

# Third ROUGE Score (second not good)  - 3
filename = "S0003687013002081.txt"

NAME = "3"

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

    if not val:
        continue

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
html.append("<h2>Generated Summary</h2>")
html.append("<hr>")
html.append("<br>")

summ = EnsembleSummariser()
sents_and_scores = summ.summarise(filename, visualise=True)

sorted_sents = sorted(sents_and_scores, key=itemgetter(2), reverse=True)
summary = sorted_sents[:10]
ordered_summary = sorted(summary, key=itemgetter(3))

html.append("<p>")
for sentence, _, prob, pos in ordered_summary:

    if sentence[0].lower() in SECTION_TITLES and sentence[0].isupper():
        continue

    print(sentence)
    print(prob)
    print()
    html.append(" ".join(sentence))
    html.append("<br><br>")

#wait()

html.append("</p>")

html.append("<h2>Other Examples</h2>")
html.append("<hr>")
html.append("<br>")
html.append("<a href=\"1_index.html\">Example One</a>")
html.append("<br>")
html.append("<a href=\"2_index.html\">Example Two</a>")
html.append("<br>")
html.append("<a href=\"3_index.html\">Example Three</a>")
html.append("<br><br>")

p_open = False
full_paper = True

if full_paper:

    html.append("<h2>Full Paper</h2>")
    html.append("<hr>")
    #html.append("<br>")

    for sentence, _, prob, pos in sents_and_scores:

        if len(sentence) < 3:
            print(sentence)

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

            if prob > 0.5:
                html.append("<span style=\"background-color:" + heatmap_simple(prob) + "\">&nbsp" + " ".join(sentence) + " </span>")
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