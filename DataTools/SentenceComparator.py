# Class containing methods to compare two sentences to see if they are the same sentence.

# ======== PROJECT CONFIGURATION IMPORTS ========
from __future__ import print_function, division
import string
from nltk.tokenize import word_tokenize

# ===============================================

class SentenceComparator:
    """Provides methods to compare sentences to eachother to see if they are the same sentence."""

    def __init__(self):
        self.BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"

    def removeCommonWords(self, sentence, common_words, tokenized=False):
        """Takes a sentence and list of stopwords and removes the stopwords from the sentence."""
        if not tokenized:
            words = sentence.split(' ')
        else:
            words = sentence
        final_sentence = []

        for word in words:
            word = word.translate(string.maketrans("", ""), string.punctuation)
            word = word.lower()
            if word in common_words:
                continue
            else:
                final_sentence.append(word)

        return final_sentence

    def compare_sentences(self, sentence1, sentence2, stopwords, tokenized=False):
        """Compares sentence1 to sentence2 to tell if they are the same sentence. Returns 1 if they are,
           0 if they are not."""

        TOLERANCE = 10
        THRESHOLD = 0.7
        scores = []

        if not tokenized:
            len_1 = float(len(word_tokenize(sentence1)))
            len_2 = float(len(word_tokenize(sentence2)))
        else:
            len_1 = float(len(sentence1))
            len_2 = float(len(sentence2))

        tolerance_upper = len_1 * (1 + (TOLERANCE / 100))
        tolerance_lower = len_1 * (1 - (TOLERANCE / 100))

        if tolerance_lower < len_2 < tolerance_upper:
            scores.append(1)
        else:
            scores.append(0)

        trim_1 = self.removeCommonWords(sentence1, stopwords, tokenized=tokenized)
        trim_2 = set(self.removeCommonWords(sentence2, stopwords, tokenized=tokenized))

        common_words_score = 0

        for word in trim_1:
            if word in trim_2:
                common_words_score += 1

        if len(trim_1) == 0:
            score_ratio = common_words_score / 1
        else:
            score_ratio = common_words_score / len(trim_1)

        scores.append(score_ratio)

        if scores[0] == 1 and scores[1] >= THRESHOLD:
            return 1
        else:
            return 0
