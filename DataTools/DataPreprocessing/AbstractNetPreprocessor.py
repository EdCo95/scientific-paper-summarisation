from __future__ import print_function, division
import os
import dill
import pickle
import sys
import random
import time
from multiprocessing.dummy import Pool as ThreadPool
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from operator import itemgetter
from Dev.DataTools.Reader import Reader
from Dev.DataTools.DataPreprocessing.DataPreprocessor import DataPreprocessor
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import wait, printlist, printdict, num2onehot, BASE_DIR, PAPER_SOURCE,\
    HIGHLIGHT_SOURCE, PAPER_BAG_OF_WORDS_LOC, KEYPHRASES_LOC, GLOBAL_COUNT_LOC, HIGHLIGHT, ABSTRACT, INTRODUCTION,\
    RESULT_DISCUSSION, METHOD, CONCLUSION, OTHER, STOPWORDS
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.LSTM_preproc.vocab import Vocab
from Dev.DataTools.LSTM_preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from Dev.DataTools.LSTM_preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import numpy as np


class AbstractNetPreprocessor(DataPreprocessor):

    def __init__(self):
        """
        Preprocessess data into a form suitable for use with networks that use the abstract vector as part of their
        input for classification.
        """

        # The number of summary sentences to extract from the paper as training data
        self.num_summary = 20

        # The number of papers to process and the loading section size which will be used to print a loading bar as
        # the papers are processed
        self.number_of_papers = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
        self.loading_section_size = self.number_of_papers / 30

        # Number of classes to classify into
        self.num_classes = 2

        # A thread pool for parallel processing of dats
        self.pool2 = ThreadPool(4)

        # Load the word2vec model
        print("Reading word2vec...")
        t = time.time()
        self.word2vec = useful_functions.load_word2vec()
        self.vocab = set(self.word2vec.index2word)
        print("Done, took ", time.time() - t, " seconds.")

        # Dictionary which contains bag of words representations for every paper
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

        # Running start time to measure running time
        self.start_time = time.time()

    def paper2orderedlist(self, filename):
        """
        Performs the first task necessary to summarise a paper: turning it into an ordered list of sentences which
        doesn't include the highlights or abstract section of the paper (as these are already summaries).
        :param filename: the filename to summarise.
        :return: the paper as an ordered list of sentences, not including abstract or highlights.
        """
        # Read the paper in
        paper = useful_functions.read_in_paper(filename, sentences_as_lists=True, preserve_order=True)

        # We don't want to make any predictions for the Abstract or Highlights as these are already summaries.
        sections_to_predict_for = []

        # Values in the dictionary have form (section_text, section_position_in_paper)
        for section, text in paper.iteritems():
            if section != "ABSTRACT" and section != "HIGHLIGHTS":
                sections_to_predict_for.append((text[0], text[1], section))

        # Sorts the sections according to the order in which they appear in the paper.
        sorted_sections_to_predict_for = sorted(sections_to_predict_for, key=itemgetter(1))

        # Creates an ordered list of the sentences in the paper
        sentence_list = []
        for sentence_text, section_position_in_paper, section in sorted_sections_to_predict_for:
            section_sentences = sentence_text
            for sentence in section_sentences:
                sentence_list.append((sentence, section))

        return sentence_list

    def prepare_for_summarisation(self, filename, visualise=False):
        """
        Prepares a paper to be summarised by the Word2Vec method.
        :param filename: the filename of the paper to summarise
        :param visualise: true if visualising
        :return: the paper in a form suitable to be summarised with the trained models.
        """
        sentences = self.paper2orderedlist(filename)

        # Final form will be an ordered list of tuples, where each tuple shall have the form
        # (sentence_text, sentence_vector, abstract_vector, features).
        final_form = []

        raw_paper = useful_functions.read_in_paper(filename, sentences_as_lists=True)

        abstract = raw_paper["ABSTRACT"]
        abs_vector = self.abstract2vector(abstract)

        prev_section = ""

        try:
            bow = self.paper_bags_of_words[filename]
        except KeyError:
            paper_str = useful_functions.read_in_paper(filename)
            paper_str = " ".join([val for _, val in paper_str.iteritems()]).lower()
            paper_bag_of_words = useful_functions.calculate_bag_of_words(paper_str)
            self.paper_bags_of_words[filename] = paper_bag_of_words

        try:
            kf = self.keyphrases[filename]
        except KeyError:
            kfs = raw_paper["KEYPHRASES"]
            self.keyphrases[filename] = kfs

        for sentence, section in sentences:

            sentence_vector = useful_functions.sentence2vec(sentence, self.word2vec)

            features = self.calculate_features(sentence,
                                               self.paper_bags_of_words[filename],
                                               self.keyphrases[filename],
                                               [" ".join(x) for x in abstract],
                                               " ".join(raw_paper["MAIN-TITLE"][0]),
                                               section,
                                               shorter=True)

            if not visualise:
                final_form.append((sentence, sentence_vector, abs_vector, features))
            else:
                if prev_section != section:
                    print("----> Adding section: ", section)
                    final_form.append(([section], np.zeros_like(sentence_vector), np.zeros_like(sentence_vector), np.zeros_like(features)))
                    prev_section = section
                final_form.append((sentence, sentence_vector, abs_vector, features))

        return final_form

    def abstract2vector(self, abstract):
        """
        Changes the abstract into a single averaged vector.
        :param abstract: the abstract to turn into a vector
        :return: a single vector representing the abstract
        """
        abstract_vecs = [useful_functions.sentence2vec(x) for x in abstract]
        avg = np.mean(abstract_vecs, axis=0)
        return avg

    def calculate_features(self, sentence, bag_of_words, keyphrases, abstract, title, section, shorter=False):
        """
        Calculates the features for a sentence.
        :param sentence: the sentence to calculate features for, as a list of words.
        :param bag_of_words: a dictionary bag of words representation for the paper, keys are words vals are counts.
        :param keyphrases: the keyphrases of the paper
        :param shorter: returns a shorter list of features
        :param abstract: the abstract of the paper as a list of strings
        :param title: the title of the paper as a string
        :param section: the section of the paper the sentence came from
        :return: a vector of features for the sentence.
        """
        # Calculate features
        abstract_rouge_score = useful_functions.compute_rouge_abstract_score(sentence, abstract)
        tf_idf = useful_functions.calculate_tf_idf(sentence, self.global_paper_count, bag_of_words)
        document_tf_idf = useful_functions.calculate_document_tf_idf(sentence, bag_of_words)
        keyphrase_score = useful_functions.calculate_keyphrase_score(sentence, keyphrases)
        title_score = useful_functions.calculate_title_score(sentence, set([x for x in title if x not in STOPWORDS]))
        sent_len = len(sentence)
        numeric_count = len([word for word in sentence if useful_functions.is_number(word)])
        sec = -1

        if "HIGHLIGHT" in section:
            sec = HIGHLIGHT
        elif "ABSTRACT" in section:
            sec = ABSTRACT
        elif "INTRODUCTION" in section:
            sec = INTRODUCTION
        elif "RESULT" in section or "DISCUSSION" in section:
            sec = RESULT_DISCUSSION
        elif "CONCLUSION" in section:
            sec = CONCLUSION
        elif "METHOD" in section:
            sec = METHOD
        else:
            sec = OTHER

        if shorter:
            return abstract_rouge_score, tf_idf, document_tf_idf, keyphrase_score, title_score, numeric_count, \
                   sent_len, sec

        if sent_len > 2 and sentence[0] in self.vocab:
            first_word = self.word2vec[sentence[0]]
        else:
            first_word = [0] * self.word2vec_feature_nums

        if sent_len > 2 and sentence[0] in self.vocab and sentence[1] in self.vocab:
            first_pair = np.concatenate((self.word2vec[sentence[0]], self.word2vec[sentence[1]]))
        else:
            first_pair = [0] * (self.word2vec_feature_nums * 2)

        return abstract_rouge_score, tf_idf, document_tf_idf, keyphrase_score, title_score, numeric_count, \
               sent_len, sec, first_word, first_pair

    def process_item(self, item):
        """
        Data item is of form:
        data = {
            "filename"
            "gold"
            "title"
            "abstract"
            "sentences"
            "description"
        }
        :param item: the data item to process
        :return: the processed data item
        """

        t = time.time()

        # Get the bag of words representation for this paper.
        bag_of_words = self.paper_bags_of_words[item["filename"]]

        # Get the keyphrases of this paper
        keyphrases = self.keyphrases[item["filename"]]

        # Get the abstract of this paper as a list of strings
        abstract = [" ".join(x) for x in item["abstract"]]

        # Get the title of this paper
        title = item["title"][0]

        # Get a vector representation of the abstract
        abs_vector = self.abstract2vector(abstract)

        # Get vector representations of each of the sentences
        sentence_vectors = [(useful_functions.sentence2vec(x), section, y) for x, section, y in item["sentences"]]

        # Get feature representations of each of the sentences
        features = [self.calculate_features(x, bag_of_words, keyphrases, abstract, title, section, True)
                    for x, section, y in item["sentences"]]

        description_text = "All text is of the form of a list of lists, where each sentence is a list of words. The" \
                           " sentences are of the form [(sentence (as a list of words), section in paper," \
                           " classification)]. The sentence vectors are of a similar form, except the sentence text is" \
                           " replaced with the vector representation of the sentence. The features are of the form " \
                           "[(AbstractROUGE, TF-IDF, Document_TF-IDF, keyphrase_score, title_score, numeric_score," \
                           " sentence_length, section)]. The dimensions of each sentence vector are [1x100]. The " \
                           "abstract vector is a single [1x100] vector also."

        new_data = {
            "filename": item["filename"],
            "gold": item["gold"],
            "title": item["title"],
            "abstract": item["abstract"],
            "abstract_vec": abs_vector,
            "sentences": item["sentences"],
            "sentence_vecs": sentence_vectors,
            "sentence_features": features,
            "description": description_text
        }

        print("Done, process took ", time.time() - t, " seconds, time since start is ",
              (time.time() - self.start_time) / 60, " minutes")

        return new_data

    def extra_processing(self):

        data_dir = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/data.pkl"
        write_dir = BASE_DIR + \
            "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

        print("----> Reading data...")
        t = time.time()
        data = useful_functions.load_pickled_object(data_dir)
        print("----> Done, took ", time.time() - t, " seconds")

        print("----> Beginning processing...")
        t = time.time()
        self.start_time = t
        new_data = self.pool2.map(self.process_item, data)
        # new_data = np.concatenate(new_data, axis=0)
        print("----> Done, took ", (time.time() - t) / 60, " minutes")

        useful_functions.pickle_list(new_data, write_dir)


    def prepare_data(self):
        """
        Puts the data in a form suitable for the Word2Vec classifier - it changes each sentence into the average of
        its constituent word vectors.
        :return: all sentences as vectors and their classification (data is balanced).
        """

        # Count of how many papers have been processed.
        count = 0

        # Sentences as vectors with their classification
        data = []

        # Count of positive data
        pos_count = 0

        # Count of negative data
        neg_count = 0

        r = Rouge()

        # Iterate over every file in the paper directory
        for filename in os.listdir(PAPER_SOURCE):

            # Ignores files which are not papers e.g. hidden files
            if filename.endswith(".txt"):

                # Display a loading bar of progress
                useful_functions.loading_bar(self.loading_section_size, count, self.number_of_papers)
                count += 1

                # Opens the paper as a dictionary, with keys corresponding to section titles and values corresponding
                # to the text in that section. The text is given as a list of lists, each list being a list of words
                # corresponding to a sentence.
                paper = useful_functions.read_in_paper(filename, sentences_as_lists=True)

                # Get the highlights of the paper
                highlights = paper["HIGHLIGHTS"]
                highlights_join = [" ".join(x) for x in highlights]
                abstract = paper["ABSTRACT"]

                sentences = []

                # Iterate over the whole paper
                for section, sents in paper.iteritems():

                    # Iterate over each sentence in the section
                    for sentence in sents:

                        # We don't want to calculate ROUGE for the abstract
                        if section != "ABSTRACT":
                            # Calculate the ROUGE score and add it to the list
                            r_score = r.calc_score([" ".join(sentence)], highlights_join)
                            sentences.append((sentence, r_score, section))

                sentences = [(x, section) for x, score, section in reversed(sorted(sentences, key=itemgetter(1)))]

                sents_pos = sentences[0:self.num_summary]
                sents_neg = sentences[self.num_summary:]

                if len(sents_neg) < len(sents_pos):
                    continue

                sents_pos = [(x[0], x[1], y) for x, y in zip(sents_pos, [1] * len(sents_pos))]
                sents_neg = [x for x in reversed(sents_neg)][:len(sents_pos)]
                sents_neg = [(x[0], x[1], y) for x, y in zip(sents_neg, [0] * len(sents_neg))]
                sents_class = sents_pos + sents_neg
                random.shuffle(sents_class)

                # Each item in the sentence list has form [(sentence, section, classification)]
                paper = {
                    "filename": filename,
                    "title": paper["MAIN-TITLE"],
                    "gold": paper["HIGHLIGHTS"],
                    "abstract": abstract,
                    "sentences": sents_class,
                    "description": "All text data is given in the form of a list of words."
                }

                data.append(paper)

                if count % 1000 == 0:
                    print("\nWriting data...")
                    write_dir = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/"
                    with open(write_dir + "data.pkl", "wb") as f:
                        pickle.dump(data, f)
                    print("Done")

        return data


if __name__ == "__main__":
    pp = AbstractNetPreprocessor()
    pp.extra_processing()

    #sent_vecs = pp.prepare_data()

    write_dir = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/"
    with open(write_dir + "abstractnet_data.pkl", "rb") as f:
        data = pickle.load(f)

    print(np.shape(data))

    #for item in data:
    #    print(item["abstract"])
    #    print(len(item["abstract"]))
    #    wait()

