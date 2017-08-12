# **** USEFUL FUNCTIONS ****
# This file provides several functions that handle small and tedious task to make the whole programming experience for
# summarisers more pleasant.

# ================ IMPORTS ================
from __future__ import print_function, division
import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from nltk.tokenize import sent_tokenize, word_tokenize
from Dev.DataTools.Reader import Reader
from Dev.DataTools.SentenceComparator import SentenceComparator
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from gensim.models import Word2Vec
from Dev.Evaluation.rouge import Rouge
from operator import itemgetter
import numpy as np
import tensorflow as tf
import re
import os
import string
import dill
import pickle
import time
import ujson as json

# =========================================

# ================ CONFIG VARIABLES ================

# The base directory of the project, from the root directory
BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"

# The path to the directory in which the papers are stored
PAPER_SOURCE = BASE_DIR + "/Data/Papers/Full/Papers_With_Section_Titles/"

# The source of the highlights - where each file in the directory contains the highlights for that paper
HIGHLIGHT_SOURCE = BASE_DIR + "/Data/Papers/Sections/highlights/"

# Path to the file containing the permitted section titles for parsing the papers
PERMITTED_TITLES_SOURCE = BASE_DIR + "/Data/Utility_Data/permitted_titles.txt"

# Location of the dictionary which contains bag of words representations of each paper - keys are filenames
PAPER_BAG_OF_WORDS_LOC = BASE_DIR + "/DataTools/pickled_dicts/paper_bag_of_words.pkl"

# Location of dictionary which contains the keyphrases for each paper - keys are filenames
KEYPHRASES_LOC = BASE_DIR + "/DataTools/pickled_dicts/keyphrases.pkl"

# Location of dictionary which is a global wordcount of the number of words counted across all papers
GLOBAL_COUNT_LOC = BASE_DIR + "/Data/Utility_Data/Global_Counts/global_wordcount.pkl"

# Location of the file with stopwords in it
STOPWORD_SOURCE = BASE_DIR + "/Data/Utility_Data/common_words.txt"

# Location of the Word2Vec model
MODEL_SOURCE = BASE_DIR + "/Word2Vec/Word2Vec_Models/summarisation_100features_5minwords_20context"

# Location to write the document word count - the count of how many different papers each word occurs in
GLOBAL_WORDCOUNT_WRITE_LOC = BASE_DIR + "/Data/Utility_Data/Global_Counts/"

# This may not be needed anymore
DD_CLASSIFICATION_MODEL_LOC_VECS = BASE_DIR + "/Trained_Models/DoubleDataset/lr_c2_5_vector.pkl"
DD_CLASSIFICATION_MODEL_LOC_FEATS = BASE_DIR + "/Trained_Models/DoubleDataset/lr_c2_5_feature.pkl"
W2V_CLASSIFICATION_MODEL_LOC = BASE_DIR + "/Trained_Models/Word2Vec_Classifier/lr_highlights_only_c3_66.pkl"

# Location to write the training data to
TRAINING_DATA_WRITE_LOC = BASE_DIR + "/Data/Training_Data/"

# Reads the stopwords as a set
STOPWORDS = set(Reader().open_file(STOPWORD_SOURCE) + list(string.punctuation))

# Counts how many papers there are in the paper directory
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])

# Integers defining the possible location of sentences in the paper
HIGHLIGHT = 0
ABSTRACT = 1
INTRODUCTION = 2
RESULT_DISCUSSION = 3
METHOD = 4
CONCLUSION = 5
OTHER = 6

# An object which can compare two sentences and tell how similar they are / if they are the same
SENTENCE_COMPARATOR_OBJ = SentenceComparator()

# ==================================================

# ================ CLASSES ================

class Color:
    """
    Small class used to print to the console in different colours.
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# ==========================================

# ================ FUNCTIONS ================


def wait(to_print=""):
    """
    A simple function which pauses execution flow, used for debugging.
    """
    print(to_print)
    raw_input("Press enter to continue...")


def printlist(list, wait_on_iteration=True):
    """
    Convenience function that prints a list.
    :param list: the list to print
    :param wait_on_iteration: if true, pause execution after printing each list item.
    """
    for item in list:
        print(item)
        print()
        if wait_on_iteration:
            wait()


def printdict(dictionary, wait_on_iteration=True):
    """
    Convenience function to print a dictionary.
    :param dictionary: the dictionary to print.
    :param wait_on_iteration: if true, pause execution after each print.
    """
    for key, value in dictionary.iteritems():
        print(key, " ", value)
        if wait_on_iteration:
            wait()


def num2onehot(number, size):
    """
    Turns a number into a one-hot vector.
    :param number: the number to represent as a one-hot vector.
    :param size: the total size of the numpy array that will be used to represent the number.
    :return: number as a one-hot vector
    """
    num1h = [0 for _ in range(size)]
    num1h[number] = 1
    return num1h


def weight_variable(shape):
    """
    A handy little function to create TensorFlow weight variables.
    :param shape: the dimensions of the variable to be created
    :return: a TensorFlow weight variable ready for training
    """
    variable = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(variable)


def bias_variable(shape):
    """
    A handy little function to create a TensorFlow bias variable.
    :param shape: the dimensions of the variable to be created
    :return: a TensorFlow bias variable ready for training
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    TensorFlow operation used for convolution.
    :param x: data to convolve weights with.
    :param W: the weights to convolve with the data.
    :return: the TensorFlow convolution operation.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    Max Pooling operation for convolutional networks.
    :param x: the data to max pool.
    :return: the max pooled data.
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def paper_tokenize(text, sentences_as_lists=False, preserve_order=False):
    """
    Takes a paper with the sections delineated by '@&#' and splits them into a dictionary where the key is the section
    and the value is the text under that section. This could probably be a bit more efficient but it works well enough.
    :param text: the text of the paper to split
    :param sentences_as_lists: if true, returns the text of each section as a list of sentences rather than a single
                               string.
    :param preserve_order: if true, tracks the order in which the paper sections occured.
    :returns: a dictionary of the form (section: section_text)
    """
    permitted_titles = set(Reader().open_file(PERMITTED_TITLES_SOURCE))

    # Split the text into sections
    if preserve_order:
        split_text_1 = re.split("@&#", text)
        split_text = zip(split_text_1, range(len(split_text_1)))
    else:
        split_text = re.split("@&#", text)

    # The key value. This value is changed if a permitted section title is encountered in the list.
    state = ""

    # After the for loop, this dictionary will have keys relating to each permitted section, and values corresponding
    # to the text of that section
    sentences_with_states = defaultdict(str)

    section_counts = defaultdict(int)

    if preserve_order:
        for text, pos in split_text:

            # Hack for proper sentence tokenization because NLTK tokeniser doesn't work properly for tokenising papers
            text = text.replace("etal.", "etal")
            text = text.replace("et al.", "etal")
            text = text.replace("Fig.", "Fig")
            text = text.replace("fig.", "fig")
            text = text.replace("Eq.", "Eq")
            text = text.replace("eq.", "eq")
            text = text.replace("pp.", "pp")
            text = text.replace("i.e.", "ie")
            text = text.replace("e.g.", "eg")
            text = text.replace("ref.", "ref")
            text = text.replace("Ref.", "Ref")
            text = text.replace("etc.", "etc")
            text = text.replace("Figs.", "Figs")
            text = text.replace("figs.", "figs")
            text = text.replace("No.", "No")
            text = text.replace("eqs.", "eqs")

            # Checks if text is a section title
            if text.lower() in permitted_titles:
                state = text
                section_counts[state] += 1
            else:
                if sentences_as_lists:
                    if section_counts[state] > 1:
                        state = state + "_" + str(section_counts[state])
                    sentences_with_states[state] = ([preprocess_sentence(x) for x in sent_tokenize(text)], pos)
                else:
                    if section_counts[state] > 1:
                        state = state + "_" + str(section_counts[state])
                    sentences_with_states[state] = (text, pos)

    if not preserve_order:
        for text in split_text:

            # Hack for proper sentence tokenization because NLTK tokeniser doesn't work properly for tokenising papers
            text = text.replace("etal.", "etal")
            text = text.replace("et al.", "etal")
            text = text.replace("Fig.", "Fig")
            text = text.replace("fig.", "fig")
            text = text.replace("Eq.", "Eq")
            text = text.replace("eq.", "eq")
            text = text.replace("pp.", "pp")
            text = text.replace("i.e.", "ie")
            text = text.replace("e.g.", "eg")
            text = text.replace("ref.", "ref")
            text = text.replace("Ref.", "Ref")
            text = text.replace("etc.", "etc")
            text = text.replace("Figs.", "Figs")
            text = text.replace("figs.", "figs")
            text = text.replace("No.", "No")
            text = text.replace("eqs.", "eqs")

            # Checks if text is a section title
            if text.lower() in permitted_titles:
                state = text
                section_counts[state] += 1
            else:
                if sentences_as_lists:
                    if section_counts[state] > 1:
                        state = state + "_" + str(section_counts[state])
                    sentences_with_states[state] = [preprocess_sentence(x) for x in sent_tokenize(text)]
                else:
                    if section_counts[state] > 1:
                        state = state + "_" + str(section_counts[state])
                    sentences_with_states[state] = text

    return sentences_with_states


def is_number(string):
    """
    Checks if a string is a number.
    :param string: the string to check whether it is a number.
    :return: True if string is a number, False otherwise.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def read_in_paper(filename, sentences_as_lists=False, preserve_order=False):
    """
    Reads in a paper and returns it as a dictionary.
    :param filename: the filename of the paper to read, of the paper file itself only not the path to the paper.
    :param sentences_as_lists: if true, will return the sentences of the paper as lists of words rather than strings.
    :param preserve_order: if true keeps track of which sections occured in what order in the paper.
    :return: a dictionary of the form (section: list of sentences in that section)
    """
    paper_to_open = PAPER_SOURCE + filename
    paper_text = Reader().open_file_single_string(paper_to_open)
    udata = paper_text.decode("utf-8")
    paper = udata.encode("ascii", "ignore")
    return paper_tokenize(paper, sentences_as_lists=sentences_as_lists, preserve_order=preserve_order)


def preprocess_sentence(sentence):
    """
    Preprocesses a sentence, turning it all to lowercase and tokenizing it into words.
    :param sentence: the sentence to pre-process.
    :return: the sentence, as a list of words, all in lowercase
    """
    sentence = sentence.lower()
    return word_tokenize(sentence)


def read_stopwords():
    """
    Reads the list of stop words that should not be included in the scoring of each sentence such as \"and\"
    and \"or\".
    """
    common_words = Reader().open_file(STOPWORD_SOURCE) + list(string.punctuation)
    return set(common_words)


def load_word2vec():
    """
    Loads the word2vec model used in this work.
    :return: a word2vec model.
    """
    return Word2Vec.load(MODEL_SOURCE)

# The word2vec model
WORD2VEC = load_word2vec()


def loading_bar(loading_section_size, count, total_number):
    """
    Prints a loading bar to the console.
    :param loading_section_size: The number of items that should be processed before an extra bar is added to the
                                 loading bar.
    :param count: The number of items processed so far.
    :param total_number: The total number of items to be processed.
    """
    print("Processing Files: [", end="")
    for i in range(31, -1, -1):
        if count > i * loading_section_size:
            for j in range(0, i):
                print("-", end="")
                sys.stdout.flush()
            for j in range(i, 30):
                print(" ", end="")
                sys.stdout.flush()
            break;
    if count == total_number:
        print("] ", count, end="\n")
    else:
        print("] ", count, end="\r")
    sys.stdout.flush()


def count_words_across_all_papers(write_location=GLOBAL_WORDCOUNT_WRITE_LOC):
    """
    Counts how many different papers each words occurs in.
    :return: nothing, but writes the wordcount dictionary to the disk.
    """

    # Count the number of papers to read and prepare the loading bar
    num_files = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
    loading_section_size = num_files / 30
    count = 0

    # The defaultdict to hold the global word count for the number of different papers each word occurs in
    global_word_count = defaultdict(int)

    # The defaultdict to hold the total count of all words
    global_total_word_count = defaultdict(int)

    for filename in os.listdir(PAPER_SOURCE):
        if filename.endswith(".txt"):

            # Display the loading bar
            loading_bar(loading_section_size, count, num_files)
            count += 1

            # Read the paper
            paper = read_in_paper(filename, sentences_as_lists=True)

            # Get the paper's vocab
            all_sections = [section for _, section in paper.iteritems()]
            paper_words = [word for section in all_sections for sentence in section for word in sentence]
            vocab = set(paper_words)

            # Add to the counter dict for words that occurred in this paper
            for word in vocab:
                global_word_count[word] += 1

            # Add to the total counts dict
            #for word in paper_words:
            #    global_total_word_count[word] += 1

    # Write the wordcount
    with open(write_location + "document_wordcount.pkl", "wb") as f:
        pickle.dump(global_word_count, f)

    #with open(GLOBAL_WORDCOUNT_WRITE_LOC + "global_total_wordcount.pkl", "wb") as f:
    #    pickle.dump(global_total_word_count, f)


def load_global_word_count():
    """
    Loads the global word count dictionary of the number of different papers each word occurs in and the word count
    dictionary of the total occurences of each word in the corpus.
    :return: a tuple of these dictionaries
    """
    with open(GLOBAL_WORDCOUNT_WRITE_LOC + "global_wordcount.pkl", "rb") as f:
        num_papers_words_occur_in = pickle.load(f)

    with open(GLOBAL_WORDCOUNT_WRITE_LOC + "global_total_wordcount.pkl", "rb") as f:
        global_word_count = pickle.load(f)

    return num_papers_words_occur_in, global_word_count


def create_paper_dictionaries(filename="", readin=True, paper=None):
    """
    Creates the metadata data structures for a specific paper required to compute the extra features which are
    appended to the sentence vector.
    :param filename: the filename only, not the path, for the paper to create dictionaries for.
    :return: a tuple of the metadata data structures for the paper.
    """

    if readin and filename != "":
        # Read the paper in as a dictionary, keys are sections and values are the section text
        paper = read_in_paper(filename)

    # Extract paper keyphrases
    keyphrases = set(filter(None, " ".join(paper["KEYPHRASES"].lower().split("\n")).split(" ")))

    # Get the paper's vocab
    full_paper = " ".join([val for _, val in paper.iteritems()]).lower()
    paper_words = word_tokenize(full_paper)
    vocab = set(paper_words)

    # Create a bag of words for the paper
    paper_bag_of_words = defaultdict(int)
    for word in paper_words:
        paper_bag_of_words[word] += 1

    # Get the title words
    title_words = set([x.lower() for x in word_tokenize(paper["MAIN-TITLE"]) if x not in STOPWORDS])

    return keyphrases, vocab, paper_bag_of_words, title_words


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

    for word in sentence:

        if word in STOPWORDS:
            continue

        # Get the number of documents containing this word - the idf denominator (1 is added to prevent division by 0)
        docs_containing_word = global_count_of_papers_words_occur_in[word] + 1

        # Count of word in this paper - the tf score
        count_word = bag_of_words[word]

        idf = np.log(NUMBER_OF_PAPERS / docs_containing_word)

        #word_tf_idf = (1 + np.log(count_word)) * idf

        word_tf_idf = count_word * idf

        sentence_tf_idf += word_tf_idf

        length += 1

    if length == 0:
        return 0
    else:
        sentence_tf_idf = sentence_tf_idf / length
        return sentence_tf_idf


def calculate_keyphrase_score(sentence, keyphrases):
    """
    Calculates the "keyphrase score" - which is the number of author defined keyphrases that the sentence contains.
    :param sentence: the sentence to calculate the score for as a list of words
    :param keyphrases: the keyphrases of the paper
    :return: the number of keyphrases used in the sentence
    """

    score = 0

    for word in sentence:
        if word in STOPWORDS:
            continue

        if word in keyphrases:
            score += 1

    return score


def calculate_title_score(sentence, title):
    """
    Calculates the "title score" - which is the number of non-stopword words from the title the sentence contains.
    :param sentence: the sentence to calculate the score for as a list of words
    :param title: the non-stopword words from the title, as a set
    :return: the number of non-stopword words from the title used in the sentence
    """

    score = 0

    for word in sentence:
        if word in STOPWORDS:
            continue

        if word in title:
            score += 1

    return score


def calculate_bag_of_words(paper_string):
    """
    Calculates the bag of words representation of a paper and returns a defaultdict.
    :param paper_string: the paper in string representation.
    :return: the paper's bag of words representation as a defaultdict.
    """
    bow = defaultdict(int)
    for word in paper_string.split():
        bow[word] += 1

    return bow


def bag_of_words_score(sentence, paper_bag_of_words):
    """
    Calculates the score for a sentence based on the bag of words representation of the paper.
    :param sentence: the sentence to calculate the score for, as a list of words
    :param paper_bag_of_words: the bag of words representation of the paper
    :return: the bag of words score for the sentence
    """

    # Get the bag of words for this sentence
    bag_of_words = paper_bag_of_words

    # Calculate the score, which is done by finding the number of times each non-stopword word from the sentence occurs
    # in the article and summing these for each word, divided by the number of non-stopword words in the sentence.
    score = 0
    count = 0

    for word in sentence:
        if word in STOPWORDS:
            continue
        else:
            score += bag_of_words[word]
            count += 1

    if count == 0:
        return 0
    else:
        return score / count


def compute_rouge_abstract_score(sentence, abstract):
    """
    Computes the ROUGE score of the given sentence compared to the given abstract.
    :param sentence: the sentence to compute the ROUGE score for, as a list of words.
    :param abstract: the abstract of the paper to compute the ROUGE score against, as a list of strings.
    :return: the ROUGE score of the sentence compared to the abstract.
    """
    r = Rouge()
    return r.calc_score([" ".join(sentence)], abstract)


def calculate_document_tf_idf(sentence, paper_bag_of_words):
    """
    Computes the TF-IDF score in relation to the document - using the words from the sentence as Term Frequency (TF) and
    total words in the paper as Document Frequency (IDF).
    :param sentence: the sentence to calculate the score for, as a list of words.
    :param paper_bag_of_words: bag of words representation of the paper, as a dictionary.
    :return: the document-based TF-IDF score of the sentence.
    """
    # Sanity check
    assert(len(paper_bag_of_words) > 0)

    sentence_bag_of_words = defaultdict(float)
    for word in sentence:

        if word in STOPWORDS:
            continue
        else:
            sentence_bag_of_words[word] += 1.0

    sent_tf_idf = 0
    length = 0

    for word in sentence:

        if word in STOPWORDS:
            continue

        tf = sentence_bag_of_words[word]

        # Add 1 to the denominator to prevent division by 0
        idf = np.log(len(paper_bag_of_words) / (paper_bag_of_words[word] + 1))

        tf_idf = tf * idf

        sent_tf_idf += tf_idf

        length += 1

    if length == 0:
        return 0
    else:
        return sent_tf_idf / length


def calculate_features(sentence, bag_of_words, document_wordcount, keyphrases, abstract, title, section):
    """
    Calculates the features for a sentence.
    :param sentence: the sentence to calculate features for, as a list of words.
    :param bag_of_words: a dictionary bag of words representation for the paper, keys are words vals are counts.
    :param document_wordcount: count of how many different papers each word occurs in.
    :param keyphrases: the keyphrases of the paper
    :param shorter: returns a shorter list of features
    :param abstract: the abstract of the paper as a list of strings
    :param title: the title of the paper as a string
    :param section: the section of the paper the sentence came from
    :return: a vector of features for the sentence.
    """
    # Calculate features
    abstract_rouge_score = compute_rouge_abstract_score(sentence, abstract)
    tf_idf = calculate_tf_idf(sentence, document_wordcount, bag_of_words)
    document_tf_idf = calculate_document_tf_idf(sentence, bag_of_words)
    keyphrase_score = calculate_keyphrase_score(sentence, keyphrases)
    title_score = calculate_title_score(sentence, set([x for x in title if x not in STOPWORDS]))
    sent_len = len(sentence)
    numeric_count = len([word for word in sentence if is_number(word)])
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

    return abstract_rouge_score, tf_idf, document_tf_idf, keyphrase_score, title_score, numeric_count, \
               sent_len, sec


def paper2vec(paper, model, metadata):
    """
    Converts a paper into a list of vectors, one vector for each sentence.
    :param paper: the paper in the form of a dictionary. The keys of the dictionary are the sections of the paper, and
                  the values are a list of lists, where each list is a list of words corresponding to a sentence in
                  that section.
    :param model: the word2vec model used to transform the paper into a vector.
    :param metadata: dictionaries containing metadata about the paper.
    :return: the paper in the form of a dictionary, the key being the section and the values being a list of vectors,
             with each vector corresponding to a sentence from that part of the paper.
    """
    stopwords = read_stopwords()
    new_paper = defaultdict(None)

    for section, text in paper.iteritems():
        section_as_vector = []
        section_text = []
        pos = text[1]
        text = text[0]
        for sentence in text:
            if "HIGHLIGHTS" in section:
                sentence_vec = sentence2vec(sentence, model, stopwords, metadata, HIGHLIGHT, False)
            elif "ABSTRACT" in section:
                sentence_vec = sentence2vec(sentence, model, stopwords, metadata, ABSTRACT, False)
            elif "INTRODUCTION" in section:
                sentence_vec = sentence2vec(sentence, model, stopwords, metadata, INTRODUCTION, False)
            elif "RESULT" in section or "DISCUSSION" in section:
                sentence_vec = sentence2vec(sentence, model, stopwords, metadata, RESULT_DISCUSSION, False)
            elif "CONCLUSION" in section:
                sentence_vec = sentence2vec(sentence, model, stopwords, metadata, CONCLUSION, False)
            else:
                sentence_vec = sentence2vec(sentence, model, stopwords, metadata, OTHER, False)
            section_as_vector.append(sentence_vec)
            section_text.append(sentence)
        new_paper[section] = (section_text, section_as_vector, pos)
    return new_paper


def sentence2vec(sentence, model=WORD2VEC, stopwords=STOPWORDS, metadata=None, section=None, wordvecs_only=True):
    """
    Changes a sentence into a vector by averaging the word vectors of every non-stopword word in the sentence.
    :param sentence: the sentence to turn into a vector, as a list of words
    :param model: the word2vec model to use to convert words to vectors
    :param stopwords: stopwords to not include in the averaging of each sentence.
    :param metadata: dictionaries of metadata for the paper.
    :param section: the section of the paper the sentence occurs in.
    :param wordvecs_only: will turn a sentence into a vector using only the the word vectors from the model, no extra
                          features.
    :return: the sentence in vector representation
    """
    # The shape of the model, used to get the number of features and its vocab
    model_shape = model.syn0.shape
    vocab = set(model.index2word)

    # The array that will be used to calculate the average word vector
    average = np.zeros((model_shape[1]), dtype="float32")
    total_word_count = 0

    for word in sentence:

        if word in stopwords:
            continue

        if word in vocab:
            word_rep = model[word]
            average += word_rep
            total_word_count += 1

    if total_word_count == 0:
        total_word_count = 1

    average = np.divide(average, total_word_count)

    sentence_vec = average

    return sentence_vec


def abstract2vector(abstract):
    """
    Changes the abstract into a single averaged vector.
    :param abstract: the abstract to turn into a vector
    :return: a single vector representing the abstract
    """
    abstract_vecs = [sentence2vec(x) for x in abstract]
    avg = np.mean(abstract_vecs, axis=0)
    return avg


def read_paper_as_vectors(filename, model):
    """
    Reads in a paper into a dictionary in vector format.
    :param filename: the paper to read, filename only not a path.
    :param model: the word2vec model to use to convert the paper to a vector.
    :return: a dictionary of form (section: (text, text as a list of vectors, section position in paper))
    """
    paper = read_in_paper(filename, sentences_as_lists=True, preserve_order=True)
    metadata = create_paper_dictionaries(filename=filename) + load_global_word_count()
    return paper2vec(paper, model, metadata)


def load_doubledataset_classification_model():
    """
    Loads the trained classification model for the DoubleDataset classifier.
    :return: the trained classification model.
    """
    with open(DD_CLASSIFICATION_MODEL_LOC_FEATS, "rb") as f:
        features = pickle.load(f)

    with open(DD_CLASSIFICATION_MODEL_LOC_VECS, "rb") as f:
        vectors = pickle.load(f)

    return vectors, features


def load_word2vec_classification_model():
    """
    Loads the trained classification model for the Word2Vec classifier.
    :return: the trained classification model.
    """
    with open(W2V_CLASSIFICATION_MODEL_LOC, "rb") as f:
        model = pickle.load(f)

    return model


def check_summary_sentences(summary_sentences, sentence):
    """
    Checks whether sentence is in summary_sentences.
    :param summary_sentences: a list of lists, each list corresponding to a tokenized sentence (the sentence is a list
                              of words)
    :param sentence: a tokenized sentence to check if whether it is in summary_sentences or not
    :return: True if sentence is in summary_sentences, False otherwise
    """
    for sum_sentence in summary_sentences:
        in_summary = SENTENCE_COMPARATOR_OBJ.compare_sentences(sum_sentence, sentence, STOPWORDS, tokenized=True)

        if in_summary == 1:
            return True

    return False


def pickle_list(list_to_pickle, write_location):
    """
    Pickles a list - writes it out in Python's pickle format at the specified location.
    :param list_to_pickle: the list to persist.
    :param write_location: the location to write the list to.
    """
    with open(write_location, "wb") as f:
        pickle.dump(list_to_pickle, f)


def load_pickled_object(object_location):
    """
    Loads a pickled object from a pickle file.
    :param object_location: the location of the pickle file to read from
    :return: the object read back in
    """
    with open(object_location, "rb") as f:
        obj = pickle.load(f)
    return obj


def numpy_save(list_to_save, write_location):
    """
    Saves a list using the numpy "Save" function which is slightly faster than pickling.
    :param list_to_save: the list to persist.
    :param write_location: the location to write the list to.
    """
    np.save(write_location, list_to_save)


def numpy_load(read_location):
    """
    Reads a list stored with numpy.save()
    :param read_location: the location to read the list from
    :return: the list
    """
    return np.load(read_location)


def read_section_titles():
    """
    Reads the section titles permitted in the paper.
    :return: A set of the permitted titles.
    """
    return set(Reader().open_file(BASE_DIR + "/Data/Utility_Data/permitted_titles.txt"))

# The title of sections permitted in papers
SECTION_TITLES = read_section_titles()


def read_definite_non_summary_section_titles():
    """
    Reads the list of sections titles from which summary statements are very rare to come from.
    :return: the list of such section titles.
    """
    return set(Reader().open_file(BASE_DIR + "/Data/Utility_Data/definite_non_summary_titles.txt"))


def write_summary(location, summary_as_list, filename):
    """
    Writes a generated summary to the specified location, writing both a pickle file and a text file; the pickle file
    for easy program reading, and a text file for easy human and ROUGE reading.
    :param location: the location to write the summary
    :param summary_as_list: the summary to write, as a list of tuples with each tuple of the form
                            (sentence, sentence_index_into_paper)
    :param filename: the name of the file to write.
    """
    with open(location + "Pickles/" + filename + ".pkl", "wb") as f:
        pickle.dump(summary_as_list, f)

    raw_sentences = [x for x, _ in summary_as_list]

    with open(location + "Text/" + filename + ".txt", "wb") as f:
        for sentence in raw_sentences:
            f.write(sentence)
            f.write("\n")


def write_gold(location, gold_as_list, filename):
    """
    Writes the gold summary, i.e. the highlights, to the specified file and location.
    :param location: the location to write the summary.
    :param gold_as_list: the gold summary as a list of sentences where each sentence is a list of words.
    :param filename: the name of the file to write.
    """
    with open(location + "Gold/" + filename, "wb") as f:
        for word_list in gold_as_list:
            sentence = " ".join(word_list)
            f.write(sentence)
            f.write("\n")


def load_cspubsumext():
    """
    Loads cspubsumext as created by DataTools/DataPreprocessing/cspubsumext_creator.py. Assumes that the papers are
    stored in the location specified in the TRAINING_DATA_WRITE_LOC variable from this file (useful_functions.py).
    :return: a list of the loaded dictionaries, one for each paper.
    """
    print("----> Loading JSON file...")
    st = time.time()
    json_file = json.load(open(TRAINING_DATA_WRITE_LOC + "all_data.json", "rb"))
    print("----> Done, took {} seconds.".format(time.time() - st))
    print("----> Converting number lists to numpy arrays...")
    i = 0
    cspubsumext = []
    for data_item in json_file:

        new_data_item = {}

        for key, val in data_item.iteritems():

            if key == "abstract_vec":
                val = np.array(val)
            elif key == "sentence_vecs":
                new_sent_vecs = []
                for vec, sec, y in val:
                    new_sent_vecs.append((np.array(vec), sec, y))
                val = new_sent_vecs

            new_data_item[key] = val

        cspubsumext.append(new_data_item)

        print("Processed {}".format(i), end="\r")
        i += 1
        sys.stdout.flush()
    print("\n----> Done.")
    return cspubsumext


# ===========================================

if __name__ == '__main__':
    t = time.time()
    cspubsumext = load_cspubsumext()
    print(time.time() - t)
    print(cspubsumext[0])