# ================ IMPORTS ================

from __future__ import print_function, division
import os
import dill
import pickle
import sys
import random
import time
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import matplotlib.pyplot as plt
from operator import itemgetter
from Dev.DataTools.Reader import Reader
from Dev.DataTools.DataPreprocessing.DataPreprocessor import DataPreprocessor
from multiprocessing.dummy import Pool as ThreadPool
from Dev.DataTools import useful_functions
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.useful_functions import Color, wait, printlist, num2onehot, BASE_DIR, PAPER_SOURCE, HIGHLIGHT_SOURCE,\
    PAPER_BAG_OF_WORDS_LOC, KEYPHRASES_LOC, GLOBAL_COUNT_LOC, WORD2VEC, weight_variable, bias_variable, conv2d, max_pool_2x2
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.LSTM_preproc.vocab import Vocab
from Dev.DataTools.LSTM_preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from Dev.DataTools.LSTM_preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import numpy as np

# =========================================

# ================ CONFIG VARIABLES ================

# The dimensions of the word vectors
WORD_DIMENSIONS = 100

# The dimension of the abstract vector
ABSTRACT_DIMENSION = 100

# The number of handcrafted features
NUM_FEATURES = 8

# The size of the batch of the data to feed
BATCH_SIZE = 100

# Weighting dimensions for the weighting for the sentences
SENTENCE_WEIGHTS = 200

# The number of units in the hidden layer
HIDDEN_LAYER_WEIGHTS = 64
FINAL_LAYER_WEIGHTS = 128

# The number of classes to classify into
NUM_CLASSES = 2

# The network learning rate
LEARNING_RATE = 0.0001

# Maximum number of epochs
MAX_EPOCHS = 1000

# How often to display network progress and test its accuracy
DISPLAY_EVERY = 100

# How many steps the network can go before it is deemed to have converged
MAX_STEPS_SINCE_SAVE = 15

MAX_SENT_LEN = 100

VOCAB = set(WORD2VEC.index2word)

# True if the model is already trained
PRETRAINED = True

# True to analyse errors
ANALYSE_ERRORS = True

# The name of this model
MODEL_NAME = "LSTM"

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

# The location to save the model at
SAVE_PATH = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/" + str(HIDDEN_LAYER_WEIGHTS) + "_units/" + MODEL_NAME + "_" +\
            str(HIDDEN_LAYER_WEIGHTS) + ".ckpt"

# The directory to save the model in
SAVE_DIR = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/" + str(HIDDEN_LAYER_WEIGHTS) + "_units/"


# ==================================================

# ================ FUNCTIONS ================

def get_data():
    print("Loading Data...")
    t = time.time()
    data = useful_functions.load_cspubsumext()
    sentences_class = []
    for item in data:
        sentences = item["sentences"]
        features = item["sentence_features"]

        for sentence, feat in zip(sentences, features):
            sent = sentence[0]
            sec = sentence[1]
            y = sentence[2]
            sentences_class.append((sent, feat, y))
    data = sentences_class
    print("Done, took ", time.time() - t, " seconds")

    print("Processing Data...")

    #new_data = [x for x in data if len(x[0]) < MAX_SENT_LEN]
    new_data = []
    for sent, feat, y in data:
        if len(sent) > MAX_SENT_LEN:
            new_sent = sent[0:MAX_SENT_LEN]
        else:
            new_sent = sent
        new_data.append((new_sent, feat, y))

    print("Done")

    return new_data

def batch2input(batch_data, num_items):
    """
    :param batch_data: of form [(sentence, label)]
    :return: 3D matrix
    """
    batch_labels = [num2onehot(y, NUM_CLASSES) for _, y in batch_data]
    lens = [len(x) for x, _ in batch_data]
    batch_data = [x for x, _ in batch_data]
    inputs_matrix = np.zeros((num_items, MAX_SENT_LEN, WORD_DIMENSIONS), dtype=np.float32)
    for i, sentence in enumerate(batch_data):
        pos = 0
        for word in sentence:
            if word in VOCAB:
                vec = WORD2VEC[word]
                inputs_matrix[i, pos, :] = vec
            pos += 1
    return inputs_matrix, lens, batch_labels

def sents2input(batch_data, num_items):
    """
    :param batch_data: list of sentences, each sentence being a list of words
    :return: 3D matrix
    """
    lens = [len(x) for x in batch_data]
    inputs_matrix = np.zeros((num_items, MAX_SENT_LEN, WORD_DIMENSIONS), dtype=np.float32)
    for i, sentence in enumerate(batch_data):
        pos = 0
        for word in sentence:
            if word in VOCAB:
                vec = WORD2VEC[word]
                inputs_matrix[i, pos, :] = vec
            pos += 1
    return inputs_matrix, lens

# ===========================================

# ================ MAIN ================

def graph():
    """
    Function to encapsulate the construction of a TensorFlow computation graph.
    :return: input placeholders, optimisation operation, loss, accuracy, prediction operations
    """

    # Define placeholders

    # Input has shape [batch_size x num_steps x num_input (word vector dimensions)]
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, MAX_SENT_LEN, WORD_DIMENSIONS])

    # The lengths of each of the sentences
    seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

    # Labels as one-hot vectors
    labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])

    # Keep probability for dropout
    keep_prob = tf.placeholder(dtype=tf.float32)

    # Define the LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_WEIGHTS, initializer=tf.contrib.layers.xavier_initializer())
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)

    # Create the RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_cell,
        cell_bw=lstm_cell,
        dtype=tf.float32,
        sequence_length=seq_lens,
        inputs=inputs
    )

    # Get the final output
    output = tf.concat(1, [states[0][1], states[1][1]])

    scores = tf.contrib.layers.linear(output, NUM_CLASSES)
    scores = tf.nn.tanh(scores)

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, labels))
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Predictions
    predictions = tf.nn.softmax(scores)

    # Calculate accuracy
    pred_answers = tf.argmax(scores, axis=1)
    correct_answers = tf.argmax(labels, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_answers, correct_answers), tf.float32))

    return_dict = {
        "inputs": inputs,
        "labels": labels,
        "keep_prob": keep_prob,
        "loss": loss,
        "opt": opt,
        "prediction_probs": predictions,
        "prediction_class": pred_answers,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "sequence_lengths": seq_lens
    }

    return return_dict

if __name__ == '__main__':

    # Construct the computation graph
    graph_outputs = graph()

    # Assign graph outputs

    # Holds input sentences
    sentence_input = graph_outputs["inputs"]

    # Holds labels in one-hot format
    labels = graph_outputs["labels"]

    # Holds keep probability for dropout
    keep_prob = graph_outputs["keep_prob"]

    # Loss function to minimize
    loss = graph_outputs["loss"]

    # Optimisation operation
    opt = graph_outputs["opt"]

    # Predictions as probabilities
    predictions = graph_outputs["prediction_probs"]

    # Argmaxed predictions and labels
    pred_answers = graph_outputs["prediction_class"]
    correct_answers = graph_outputs["correct_answers"]

    # Accuracy operation
    accuracy = graph_outputs["accuracy"]

    # Sequence lengths
    seq_lens = graph_outputs["sequence_lengths"]

    with tf.Session() as sess:

        # Initialise all variables
        sess.run(tf.global_variables_initializer())

        # Saving object
        saver = tf.train.Saver()

        data = get_data()

        test_len = int(len(data) * (1/3))
        train_test_len = int(len(data) * (1/20))
        test_data = data[0:test_len]
        train_data = data[test_len:]
        train_test_data = test_data[0:train_test_len]

        print("Length of Training Data: ", len(train_data))
        print("Length of Testing Data: ", len(test_data))

        num_batches = int(len(train_data) / BATCH_SIZE)

        if not PRETRAINED:

            # Accuracies and losses for training curves
            accuracies = []
            losses = []

            # Lowest loss
            lowest_loss = 1000

            # Number of steps since the model was saved
            steps_since_save = 0

            # Breaks out of the training loop if true for early stopping
            breakout = False

            for epoch in range(MAX_EPOCHS):

                if breakout:
                    break

                for batch in range(num_batches):

                    print("Running Batch: ", batch, " / ", num_batches, end="\r")
                    sys.stdout.flush()

                    # Sample a random batch of data
                    batch_data = random.sample(train_data, BATCH_SIZE)

                    # Turn batch into 3D matrix and one-hot labels
                    batch_inputs, lens, batch_labels = batch2input(batch_data, BATCH_SIZE)

                    # Create the feed_dict
                    feed_dict = {
                        sentence_input: batch_inputs,
                        labels: batch_labels,
                        seq_lens: lens,
                        keep_prob: 0.5
                    }

                    # Runs optimisation
                    sess.run(opt, feed_dict=feed_dict)

                    if batch % DISPLAY_EVERY == 0:

                        print("\n*** Running Testing ***")

                        # Get the batch of test data
                        batch_data = train_test_data

                        # Turn batch into 3D matrix and one-hot labels
                        batch_inputs, lens, batch_labels = batch2input(batch_data, len(batch_data))

                        # Create the feed_dict
                        feed_dict = {
                            sentence_input: batch_inputs,
                            labels: batch_labels,
                            seq_lens: lens,
                            keep_prob: 1
                        }

                        # Run accuracy and loss
                        l, acc = sess.run([loss, accuracy], feed_dict=feed_dict)

                        accuracies.append(acc)
                        losses.append(l)

                        print("\n\n**** EPOCH ", epoch, " ****")
                        print("Model Accuracy on Iteration ", batch, " is: ", acc)
                        print("Model Loss on Iteration ", batch, " is: ", l)

                        if l < lowest_loss:
                            lowest_loss = l
                            print(">> Saving Model <<")
                            saver.save(sess, SAVE_PATH)
                            print(">> Model Saved <<")
                            steps_since_save = 0
                        else:
                            steps_since_save += 1

                        print()

                        if steps_since_save > MAX_STEPS_SINCE_SAVE:
                            print(">>>> MODEL CONVERGED, STOPPING EARLY <<<<")
                            breakout = True
                            break

            with open(SAVE_DIR + "loss_" + str(HIDDEN_LAYER_WEIGHTS) + ".pkl", "wb") as f:
                pickle.dump(losses, f)

            with open(SAVE_DIR + "accuracies_" + str(HIDDEN_LAYER_WEIGHTS) + ".pkl", "wb") as f:
                pickle.dump(accuracies, f)

            plt.plot(accuracies)
            plt.ylabel("Accuracy")
            plt.xlabel("Training Iteration")
            plt.title(MODEL_NAME + " Test Accuracy During Training")
            plt.savefig(SAVE_DIR + MODEL_NAME + "_accuracy_" + str(HIDDEN_LAYER_WEIGHTS) + ".png")
            plt.show()

            plt.plot(losses)
            plt.ylabel("Loss")
            plt.xlabel("Training Iteration")
            plt.title(MODEL_NAME + " Test Loss During Training")
            plt.savefig(SAVE_DIR + MODEL_NAME + "_loss_" + str(HIDDEN_LAYER_WEIGHTS) + ".png")
            plt.show()

        # ======== Test the model ========

        # Restore the trained parameters
        saver.restore(sess, SAVE_PATH)

        # Get the batch of test data
        len_batch_data = int(len(test_data) * 1 / 4)
        test_1 = [(sent, y) for sent, _, y in test_data[0:len_batch_data]]
        test_2 = [(sent, y) for sent, _, y in test_data[len_batch_data:(2*len_batch_data)]]
        test_3 = [(sent, y) for sent, _, y in test_data[(2*len_batch_data):(3*len_batch_data)]]
        test_4 = [(sent, y) for sent, _, y in test_data[(3*len_batch_data):]]

        t1_sents = [x for x, _ in test_1]
        t2_sents = [x for x, _ in test_2]
        t3_sents = [x for x, _ in test_3]
        t4_sents = [x for x, _ in test_4]

        t1_feats = [feat for _, feat, _ in test_data[0:len_batch_data]]
        t2_feats = [feat for _, feat, _ in test_data[len_batch_data:(2*len_batch_data)]]
        t3_feats = [feat for _, feat, _ in test_data[(2*len_batch_data):(3*len_batch_data)]]
        t4_feats = [feat for _, feat, _ in test_data[(3*len_batch_data):]]

        # Batch 1
        print("Test 1")
        batch_data = test_1

        avg_l = 0
        avg_acc = 0
        avg_p = 0
        avg_r = 0
        avg_f = 0

        # Turn batch into 3D matrix and one-hot labels
        batch_inputs, lens, batch_labels = batch2input(batch_data, len(test_1))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true, prob = sess.run([loss, accuracy, pred_answers, correct_answers, predictions],
                                                feed_dict=feed_dict)

        t1_sents_pred_y = [x for x in zip(t1_sents, t1_feats, y_pred, y_true, prob) if x[2] != x[3]]

        # Compute Precision, Recall and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        avg_l += l
        avg_acc += acc
        avg_p += precision
        avg_r += recall
        avg_f += f1

        # Batch 2
        print("Test 2")
        batch_data = test_2

        # Turn batch into 3D matrix and one-hot labels
        batch_inputs, lens, batch_labels = batch2input(batch_data, len(test_2))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true, prob = sess.run([loss, accuracy, pred_answers, correct_answers, predictions], feed_dict=feed_dict)

        t2_sents_pred_y = [x for x in zip(t2_sents, t2_feats, y_pred, y_true, prob) if x[2] != x[3]]

        # Compute Precision, Recall and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        avg_l += l
        avg_acc += acc
        avg_p += precision
        avg_r += recall
        avg_f += f1

        # Batch 3
        print("Test 3")
        batch_data = test_3

        # Turn batch into 3D matrix and one-hot labels
        batch_inputs, lens, batch_labels = batch2input(batch_data, len(test_3))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true, prob = sess.run([loss, accuracy, pred_answers, correct_answers, predictions],
                                                feed_dict=feed_dict)

        t3_sents_pred_y = [x for x in zip(t3_sents, t3_feats, y_pred, y_true, prob) if x[2] != x[3]]

        # Compute Precision, Recall and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        avg_l += l
        avg_acc += acc
        avg_p += precision
        avg_r += recall
        avg_f += f1

        # Batch 4
        print("Test 4")
        batch_data = test_4

        # Turn batch into 3D matrix and one-hot labels
        batch_inputs, lens, batch_labels = batch2input(batch_data, len(test_4))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true, prob = sess.run([loss, accuracy, pred_answers, correct_answers, predictions],
                                                feed_dict=feed_dict)

        t4_sents_pred_y = [x for x in zip(t4_sents, t4_feats, y_pred, y_true, prob) if x[2] != x[3]]

        # Compute Precision, Recall and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        avg_l += l
        avg_acc += acc
        avg_p += precision
        avg_r += recall
        avg_f += f1

        if ANALYSE_ERRORS:

            error_sentences = t1_sents_pred_y + t2_sents_pred_y + t3_sents_pred_y + t4_sents_pred_y

            false_positives = [x for x in error_sentences if x[2] == 1 and x[3] == 0]

            false_negatives = [x for x in error_sentences if x[2] == 0 and x[3] == 1]

            section_counts_fp = [0, 0, 0, 0, 0, 0, 0]
            section_counts_fn = [0, 0, 0, 0, 0, 0, 0]
            names = ["Highlight", "Abstract", "Introduction", "Results/Analysis/Discussion", "Method", "Conclusion",
                     "Other"]

            for sent, feats, pred, y, prob in false_positives:
                section = feats[-1]
                section_counts_fp[section] += 1

            for sent, feats, pred, y, prob in false_negatives:
                section = feats[-1]
                section_counts_fn[section] += 1

            print("====> FALSE POSITIVE SECTION ANALYSIS <====")
            for item in zip(names, section_counts_fp):
                print(item)

            print("\n")
            print("====> FALSE NEGATIVE SECTION ANALYSIS <====")
            for item in zip(names, section_counts_fn):
                print(item)

            print("\n")
            print("Number of False Positives: ", len(false_positives))
            print("Number of False Negatives: ", len(false_negatives))

            sys.exit()

            error_sentences = t1_sents_pred_y + t2_sents_pred_y + t3_sents_pred_y + t4_sents_pred_y

            false_positives = [x for x in error_sentences if x[1] == 1 and x[2] == 0]

            false_negatives = [x for x in error_sentences if x[1] == 0 and x[2] == 1]

            print("Number of False Positives: ", len(false_positives))
            print("Number of False Negatives: ", len(false_negatives))

            errors_to_analyse = random.sample(false_positives, 50) + random.sample(false_negatives, 50)

            out_str = ""

            i = 0
            for sent, pred, y, prob in errors_to_analyse:

                out_str += "\nSentence: " + str(i) + "\n"
                print("\nSentence: ", i, "\n")

                if pred == 1 and y == 0:

                    out_str += "====> FALSE POSITIVE <====\n"
                    out_str += "----> SENTENCE:\n" + " ".join(sent) + "\n"
                    out_str += "----> PREDICTED: " + str(pred) + " ACTUAL: " + str(y) + " WITH CONFIDENCE: " + \
                               str(prob[1]) + "\n"

                    print("====> FALSE POSITIVE <====")
                    print("----> SENTENCE:\n", " ".join(sent))
                    print("----> PREDICTED: ", pred, " ACTUAL: ", y, " WITH CONFIDENCE: ", prob)

                else:

                    out_str += "====> FALSE NEGATIVE <====\n"
                    out_str += "----> SENTENCE:\n" + " ".join(sent) + "\n"
                    out_str += "----> PREDICTED: " + str(pred) + " ACTUAL: " + str(y) + " WITH CONFIDENCE: " + str(
                        prob[0]) + "\n"

                    print("====> FALSE NEGATIVE <====")
                    print("----> SENTENCE:\n", " ".join(sent))
                    print("----> PREDICTED: ", pred, " ACTUAL: ", y, " WITH CONFIDENCE: ", prob)

                i += 1

            with open(BASE_DIR + "/Analysis/Error_Analysis/sents_lstm.txt", "w") as f:
                f.write(out_str)

            wait()


        acc = avg_acc / 4
        l = avg_l / 4
        precision = avg_p / 4
        recall = avg_r / 4
        f1 = avg_f / 4

        print("\n>>>> FINAL TESTING ACCURACY AND LOSS FOR " + MODEL_NAME + " <<<<")
        print(">> Hidden Layer Weights: ", Color.YELLOW, HIDDEN_LAYER_WEIGHTS, Color.END)
        print(">> Accuracy: ", Color.CYAN, acc, Color.END)
        print(">> Loss: ", Color.YELLOW, l, Color.END)
        print(">> Precision: ", Color.YELLOW, precision, Color.END)
        print(">> Recall: ", Color.YELLOW, recall, Color.END)
        print(">> F1: ", Color.PURPLE, f1, Color.END)
        print()

        # Write these values to a text file
        with open(SAVE_DIR + "final_test.txt", "wb") as f:
            f.write(str(acc) + "\n")
            f.write(str(l) + "\n")
            f.write(str(precision) + "\n")
            f.write(str(recall) + "\n")
            f.write(str(f1) + "\n")

# ======================================