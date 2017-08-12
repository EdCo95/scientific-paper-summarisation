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
NUM_FEATURES = 7

# The size of the batch of the data to feed
BATCH_SIZE = 100

# Weighting dimensions for the weighting for the sentences
SENTENCE_WEIGHTS = 200

# The number of units in the hidden layer
HIDDEN_LAYER_WEIGHTS = 128
FINAL_LAYER_WEIGHTS = 1024

# The number of classes to classify into
NUM_CLASSES = 2

# The network learning rate
LEARNING_RATE = 0.0001

# Maximum number of epochs
MAX_EPOCHS = 1000

# How often to display network progress and test its accuracy
DISPLAY_EVERY = 100

# How many steps the network can go without improving before it is deemed to have converged
MAX_STEPS_SINCE_SAVE = 15

# The maximum length that a sentence could be (sentences longer than this are discarded)
MAX_SENT_LEN = 100

# The size of the validation set of data
VALIDATION_SIZE = 3000

# The vocabulary of words in the wor2vec model. Words not in the vocab are given vectors of all zeros
VOCAB = set(WORD2VEC.index2word)

# True if the model is already trained
PRETRAINED = False

# The name of this model
MODEL_NAME = "LowDataSummariserNet"

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

# The location to save the model at
SAVE_PATH = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/" + str(HIDDEN_LAYER_WEIGHTS) + "_units/" + MODEL_NAME + "_" + ".ckpt"

# The directory to save the model in
SAVE_DIR = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/" + str(HIDDEN_LAYER_WEIGHTS) + "_units/"


# ==================================================

# ================ FUNCTIONS ================

def get_data():
    """
    Loads the data from the data directory given above and puts it into the form required by the summarisers. In this
    summariser the data we require is: the raw sentences, the abstract and the features.
    :return: The data, but discarding the sentences longer than the maximum length.
    """

    print("Loading Data...")
    t = time.time()

    # The data is a pickled object
    data = useful_functions.load_cspubsumext()

    # Data list
    sents_absvec_feats_class = []
    pos_count = 0
    for item in data:

        sentences = item["sentences"]
        abstract_vec = item["abstract_vec"]
        features = item["sentence_features"]

        for sentence, feats in zip(sentences, features):
            if feats[7] == 0:
                sent = sentence[0]
                sec = sentence[1]
                y = sentence[2]
                new_feats = (feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6])
                sents_absvec_feats_class.append((sent, abstract_vec, new_feats, y))
                pos_count += 1

    neg_count = 0

    for item in data:

        sentences = item["sentences"]
        abstract_vec = item["abstract_vec"]
        features = item["sentence_features"]

        for sentence, feats in zip(sentences, features):
            if sentence[2] == 0 and neg_count < pos_count:
                sent = sentence[0]
                sec = sentence[1]
                y = sentence[2]
                new_feats = (feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6])
                sents_absvec_feats_class.append((sent, abstract_vec, new_feats, y))
                neg_count += 1

    data = sents_absvec_feats_class

    print("Done, took ", time.time() - t, " seconds")

    print("Processing Data...")

    new_data = []
    for sent, abs_vec, feat, y in data:
        if len(sent) > MAX_SENT_LEN:
            new_sent = sent[0:MAX_SENT_LEN]
        else:
            new_sent = sent
        new_data.append((new_sent, abs_vec, feat, y))

    return new_data

def batch2input(batch_data, num_items):
    """
    :param batch_data: of form [(sentence, abstract_vector, feature_vector, label)]
    :return: 3D matrix of embedded sentences, sentence lengths, labels, abstracts, features
    """
    batch_labels = [num2onehot(y, NUM_CLASSES) for _, _, _, y in batch_data]
    lens = [len(x) for x, _, _, _ in batch_data]
    batch_sentences = [x for x, _, _, _ in batch_data]
    batch_abstracts = [x for _, x, _, _ in batch_data]
    batch_features = [x for _, _, x, _ in batch_data]
    inputs_matrix = np.zeros((num_items, MAX_SENT_LEN, WORD_DIMENSIONS), dtype=np.float32)
    for i, sentence in enumerate(batch_sentences):
        pos = 0
        for word in sentence:
            if word in VOCAB:
                vec = WORD2VEC[word]
                inputs_matrix[i, pos, :] = vec
            pos += 1
    return inputs_matrix, lens, batch_labels, batch_abstracts, batch_features

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

def graph_I():
    """
    Function to encapsulate the construction of a TensorFlow computation graph.
    :return: input placeholders, optimisation operation, loss, accuracy, prediction operations
    """

    # ----> Define placeholders <----

    # Input has shape [batch_size x num_steps x num_input (word vector dimensions)]
    sentence_input = tf.placeholder(dtype=tf.float32, shape=[None, MAX_SENT_LEN, WORD_DIMENSIONS])

    # Abstract input
    abstract_input = tf.placeholder(dtype=tf.float32, shape=[None, WORD_DIMENSIONS])

    # Features input
    features_input = tf.placeholder(dtype=tf.float32, shape=[None, NUM_FEATURES])

    # The lengths of each of the sentences
    seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

    # Labels as one-hot vectors
    labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])

    # Keep probability for dropout
    keep_prob = tf.placeholder(dtype=tf.float32)

    # ----> LSTM to Read Sentences <----

    # Define the LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_WEIGHTS, initializer=tf.contrib.layers.xavier_initializer())
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)

    # Create the RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_cell,
        cell_bw=lstm_cell,
        dtype=tf.float32,
        sequence_length=seq_lens,
        inputs=sentence_input
    )

    # Get the final output has shape [batch_size x 128]
    output = tf.concat(1, [states[0][1], states[1][1]])

    # We project the LSTM output down to shape [batch_size x 100], which can then be combined with the abstract vector
    lstm_output = tf.nn.relu(tf.contrib.layers.linear(output, WORD_DIMENSIONS))

    # ----> Abstract Gate to process the Abstract Vector <----

    # Apply linear layer and ReLU activation
    abstract_gate = tf.nn.relu(tf.contrib.layers.linear(abstract_input, WORD_DIMENSIONS))

    # ----> Combine the Abstract Information with the LSTM Output <----

    # Take the Hadamard product of the abstract and the LSTM output
    combined_sent_abstract = tf.mul(lstm_output, abstract_gate)

    # Pass this through an activation function
    combined_sent_abstract = tf.nn.relu(tf.contrib.layers.linear(combined_sent_abstract, WORD_DIMENSIONS))

    # ----> Feature Gate to process the Features <----

    # Apply linear layer and ReLU activation
    feature_gate = tf.nn.relu(tf.contrib.layers.linear(features_input, NUM_FEATURES))

    # ----> Combine the Feature Information with the Sentence and Abstract Information <----

    # Concatenate the two vectors of information
    combined_info = tf.concat_v2([combined_sent_abstract, feature_gate], axis=1)

    # ----> Pass the combined information through a final hidden layer <---

    # We will use a hidden layer with 1024 units
    final_hidden_layer = tf.nn.relu(tf.contrib.layers.linear(combined_info, FINAL_LAYER_WEIGHTS))

    # ----> Project the final output down to the number of classes <----

    scores = tf.contrib.layers.linear(final_hidden_layer, NUM_CLASSES)

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
        "sentence_input": sentence_input,
        "abstract_input": abstract_input,
        "features_input": features_input,
        "sequence_lengths": seq_lens,
        "keep_prob": keep_prob,
        "labels": labels,
        "loss": loss,
        "optimisation": opt,
        "raw_predictions": predictions,
        "classification_predictions": pred_answers,
        "correct_answers": correct_answers,
        "accuracy": accuracy
    }

    return return_dict

def graph():
    """
    Function to encapsulate the construction of a TensorFlow computation graph.
    :return: input placeholders, optimisation operation, loss, accuracy, prediction operations
    """

    # ----> Define placeholders <----

    # Input has shape [batch_size x num_steps x num_input (word vector dimensions)]
    sentence_input = tf.placeholder(dtype=tf.float32, shape=[None, MAX_SENT_LEN, WORD_DIMENSIONS])

    # Abstract input
    abstract_input = tf.placeholder(dtype=tf.float32, shape=[None, WORD_DIMENSIONS])

    # Features input
    features_input = tf.placeholder(dtype=tf.float32, shape=[None, NUM_FEATURES])

    # The lengths of each of the sentences
    seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])

    # Labels as one-hot vectors
    labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])

    # Keep probability for dropout
    keep_prob = tf.placeholder(dtype=tf.float32)

    # ----> LSTM to Read Sentences <----

    # Define the LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_LAYER_WEIGHTS, initializer=tf.contrib.layers.xavier_initializer())
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)

    # Create the RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_cell,
        cell_bw=lstm_cell,
        dtype=tf.float32,
        sequence_length=seq_lens,
        inputs=sentence_input
    )

    # Get the final output has shape [batch_size x 128]
    lstm_output = tf.concat(1, [states[0][1], states[1][1]])

    # We project the LSTM output down to shape [batch_size x 100], which can then be combined with the abstract vector
    lstm_output = tf.nn.relu(tf.contrib.layers.linear(lstm_output, WORD_DIMENSIONS))

    # ----> Preprocess the abstract and features <----

    # Concatenate into one vector
    abs_feats = tf.concat_v2([abstract_input, features_input], axis=1)

    # Pass through a ReLU
    abs_feats = tf.nn.relu(tf.contrib.layers.linear(abs_feats, NUM_FEATURES + WORD_DIMENSIONS))

    # ----> Concatenate the abstract and feature information with the LSTM ouput <----

    # Concatenate with the LSTM output
    full_info = tf.concat_v2([lstm_output, abs_feats], axis=1)

    # ----> Pass the full information through a hidden layer <----

    # We will use ReLU activation and 1024 hidden units
    final_hidden_layer = tf.nn.relu(tf.contrib.layers.linear(full_info, FINAL_LAYER_WEIGHTS))

    # ----> Add dropout for regularisation <----

    final_hidden_layer = tf.nn.dropout(final_hidden_layer, keep_prob)

    # ----> Project the final output down to the number of classes <----

    scores = tf.contrib.layers.linear(final_hidden_layer, NUM_CLASSES)

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
        "sentence_input": sentence_input,
        "abstract_input": abstract_input,
        "features_input": features_input,
        "sequence_lengths": seq_lens,
        "keep_prob": keep_prob,
        "labels": labels,
        "loss": loss,
        "optimisation": opt,
        "raw_predictions": predictions,
        "classification_predictions": pred_answers,
        "correct_answers": correct_answers,
        "accuracy": accuracy
    }

    return return_dict

if __name__ == '__main__':

    # Construct the computation graph
    graph_outputs = graph()

    # ====> Assign graph outputs <====

    # ----> Inputs to the Graph <----

    sentence_input = graph_outputs["sentence_input"]
    abstract_input = graph_outputs["abstract_input"]
    features_input = graph_outputs["features_input"]
    seq_lens = graph_outputs["sequence_lengths"]
    keep_prob = graph_outputs["keep_prob"]
    labels = graph_outputs["labels"]

    # ----> Loss and Optimisation <----

    loss = graph_outputs["loss"]
    opt = graph_outputs["optimisation"]

    # ----> Network Outputs <----

    raw_predictions = graph_outputs["raw_predictions"]
    predictions = graph_outputs["classification_predictions"]
    correct_answers = graph_outputs["correct_answers"]
    accuracy = graph_outputs["accuracy"]

    # ====> Train the Network <====

    with tf.Session() as sess:

        # Initialise all variables
        sess.run(tf.global_variables_initializer())

        # Saving object
        saver = tf.train.Saver()

        # Load the data
        data = get_data()

        # Split into test and train sets
        test_len = int(len(data) * (1/3))
        test_data = data[0:test_len]
        train_data = data[test_len:]

        # Carve out a small validation set from the test set
        validation_data = test_data[0:VALIDATION_SIZE]

        print("Length of Training Data: ", len(train_data))
        print("Length of Testing Data: ", len(test_data))

        wait()

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
                    batch_inputs, lens, batch_labels, batch_abstracts, batch_features = batch2input(batch_data, BATCH_SIZE)

                    # Create the feed_dict
                    feed_dict = {
                        sentence_input: batch_inputs,
                        abstract_input: batch_abstracts,
                        features_input: batch_features,
                        labels: batch_labels,
                        seq_lens: lens,
                        keep_prob: 0.5
                    }

                    #print(np.shape(sess.run(graph_outputs["extra"], feed_dict=feed_dict)))
                    #wait()

                    # Runs optimisation
                    sess.run(opt, feed_dict=feed_dict)

                    if batch % DISPLAY_EVERY == 0:

                        print("\n*** Running Testing ***")

                        # Get the batch of test data
                        batch_data = validation_data

                        # Turn batch into 3D matrix and one-hot labels
                        batch_inputs, lens, batch_labels, batch_abstracts, batch_features = batch2input(batch_data,
                                                                                                        len(batch_data))

                        # Create the feed_dict
                        feed_dict = {
                            sentence_input: batch_inputs,
                            abstract_input: batch_abstracts,
                            features_input: batch_features,
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

        # Test the model

        # Restore the trained parameters
        saver.restore(sess, SAVE_PATH)

        # Get the batch of test data
        len_batch_data = int(len(test_data) * 1 / 4)
        test_1 = test_data[0:len_batch_data]
        test_2 = test_data[len_batch_data:(2*len_batch_data)]
        test_3 = test_data[(2*len_batch_data):(3*len_batch_data)]
        test_4 = test_data[(3*len_batch_data):]

        # Batch 1
        print("Test 1")
        batch_data = test_1

        avg_l = 0
        avg_acc = 0
        avg_p = 0
        avg_r = 0
        avg_f = 0

        # Turn batch into 3D matrix and one-hot labels
        batch_inputs, lens, batch_labels, batch_abstracts, batch_features = batch2input(test_1, len(test_1))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            abstract_input: batch_abstracts,
            features_input: batch_features,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true = sess.run([loss, accuracy, predictions, correct_answers], feed_dict=feed_dict)

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
        batch_inputs, lens, batch_labels, batch_abstracts, batch_features = batch2input(test_1, len(test_1))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            abstract_input: batch_abstracts,
            features_input: batch_features,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true = sess.run([loss, accuracy, predictions, correct_answers], feed_dict=feed_dict)

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
        batch_inputs, lens, batch_labels, batch_abstracts, batch_features = batch2input(test_1, len(test_1))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            abstract_input: batch_abstracts,
            features_input: batch_features,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true = sess.run([loss, accuracy, predictions, correct_answers], feed_dict=feed_dict)

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
        batch_inputs, lens, batch_labels, batch_abstracts, batch_features = batch2input(test_1, len(test_1))

        # Create the feed_dict
        feed_dict = {
            sentence_input: batch_inputs,
            abstract_input: batch_abstracts,
            features_input: batch_features,
            labels: batch_labels,
            seq_lens: lens,
            keep_prob: 1
        }

        # Run accuracy and loss
        l, acc, y_pred, y_true = sess.run([loss, accuracy, predictions, correct_answers], feed_dict=feed_dict)

        # Compute Precision, Recall and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        avg_l += l
        avg_acc += acc
        avg_p += precision
        avg_r += recall
        avg_f += f1

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