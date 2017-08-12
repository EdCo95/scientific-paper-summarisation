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
    PAPER_BAG_OF_WORDS_LOC, KEYPHRASES_LOC, GLOBAL_COUNT_LOC, weight_variable, bias_variable
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

# The size of the batch of the data to feed
BATCH_SIZE = 100

# Weighting dimensions for the weighting for the sentences
SENTENCE_WEIGHTS = 100

# The number of units in the hidden layer
HIDDEN_LAYER_WEIGHTS = 128

# The number of classes to classify into
NUM_CLASSES = 2

# The network learning rate
LEARNING_RATE = 0.0001

# Maximum number of epochs
MAX_EPOCHS = 1000

# How often to display network progress and test its accuracy
DISPLAY_EVERY = 300

# How many steps the network can go before it is deemed to have converged
MAX_STEPS_SINCE_SAVE = 15

# True if the model is already trained
PRETRAINED = True

# The name of this model
MODEL_NAME = "AbstractNet"

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

# The location to save the model at
SAVE_PATH = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/" + str(HIDDEN_LAYER_WEIGHTS) + "_units/" + "abstractnet_" +\
            str(HIDDEN_LAYER_WEIGHTS) + ".ckpt"

# The directory to save the model in
SAVE_DIR = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/" + str(HIDDEN_LAYER_WEIGHTS) + "_units/"


# ==================================================

# ================ FUNCTIONS ================

# ===========================================

# ================ MAIN ================

def graph():
    """
    Function to encapsulate the construction of a TensorFlow computation graph.
    :return: input placeholders, optimisation operation, loss, accuracy, prediction operations
    """

    # Define placeholders for the data

    # The sentence to classify, has shape [batch_size x word_dimensions]
    sentence_input = tf.placeholder(tf.float32, shape=[None, WORD_DIMENSIONS])

    # The abstract of the papers the sentences come from. Has shape
    # [batch_size x word_dimensions]
    abstract_input = tf.placeholder(tf.float32, shape=[None, WORD_DIMENSIONS])

    # The labels for the sentences as one-hot vectors, of the form [batch_size x num_classes]
    labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # Keep probability for dropout
    keep_prob = tf.placeholder(tf.float32)

    # Define the computation graph

    # The sentence gate - decides which parts of the sentence are most pertinent
    sent_weight = weight_variable([WORD_DIMENSIONS, SENTENCE_WEIGHTS])
    sent_bias = bias_variable([SENTENCE_WEIGHTS])
    sentence_gate = tf.nn.sigmoid(tf.matmul(sentence_input, sent_weight) + sent_bias)

    # The abstract gate - decides which parts of the abstract are pertinent
    abstract_weight = weight_variable([WORD_DIMENSIONS, SENTENCE_WEIGHTS])
    abstract_bias = bias_variable([SENTENCE_WEIGHTS])
    abstract_gate = tf.nn.sigmoid(tf.matmul(abstract_input, abstract_weight) + abstract_bias)

    # Multiply the abstract and sentence gate outputs together
    combined_information = tf.mul(sentence_gate, abstract_gate)

    # Add a ReLU layer to this
    classification_layer_weights = weight_variable([WORD_DIMENSIONS, HIDDEN_LAYER_WEIGHTS])
    classification_layer_bias = bias_variable([HIDDEN_LAYER_WEIGHTS])
    classification_layer = tf.nn.relu(tf.matmul(combined_information, classification_layer_weights) +
                                      classification_layer_bias)

    # Add dropout
    hidden_layer_output = tf.nn.dropout(classification_layer, keep_prob)

    # Project to two classes
    final_layer_weights = weight_variable([HIDDEN_LAYER_WEIGHTS, NUM_CLASSES])
    final_layer_bias = bias_variable([NUM_CLASSES])
    output = tf.matmul(classification_layer, final_layer_weights) + final_layer_bias

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, labels))
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Predictions
    predictions = tf.nn.softmax(output)

    # Calculate accuracy
    pred_answers = tf.argmax(output, axis=1)
    correct_answers = tf.argmax(labels, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_answers, correct_answers), tf.float32))

    return sentence_input, abstract_input, labels, keep_prob, loss, opt, predictions, pred_answers, correct_answers,\
           accuracy

# Construct the computation graph
graph_outputs = graph()

# Assign graph outputs

# Holds input sentences
sentence_input = graph_outputs[0]

# Holds input abstract
abstract_input = graph_outputs[1]

# Holds labels in one-hot format
labels = graph_outputs[2]

# Holds keep probability for dropout
keep_prob = graph_outputs[3]

# Loss function to minimize
loss = graph_outputs[4]

# Optimisation operation
opt = graph_outputs[5]

# Predictions as probabilities
predictions = graph_outputs[6]

# Argmaxed predictions and labels
pred_answers = graph_outputs[7]
correct_answers = graph_outputs[8]

# Accuracy operation
accuracy = graph_outputs[9]

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    print("Loading Data...")
    t = time.time()
    data = useful_functions.load_cspubsumext()
    sentence_class_abstract = []
    for item in data:
        sentence_vecs = item["sentence_vecs"]
        abstract_vec = item["abstract_vec"]
        for sent, sec, y in sentence_vecs:
            sentence_class_abstract.append((sent, y, abstract_vec))
    data = sentence_class_abstract
    print("Done, took ", time.time() - t, " seconds")

    test_len = int(len(data) * (1/3))
    test_data = data[0:test_len]
    train_data = data[test_len:]

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

                # Sample a random batch of data
                batch_data = random.sample(train_data, BATCH_SIZE)

                # Extract the data into three numpy arrays
                batch_sentences = np.asarray([x for x, _, _ in batch_data])
                batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x, _ in batch_data])
                batch_abstracts = np.asarray([x for _, _, x in batch_data])

                # Create the feed_dict
                feed_dict = {
                    sentence_input: batch_sentences,
                    abstract_input: batch_abstracts,
                    labels: batch_labels,
                    keep_prob: 0.5
                }

                # Runs optimisation
                sess.run(opt, feed_dict=feed_dict)

                if batch % DISPLAY_EVERY == 0:

                    # Get the batch of test data
                    batch_data = test_data

                    # Extract the data into three numpy arrays
                    batch_sentences = np.asarray([x for x, _, _ in batch_data])
                    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x, _ in batch_data])
                    batch_abstracts = np.asarray([x for _, _, x in batch_data])

                    # Create the feed_dict
                    feed_dict = {
                        sentence_input: batch_sentences,
                        abstract_input: batch_abstracts,
                        labels: batch_labels,
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
    batch_data = test_data

    # Extract the data into three numpy arrays
    batch_sentences = np.asarray([x for x, _, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x, _ in batch_data])
    batch_abstracts = np.asarray([x for _, _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        abstract_input: batch_abstracts,
        labels: batch_labels,
        keep_prob: 1
    }

    # Run accuracy and loss
    l, acc, y_pred, y_true = sess.run([loss, accuracy, pred_answers, correct_answers], feed_dict=feed_dict)

    # Compute Precision, Recall and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

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

    # Get the batch of test data
    batch_data = train_data

    # Extract the data into three numpy arrays
    batch_sentences = np.asarray([x for x, _, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x, _ in batch_data])
    batch_abstracts = np.asarray([x for _, _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        abstract_input: batch_abstracts,
        labels: batch_labels,
        keep_prob: 1
    }

    # Run accuracy and loss
    l, acc, y_pred, y_true = sess.run([loss, accuracy, pred_answers, correct_answers], feed_dict=feed_dict)

    # Compute Precision, Recall and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    print("\n>>>> FINAL TRAINING ACCURACY AND LOSS FOR " + MODEL_NAME + " <<<<")
    print(">> Hidden Layer Weights: ", Color.YELLOW, HIDDEN_LAYER_WEIGHTS, Color.END)
    print(">> Accuracy: ", Color.CYAN, acc, Color.END)
    print(">> Loss: ", Color.YELLOW, l, Color.END)
    print(">> Precision: ", Color.YELLOW, precision, Color.END)
    print(">> Recall: ", Color.YELLOW, recall, Color.END)
    print(">> F1: ", Color.PURPLE, f1, Color.END)
    print()

    # Write these values to a text file
    with open(SAVE_DIR + "final_train.txt", "wb") as f:
        f.write(str(acc) + "\n")
        f.write(str(l) + "\n")
        f.write(str(precision) + "\n")
        f.write(str(recall) + "\n")
        f.write(str(f1) + "\n")

# ======================================