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
NUM_FEATURES = 7

# The size of the batch of the data to feed
BATCH_SIZE = 100

# Weighting dimensions for the weighting for the sentences
SENTENCE_WEIGHTS = 100

# The number of units in the hidden layer
HIDDEN_LAYER_WEIGHTS = 32

# The number of classes to classify into
NUM_CLASSES = 2

# The network learning rate
LEARNING_RATE = 0.0001

# Maximum number of epochs
MAX_EPOCHS = 20

# How often to display network progress and test its accuracy
DISPLAY_EVERY = 300

# How many steps the network can go before it is deemed to have converged
MAX_STEPS_SINCE_SAVE = 15

# True if the model is already trained
PRETRAINED = False

# The name of this model
MODEL_NAME = "FeaturesLinear"

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

# The location to save the model at
SAVE_PATH = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/NoAbsRouge/features_linearclass.ckpt"

# The directory to save the model in
SAVE_DIR = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/NoAbsRouge/"

# ==================================================

# ================ FUNCTIONS ================

# ===========================================

# ================ MAIN ================

def graph():

    # Define placeholders for the data

    # The sentence to classify, has shape [batch_size x word_dimensions]
    sentence_input = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES])

    # The labels for the sentences as one-hot vectors, of the form [batch_size x num_classes]
    labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # Define the computation graph

    # Linear layer
    sent_weight = weight_variable([NUM_FEATURES, NUM_CLASSES])
    sent_bias = bias_variable([NUM_CLASSES])

    output = tf.matmul(sentence_input, sent_weight) + sent_bias

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, labels))
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Predictions
    predictions = tf.nn.softmax(output)

    # Calculate accuracy
    pred_answers = tf.argmax(output, axis=1)
    correct_answers = tf.argmax(labels, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_answers, correct_answers), tf.float32))

    return sentence_input, labels, loss, opt, predictions, pred_answers, correct_answers, accuracy, sent_weight

graph_outputs = graph()
sentence_input = graph_outputs[0]
labels = graph_outputs[1]
loss = graph_outputs[2]
opt = graph_outputs[3]
predictions = graph_outputs[4]
pred_answers = graph_outputs[5]
correct_answers = graph_outputs[6]
accuracy = graph_outputs[7]
sent_weight = graph_outputs[8]

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # Data is 3D - of form [sentence_vector, classification, abstract_vector]. We will ignore the abstract vector here.
    print("Loading Data...")
    t = time.time()
    data = useful_functions.load_cspubsumext()
    features_class = []
    for item in data:
        features = item["sentence_features"]
        sentences = item["sentences"]
        for sent, feats in zip(sentences, features):
            new_feats = (feats[1], feats[2], feats[3], feats[4], feats[5], feats[6], feats[7])
            features_class.append((new_feats, sent[2]))
    data = features_class
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
                #print(np.shape(batch_data))
                #printlist(batch_data)
                batch_sentences = np.asarray([x for x, _ in batch_data])
                batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x in batch_data])

                # Create the feed_dict
                feed_dict = {
                    sentence_input: batch_sentences,
                    labels: batch_labels
                }

                # Runs optimisation
                sess.run(opt, feed_dict=feed_dict)

                if batch % DISPLAY_EVERY == 0:

                    # Get the batch of test data
                    batch_data = test_data

                    # Extract the data into three numpy arrays
                    batch_sentences = np.asarray([x for x, _ in batch_data])
                    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x in batch_data])

                    # Create the feed_dict
                    feed_dict = {
                        sentence_input: batch_sentences,
                        labels: batch_labels
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
        plt.savefig(SAVE_DIR + MODEL_NAME + "_accuracy.png")
        plt.show()

        plt.plot(losses)
        plt.ylabel("Loss")
        plt.xlabel("Training Iteration")
        plt.title(MODEL_NAME + " Test Loss During Training")
        plt.savefig(SAVE_DIR + MODEL_NAME + "_loss.png")
        plt.show()

    # Test the model

    # Restore the trained parameters
    saver.restore(sess, SAVE_PATH)

    w = sess.run(sent_weight)
    with open(BASE_DIR + "/Data/Generated_Data/Weights/feature_weights_no_abs_rouge.npy", "wb") as f:
        np.save(f, w)
    print(w)
    wait()

    # Get the batch of test data
    batch_data = test_data

    # Extract the data into three numpy arrays
    batch_sentences = np.asarray([x for x, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        labels: batch_labels
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
    batch_sentences = np.asarray([x for x, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        labels: batch_labels
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