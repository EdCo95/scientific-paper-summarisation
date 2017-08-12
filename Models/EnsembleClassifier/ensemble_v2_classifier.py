# ======== PROJECT CONFIGURATION IMPORTS ========

from __future__ import print_function, division
import sys
import os
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import time
import numpy as np
from scipy import spatial
import tensorflow as tf
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import printlist, wait, num2onehot, BASE_DIR, PAPER_SOURCE
from Dev.DataTools.DataPreprocessing.AbstractNetPreprocessor import AbstractNetPreprocessor
from Dev.Models.LSTMClassifier.lstm_classifier import graph, batch2input, SAVE_PATH, NUM_FEATURES,\
    ABSTRACT_DIMENSION, WORD_DIMENSIONS, MAX_SENT_LEN
from Dev.Models.FeaturesClassifier import features_mlp
from Dev.Models.LSTMClassifier import lstm_classifier
from Dev.Models.SummariserNetClassifier import summariser_net
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from operator import itemgetter
from sklearn import linear_model
from Dev.Evaluation.rouge import Rouge

SUMMARY_WRITE_LOC = BASE_DIR + "/Data/Generated_Data/Generated_Summaries/EnsembleSummariser/"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

# ===============================================

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

NUM_CLASSES = 2

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

    for item in data:

        sentences = item["sentences"]
        abstract_vec = item["abstract_vec"]
        features = item["sentence_features"]

        for sentence, feat in zip(sentences, features):
            sent = sentence[0]
            sec = sentence[1]
            y = sentence[2]
            sents_absvec_feats_class.append((sent, abstract_vec, feat, y))

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

# Hyperparameter to tune weight given to feature probability
C = 0.3

# Load the data
data = get_data()

# Split into test and train sets
test_len = int(len(data) * (1/3))
test_data = data[0:test_len]
train_data = data[test_len:]

# Get the batch of test data
len_batch_data = int(len(test_data) * 1 / 4)
test_1 = test_data[0:len_batch_data]
t1_labels = [y for _, _, _, y in test_1]
test_2 = test_data[len_batch_data:(2*len_batch_data)]
t2_labels = [y for _, _, _, y in test_2]
test_3 = test_data[(2*len_batch_data):(3*len_batch_data)]
t3_labels = [y for _, _, _, y in test_3]
test_4 = test_data[(3*len_batch_data):]
t4_labels = [y for _, _, _, y in test_4]

avg_acc = 0
avg_precision = 0
avg_recall = 0
avg_f1 = 0

# ========> Code from here on is summariser specific <========

tf.reset_default_graph()
computation_graph = lstm_classifier.graph()
sentence_input = computation_graph["inputs"]
seq_lens = computation_graph["sequence_lengths"]
raw_predictions = computation_graph["prediction_probs"]
labels = computation_graph["labels"]
keep_prob = computation_graph["keep_prob"]

print("Test 1")
batch_data = test_1
batch_final_labels = t1_labels

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # Restore the saved model
    saver.restore(sess, lstm_classifier.SAVE_PATH)

    # Get the matrix representation of the sentences
    batch_inputs, lens, batch_labels, batch_abstracts, batch_features = summariser_net.batch2input(batch_data, len(batch_data))

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_inputs,
        labels: batch_labels,
        seq_lens: lens,
        keep_prob: 1
    }

    # Run accuracy and loss
    raw_probs_summnet = sess.run(raw_predictions, feed_dict=feed_dict)
    prob_pos_summnet = raw_probs_summnet[:, 1]

tf.reset_default_graph()
features_graph = features_mlp.graph()
features_prediction_probs = features_graph["prediction_probs"]
sentence_input = features_graph["features_input"]
labels = features_graph["labels"]
loss = features_graph["loss"]
raw_predictions = features_graph["prediction_probs"]
pred_answers = features_graph["prediction_class"]
correct_answers = features_graph["correct_answers"]
accuracy = features_graph["accuracy"]

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # ====> Run the second graph <====
    saver.restore(sess, features_mlp.SAVE_PATH)

    batch_sentences = np.asarray([x for _, _, x, _ in test_1])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, _, _, x in test_1])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        labels: batch_labels
    }

    # Run accuracy and loss
    raw_probs_feats = sess.run(raw_predictions, feed_dict=feed_dict)
    prob_pos_feats = raw_probs_feats[:, 1]

# ====> Combine the results <====

summary = []
sents_already_added = set()

# ====> Attempt Four <====
final_probs = []

for item in zip(prob_pos_summnet, prob_pos_feats):
    prob_summNet = item[0] * (1 - C)
    prob_Features = item[1] * (1 + C)
    avg_prob = (prob_summNet + prob_Features) / 2
    final_probs.append((1-avg_prob, avg_prob))

predictions = np.argmax(np.asarray(final_probs), axis=1)

y_pred = predictions
y_true = np.asarray(batch_final_labels)

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
acc = accuracy_score(y_true, y_pred)

print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

avg_acc += acc
avg_precision += precision
avg_recall += recall
avg_f1 += f1

tf.reset_default_graph()
computation_graph = lstm_classifier.graph()
sentence_input = computation_graph["inputs"]
seq_lens = computation_graph["sequence_lengths"]
raw_predictions = computation_graph["prediction_probs"]
labels = computation_graph["labels"]
keep_prob = computation_graph["keep_prob"]

print("Test 2")
batch_data = test_2
batch_final_labels = t2_labels

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # Restore the saved model
    saver.restore(sess, lstm_classifier.SAVE_PATH)

    # Get the matrix representation of the sentences
    batch_inputs, lens, batch_labels, batch_abstracts, batch_features = summariser_net.batch2input(batch_data, len(batch_data))

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_inputs,
        labels: batch_labels,
        seq_lens: lens,
        keep_prob: 1
    }

    # Run accuracy and loss
    raw_probs_summnet = sess.run(raw_predictions, feed_dict=feed_dict)
    prob_pos_summnet = raw_probs_summnet[:, 1]

tf.reset_default_graph()
features_graph = features_mlp.graph()
features_prediction_probs = features_graph["prediction_probs"]
sentence_input = features_graph["features_input"]
labels = features_graph["labels"]
loss = features_graph["loss"]
predictions = features_graph["prediction_probs"]
pred_answers = features_graph["prediction_class"]
correct_answers = features_graph["correct_answers"]
accuracy = features_graph["accuracy"]

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # ====> Run the second graph <====
    saver.restore(sess, features_mlp.SAVE_PATH)

    batch_sentences = np.asarray([x for _, _, x, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, _, _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        labels: batch_labels
    }

    # Run accuracy and loss
    raw_probs_feats = sess.run(predictions, feed_dict=feed_dict)
    prob_pos_feats = raw_probs_feats[:, 1]

# ====> Combine the results <====

summary = []
sents_already_added = set()

# ====> Attempt Four <====
final_probs = []

for item in zip(prob_pos_summnet, prob_pos_feats):
    prob_summNet = item[0] * (1 - C)
    prob_Features = item[1] * (1 + C)
    avg_prob = (prob_summNet + prob_Features) / 2
    final_probs.append((1-avg_prob, avg_prob))

predictions = np.argmax(np.asarray(final_probs), axis=1)

y_pred = predictions
y_true = np.asarray(batch_final_labels)

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
acc = accuracy_score(y_true, y_pred)

print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

avg_acc += acc
avg_precision += precision
avg_recall += recall
avg_f1 += f1

tf.reset_default_graph()
computation_graph = lstm_classifier.graph()
sentence_input = computation_graph["inputs"]
seq_lens = computation_graph["sequence_lengths"]
raw_predictions = computation_graph["prediction_probs"]
labels = computation_graph["labels"]
keep_prob = computation_graph["keep_prob"]
# ----> Loss and Optimisation <----

# ----> Network Outputs <----

print("Test 3")
batch_data = test_3
batch_final_labels = t3_labels

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # Restore the saved model
    saver.restore(sess, lstm_classifier.SAVE_PATH)

    # Get the matrix representation of the sentences
    batch_inputs, lens, batch_labels, batch_abstracts, batch_features = summariser_net.batch2input(batch_data, len(batch_data))

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_inputs,
        labels: batch_labels,
        seq_lens: lens,
        keep_prob: 1
    }

    # Run accuracy and loss
    raw_probs_summnet = sess.run(raw_predictions, feed_dict=feed_dict)
    prob_pos_summnet = raw_probs_summnet[:, 1]

tf.reset_default_graph()
features_graph = features_mlp.graph()
features_prediction_probs = features_graph["prediction_probs"]
sentence_input = features_graph["features_input"]
labels = features_graph["labels"]
loss = features_graph["loss"]
predictions = features_graph["prediction_probs"]
pred_answers = features_graph["prediction_class"]
correct_answers = features_graph["correct_answers"]
accuracy = features_graph["accuracy"]

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # ====> Run the second graph <====
    saver.restore(sess, features_mlp.SAVE_PATH)

    batch_sentences = np.asarray([x for _, _, x, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, _, _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        labels: batch_labels
    }

    # Run accuracy and loss
    raw_probs_feats = sess.run(predictions, feed_dict=feed_dict)
    prob_pos_feats = raw_probs_feats[:, 1]

# ====> Combine the results <====

summary = []
sents_already_added = set()

# ====> Attempt Four <====
final_probs = []

for item in zip(prob_pos_summnet, prob_pos_feats):
    prob_summNet = item[0] * (1 - C)
    prob_Features = item[1] * (1 + C)
    avg_prob = (prob_summNet + prob_Features) / 2
    final_probs.append((1-avg_prob, avg_prob))

predictions = np.argmax(np.asarray(final_probs), axis=1)

y_pred = predictions
y_true = np.asarray(batch_final_labels)

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
acc = accuracy_score(y_true, y_pred)

print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

avg_acc += acc
avg_precision += precision
avg_recall += recall
avg_f1 += f1

tf.reset_default_graph()
computation_graph = lstm_classifier.graph()
sentence_input = computation_graph["inputs"]
seq_lens = computation_graph["sequence_lengths"]
raw_predictions = computation_graph["prediction_probs"]
labels = computation_graph["labels"]
keep_prob = computation_graph["keep_prob"]

print("Test 4")
batch_data = test_4
batch_final_labels = t4_labels

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # Restore the saved model
    saver.restore(sess, lstm_classifier.SAVE_PATH)

    # Get the matrix representation of the sentences
    batch_inputs, lens, batch_labels, batch_abstracts, batch_features = summariser_net.batch2input(batch_data, len(batch_data))

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_inputs,
        labels: batch_labels,
        seq_lens: lens,
        keep_prob: 1
    }

    # Run accuracy and loss
    raw_probs_summnet = sess.run(raw_predictions, feed_dict=feed_dict)
    prob_pos_summnet = raw_probs_summnet[:, 1]

tf.reset_default_graph()
features_graph = features_mlp.graph()
features_prediction_probs = features_graph["prediction_probs"]
sentence_input = features_graph["features_input"]
labels = features_graph["labels"]
loss = features_graph["loss"]
predictions = features_graph["prediction_probs"]
pred_answers = features_graph["prediction_class"]
correct_answers = features_graph["correct_answers"]
accuracy = features_graph["accuracy"]

with tf.Session() as sess:

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    # Saving object
    saver = tf.train.Saver()

    # ====> Run the second graph <====
    saver.restore(sess, features_mlp.SAVE_PATH)

    batch_sentences = np.asarray([x for _, _, x, _ in batch_data])
    batch_labels = np.asarray([num2onehot(x, NUM_CLASSES) for _, _, _, x in batch_data])

    # Create the feed_dict
    feed_dict = {
        sentence_input: batch_sentences,
        labels: batch_labels
    }

    # Run accuracy and loss
    raw_probs_feats = sess.run(predictions, feed_dict=feed_dict)
    prob_pos_feats = raw_probs_feats[:, 1]

# ====> Combine the results <====

summary = []
sents_already_added = set()

# ====> Attempt Four <====
final_probs = []

for item in zip(prob_pos_summnet, prob_pos_feats):
    prob_summNet = item[0] * (1 - C)
    prob_Features = item[1] * (1 + C)
    avg_prob = (prob_summNet + prob_Features) / 2
    final_probs.append((1-avg_prob, avg_prob))

predictions = np.argmax(np.asarray(final_probs), axis=1)

y_pred = predictions
y_true = np.asarray(batch_final_labels)

# Compute Precision, Recall and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
acc = accuracy_score(y_true, y_pred)

print("----> Accuracy: ", acc)
print("----> Precision: ", precision)
print("----> Recall: ", recall)
print("----> F1: ", f1)

avg_acc += acc
avg_precision += precision
avg_recall += recall
avg_f1 += f1

print(">>>> FINAL STATS ON ENSEMBLER V2 <<<<")
print("----> Accuracy: ", avg_acc / 4)
print("----> Precision: ", avg_precision / 4)
print("----> Recall: ", avg_recall / 4)
print("----> F1: ", avg_f1 / 4)