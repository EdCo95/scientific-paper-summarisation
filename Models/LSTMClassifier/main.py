from __future__ import print_function, division
import os
import dill
import pickle
import sys
import random
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from operator import itemgetter
from Dev.DataTools import useful_functions
from Dev.DataTools.useful_functions import wait, printlist, num2onehot, BASE_DIR, PAPER_SOURCE
from Dev.Evaluation.rouge import Rouge
from Dev.DataTools.DataPreprocessing.LSTMPreprocessor import LSTMPreprocessor
from Dev.DataTools.LSTM_preproc.vocab import Vocab
from Dev.DataTools.LSTM_preproc.batch import get_batches, GeneratorWithRestart, get_feed_dicts, get_feed_dicts_old
from Dev.DataTools.LSTM_preproc.map import numpify, tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample, jtr_map_to_targets
import time
import tensorflow as tf
import numpy as np

MODEL_BASE_DIR = BASE_DIR + "/Trained_Models/LSTM/"
MODEL_SAVE_PATH = BASE_DIR + "/Trained_Models/LSTM/LSTM.ckpt"
VOCAB_DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/LSTM/"
NUMBER_OF_PAPERS = len([name for name in os.listdir(PAPER_SOURCE) if name.endswith(".txt")])
LOADING_SECTION_SIZE = NUMBER_OF_PAPERS / 30

PRETRAINED = False

# The number of classes a sentence could be classified into
NUM_CLASSES = 2

# How often to display testing loss
DISPLAY_EVERY = 100

# The number of summary sentences to extract from the paper as training data
NUM_SUMMARY = 20

# The name of this model
MODEL_NAME = "LSTM"

# Directory for data
DATA_DIR = BASE_DIR + "/Data/Generated_Data/Sentences_And_SummaryBool/Abstract_Neg/AbstractNet/abstractnet_data.pkl"

# The location to save the model at
SAVE_PATH = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/"  + MODEL_NAME + "_.ckpt"

# The directory to save the model in
SAVE_DIR = BASE_DIR + "/Trained_Models/" + MODEL_NAME + "/"


def dummy_data(sentences=None):
    data = {"sentences": ["in this project we use a bilstm for extractive summarisation", "this not a summary sentence"],
            "sentence_labels": [[1, 0], [0, 1]]}  # label-length vector - [0, 1] for positive examples, [1, 0] for negative examples
    return data

def get_data():

    print("Loading Data...")
    t = time.time()

    data = useful_functions.load_cspubsumext()
    sents = []
    labs = []
    for item in data:
        sentences = item["sentences"]
        for sent, sec, y in sentences:
            sents.append(sent)
            labs.append(num2onehot(y, NUM_CLASSES))

    print("Done, took ", time.time() - t, " seconds")

    data = {
        "sentences": sents,
        "labels": labs
    }

    return data


def create_placeholders():
    sentences = tf.placeholder(tf.int32, [None, None], name="sentences")  # [batch_size, max_num_tokens]
    sentences_lengths = tf.placeholder(tf.int32, [None], name="sentences_lengths") # [batch_size]
    sentence_labels = tf.placeholder(tf.int32, [None, None], name="sentence_labels")  # [batch_size]

    placeholders = {"sentences": sentences, "sentences_lengths": sentences_lengths, "sentence_labels": sentence_labels}

    return placeholders


def bilstm_reader(placeholders, vocab_size, emb_dim, drop_keep_prob=1.0):
    # [batch_size, max_seq_length]
    sentences = placeholders['sentences']

    # [batch_size, candidate_size]
    targets = tf.to_float(placeholders['sentence_labels'])

    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("word_embeddings", [vocab_size, emb_dim], dtype=tf.float32)

    with tf.variable_scope("embedders") as varscope:
        sentences_embedded = tf.nn.embedding_lookup(embeddings, sentences)

    with tf.variable_scope("bilstm_reader") as varscope1:

        # states: (c_fw, h_fw), (c_bw, h_bw)
        outputs, states = reader(sentences_embedded, placeholders['sentences_lengths'], emb_dim,
                                scope=varscope1, drop_keep_prob=drop_keep_prob)

        # concat fw and bw outputs
        output = tf.concat(1, [states[0][1], states[1][1]])

    scores = tf.contrib.layers.linear(output, 2)  # we don't strictly need this as we've only got 2 targets
    # add non-linearity
    scores = tf.nn.tanh(scores)
    loss = tf.nn.softmax_cross_entropy_with_logits(scores, targets)
    predict = tf.nn.softmax(scores)

    predictions = tf.argmax(predict, axis=1)
    true_vals = tf.argmax(targets, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_vals), tf.float32))

    saver = tf.train.Saver()

    return scores, loss, predict, accuracy, saver


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None, drop_keep_prob=1.0):
    """Dynamic bi-LSTM reader; can be conditioned with initial state of other rnn.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial (forward, backward) states
                                  for the LSTM
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the bi-LSTM.
        States (tensor): The cell states from the bi-LSTM.
    """
    with tf.variable_scope(scope or "reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if drop_keep_prob != 1.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=drop_keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # ( (outputs_fw,outputs_bw) , (output_state_fw,output_state_bw) )
        # in case LSTMCell: output_state_fw = (c_fw,h_fw), and output_state_bw = (c_bw,h_bw)
        # each [batch_size x max_seq_length x output_size]
        return outputs, states



def train(placeholders, train_feed_dicts, test_feed_dicts, vocab, max_epochs=1000, emb_dim=64, l2=0.0, clip=None, clip_op=tf.clip_by_value, sess=None):

    # create model
    logits, loss, preds, accuracy, saver = bilstm_reader(placeholders, len(vocab), emb_dim)

    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    #optim = tf.train.AdadeltaOptimizer(learning_rate=1.0)

    if l2 != 0.0:
        loss = loss + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

    if clip is not None:
        gradients = optim.compute_gradients(loss)
        if clip_op == tf.clip_by_value:
            capped_gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                                for grad, var in gradients]
        elif clip_op == tf.clip_by_norm:
            capped_gradients = [(tf.clip_by_norm(grad, clip), var)
                                for grad, var in gradients]
        min_op = optim.apply_gradients(capped_gradients)
    else:
        min_op = optim.minimize(loss)

    tf.global_variables_initializer().run(session=sess)

    if not PRETRAINED:
        prev_loss = 1000
        steps_since_save = 0
        breakout = False

        for i in range(1, max_epochs + 1):

            if breakout:
                break

            loss_all = []
            avg_acc = 0
            count = 0

            for j, batch in enumerate(train_feed_dicts):

                print("Training iteration: ", j, end="\r")
                sys.stdout.flush()

                _, current_loss, p, acc = sess.run([min_op, loss, preds, accuracy], feed_dict=batch)

                avg_acc += acc
                count += 1

                loss_all.append(np.mean(current_loss))

                if j % DISPLAY_EVERY == 0:
                    print()
                    avg_test_acc = 0
                    avg_test_loss = 0
                    count = 0
                    for k, batch in enumerate(test_feed_dicts):
                        print("Testing iteration: ", k, end="\r")
                        sys.stdout.flush()
                        acc, l = sess.run([accuracy, loss], feed_dict=batch)
                        avg_test_acc += acc
                        avg_test_loss += np.mean(l)
                        count += 1

                    avg_test_loss /= count
                    avg_test_acc /= count

                    print("\n\t\t**** EPOCH ", i, " ****")
                    print("Test Accuracy on Iteration ", j, " is: ", avg_test_acc)
                    print("Test Loss on Iteration ", j, " is: ", avg_test_loss)

                    if avg_test_loss < prev_loss:
                        print(">> New Lowest Loss <<")
                        saver.save(sess=sess, save_path=MODEL_SAVE_PATH)
                        print(">> Model Saved <<")
                        prev_loss = avg_test_loss
                        steps_since_save = 0
                    else:
                        steps_since_save += 1

                    if steps_since_save > 10:
                        breakout = True
                        break

            l = np.mean(loss_all)

            #print('Epoch %d :' % i, l, " Accuracy: ", avg_acc / count, "\n")

    # Restore the model
    saver.restore(sess, MODEL_SAVE_PATH)

    return logits, loss, preds, accuracy, saver


def load_data(placeholders):

    train_data = get_data()

    train_data, vocab = prepare_data(train_data)

    with open(VOCAB_DATA_DIR + "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_data = numpify(train_data, pad=0)  # padding to same length and converting lists to numpy arrays
    train_feed_dicts = get_feed_dicts(train_data, placeholders, batch_size=100, inst_length=len(train_data["sentences"]))
    return train_feed_dicts, vocab


def prepare_data(data, vocab=None):
    data_tokenized = deep_map(data, tokenize, ['sentences'])
    data_lower = deep_seq_map(data_tokenized, lower, ['sentences'])
    data = deep_seq_map(data_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["sentences"])
    if vocab is None:
        vocab = Vocab()
        for instance in data["sentences"]:
            for token in instance:
                vocab(token)
    vocab.freeze()
    data_ids = deep_map(data, vocab, ["sentences"])
    data_ids = deep_seq_map(data_ids, lambda xs: len(xs), keys=['sentences'], fun_name='lengths', expand=True)

    return data_ids, vocab


def main():

    # Create the TensorFlow placeholders
    placeholders = create_placeholders()

    # Get the training feed dicts and define the length of the test set.
    train_feed_dicts, vocab = load_data(placeholders)
    num_test = int(len(train_feed_dicts) * (1 / 5))

    print("Number of Feed Dicts: ", len(train_feed_dicts))
    print("Number of Test Dicts: ", num_test)

    # Slice the dictionary list into training and test sets
    final_test_feed_dicts = train_feed_dicts[0:num_test]
    test_feed_dicts = train_feed_dicts[0:50]
    train_feed_dicts = train_feed_dicts[num_test:]

    # Do not take up all the GPU memory, all the time.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        logits, loss, preds, accuracy, saver = train(placeholders, train_feed_dicts, test_feed_dicts, vocab, sess=sess)
        print('============')

        # Test on train data - later, test on test data
        avg_acc = 0
        count = 0
        for j, batch in enumerate(final_test_feed_dicts):
            acc = sess.run(accuracy, feed_dict=batch)
            print("Accuracy on test set is: ", acc)
            avg_acc += acc
            count += 1
            print('-----')

        print("Overall Average Accuracy on the Test Set Is: ", avg_acc / count)

if __name__ == "__main__":
    main()
