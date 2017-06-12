# DataTools

This is one of the most important pieces of code for the summariser. The files and folders are:

* **DataPreprocessing** - contains `DataPreprocessor.py`, an abstract base class used to implement the Strategy design pattern for preprocessors; and `AbstractNetPreprocessor.py` which is a concrete data preprocessor. It reads each paper and transforms them into a randomized list of training data where each item in the list is a dictionary of information for each sentence including the sentence itself, features, vector of the paper's abstract and more. It also prepares papers for summarisation by reading them into an ordered list of sentences to be ranked.
* **LSTM_preproc** - not currently used software to preprocess sentences and labels into feed_dicts for training an LSTM. Automatically handles bucketing and batching however is very memory intensive.
* **Reader.py** - Object which simplifies the reading of text files.
* **SentenceComparator.py** - Object used to compare sentences and see how similar they are - currently a very simplistic tool that could do with development.
* **useful_functions.py** - An extremely important file full of functions used throughout the summariser, all kept in a centralised place to make their use easier.