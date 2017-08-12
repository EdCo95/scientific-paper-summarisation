# DataTools

This is one of the most important pieces of code for the summariser. The files and folders are:

* **DataPreprocessing** - contains `DataPreprocessor.py`, an abstract base class used to implement the Strategy design pattern for preprocessors. This will probably be deprecated soon due to issues with using the multiprocessing module with Python objects. Now contains `cspubsumext_creator.py`, which will turn the parsed papers into training data suitable to train summarisation classification models.
* **LSTM_preproc** - not currently used software to preprocess sentences and labels into feed_dicts for training an LSTM. Automatically handles bucketing and batching however is very memory intensive.
* **Reader.py** - Object which simplifies the reading of text files.
* **SentenceComparator.py** - Object used to compare sentences and see how similar they are - currently a very simplistic tool that could do with development.
* **useful_functions.py** - An extremely important file full of functions used throughout the summariser, all kept in a centralised place to make their use easier. All paths are maintained here.