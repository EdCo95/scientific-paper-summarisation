# Automatic Summarisation of Scientific Papers

Have you ever had to do a literature review as part of a research project and thought "I wish there was a quicker way of doing this"? This code aims to create that quicker way by developing a supervised-learning based extractive summarisation system for the summarisation of scientific papers.

For more information on the project, please see:

Ed Collins, Isabelle Augenstein, Sebastian Riedel. [A Supervised Approach to Extractive Summarisation of Scientific Papers](https://arxiv.org/abs/1706.03946). To appear in Proceedings of CoNLL, July 2017.

Ed Collins. A supervised approach to extractive summarisation of scientific papers. UCL MEng thesis, May 2017.

## Code Description
The various code files and folders are described here. Note that the data used is not uploaded here but nonetheless the repository is still over 1GB in size.

* **Analysis** - A folder containing code used to analyse the generated summaries and create various pretty graphs. It is not essential to the functioning of the summarisers and will not work without the data.
* **Data** - Where all data should be stored. In the folder `Utility_Data` are things such as stopword lists, permitted titles and a count of how many different papers each word occurs in (used for TF-IDF; calculated automatically by `DataTools/DataPreprocessing/cspubsumext_creator.py`.
* **DataTools** - Contains files for manipulating and preprocessing the data. There are two particularly important files in this folder. `useful_functions.py` contains many important functions used to run the system. `DataPreprocessing/cspubsumext_creator.py` will take the parsed papers which are produced by the code in `DataDownloader` and preprocess them into the form used to train the models in the research automatically.
* **Evaluation** - Contains code to evaluate summaries and calculate the ROUGE-L metric, with thanks to [hapribot](https://github.com/harpribot/nlp-metrics/blob/master/rouge/rouge.py).
* **Models** - Contains the code which constructs and trains each of the supervised learning modules that form the core of the summarisation system. All written in TensorFlow.
* **Summarisers** - Contains the code which takes the trained models and uses them to actually create summaries of papers.
* **Visualisations** - Contains code which visualises summaries by colouring them and saving them as HTML files. This is not essential to run the system.
* **Word2Vec** - Contains the code necessary to train the Word2Vec model used for word embeddings. The actual trained Word2Vec model is not uploaded because it is too large.
* **DataDownloader** - Contains code to download and parse the original XML paper files into the format currently used by this system - where each section title is delineated by "@&#" so the paper can easily be read and split into constituent sections by reading the whole paper as a string and splitting the string on this symbol which is very unlikely to ever occur in the text. The important file is `acquire_data.py`.

## Running the Code
Before attempting to run this code you should setup a suitable virtualenv using Python 2.7. Install all of the requirements listed in `requirements.txt` with `pip install -r requirements.txt`.

To download the dataset and preprocess it into the form used to train the models in the paper, first run `DataDownloader/acquire_data.py`. This will download all of the papers and parse them into the form used - with sections separated by a special symbol - "@&#" - so that the papers can be read as strings then split into sections and titles by splitting on this symbol.

To turn these downloaded papers into training data, run `DataTools/DataPreprocessing/cspubsumext_creator.py`. This will take a while to run depending on your machine and number of cores (~2 hours on late 2016 MacBook Pro with dual core i7) but will handle creating all of the necessary files to train models. These are stored by default in `Data/Training_Data/`, with there being an individual JSON file for each paper and a single JSON file called `all_data.json` which is a list of all of the individual items of training data. This code now uses the ultra-fast uJSON library which reads the data much faster than the previous version which used pickle.

All of the models and summarisers should then be usable.

Be sure to check that all of the paths are correctly set! These are in `DataDownloader/acquire_data.py` for downloading papers, and in `DataTools/useful_functions.py` otherwise.

**NOTE**: The code in `DataTools/DataPreprocessing/AbstractNetPreprocessor.py` is still unpleasently inefficient and is still currently used in the summarisers themselves. The next code update will fix this and streamline the process of running the trained summarisers.

## Other Notes
If you have read or are reading the MEng thesis or CoNLL paper corresponding to this code, then SAFNet = SummariserNet, SFNet = SummariserNetV2, SNet = LSTM, SAF+F Ens = EnsembleSummariser, S+F Ens = EnsembleV2Summariser.
