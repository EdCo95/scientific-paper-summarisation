# Automatic Summarisation of Scientific Papers

Have you ever had to do a literature review as part of a research project and thought "I wish there was a quicker way of doing this"? This code aims to create that quicker way by developing a supervised-learning based extractive summarisation system for the summarisation of scientific papers.

## Code Description
The various code files and folders are described here. Note that the data used is not uploaded here but nonetheless the repository is still over 1GB in size. If this work is accepted at the ConLL Conference in May 2017 then a script to access the data will be uploaded.

* **Analysis** - A folder containing code used to analyse the generated summaries and create various pretty graphs. It is not essential to the functioning of the summarisers and will not work without the data.
* **DataTools** - Contains files for manipulating and preprocessing the data. The most important file is `useful_functions.py` which contains many important functions used to run the system.
* **Evaluation** - Contains code to evaluate summaries and calculate the ROUGE-L metric, with thanks to [hapribot](https://github.com/harpribot/nlp-metrics/blob/master/rouge/rouge.py).
* **Models** - Contains the code which constructs and trains each of the supervised learning modules that form the core of the summarisation system. All written in TensorFlow.
* **Summarisers** - Contains the code which takes the trained models and uses them to actually create summaries of papers.
* **Visualisations** - Contains code which visualises summaries by colouring them and saving them as HTML files. This is not essential to run the system.
* **Word2Vec** - Contains the code necessary to train the Word2Vec model used for word embeddings. The actual trained Word2Vec model is not uploaded because it is too large.
* **DataDownloader** - Contains code to download and parse the original XML paper files into the format currently used by this system - where each section title is delineated by "@&#" so the paper can easily be read and split into constituent sections by reading the whole paper as a string and splitting the string on this symbol which is very unlikely to ever occur in the text. The important file is `acquire_data.py`.

## Running the Code
Before attempting to run this code you should setup a suitable virtualenv using Python 2.7. Install all of the requirements listed in `requirements.txt` with `pip install -r requirements.txt`.

To then run this code you will need paper data in the following format: every paper is in a directory and is a `.txt` file, where the section headings of every section in the paper are surrounded on both sides by the symbol "@&#". You will also need to create a stopword list and list of permitted paper section titles and a word embedding model. Finally you will need to create dictionaries which keep bag of words representations of every paper for calculating features. Finally you will also need to update all the paths currently listed in the project so that they match your own. You will then need to update the loading functions in `useful_functions.py` to load all of these things by changing the paths to point at the correct locations.

## Other Notes
If you have read or are reading the thesis corresponding to this code, then SAFNet = SummariserNet, SFNet = SummariserNetV2, SNet = LSTM, SAF+F Ens = EnsembleSummariser, S+F Ens = EnsembleV2Summariser.