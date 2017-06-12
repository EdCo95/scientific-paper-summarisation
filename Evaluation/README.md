# Evaluation
Contains methods for computing evaluation metrics and evaluating summaries.
* **evaluater.py** - evaluates a list of summaries given two directories full of text files: one for gold summaries and one for generated summaries, with the files in gold and candidate summaries having matching names. Outputs a .csv and .pkl file which contains the summary score for each file.
* **rouge.py** - comes from [hapribot](https://github.com/harpribot/nlp-metrics/blob/master/rouge/rouge.py). Computed the ROUGE-L score between sentence and reference text.