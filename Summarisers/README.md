# Summarisers

Code which actually summarises papers. Implemented as strategy design pattern with `Summariser.py` as the abstract base class. All summarisers must implement the `summarise()` method which takes a filename of a paper and returns a string summary of the paper; and `prepare_paper()` which prepares the raw paper for summarisation.