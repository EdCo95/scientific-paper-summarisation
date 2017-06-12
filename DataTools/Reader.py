# Class to simplify the reading of text files in Python.
# Created by Ed Collins on Tuesday 7th June 2016.

from nltk.tokenize import sent_tokenize
import os

class Reader(object):
    """Simplifies the reading of files to extract a list of strings. Simply pass in the name of the file and it will automatically be read, returning you a list of the strings in that file."""

    def __init__(self):
        """\"name\" is the name of the file to open. The file should be a series of lines, each line separated with a newline character."""
        #self.name = name;
        pass

    def open_file(self, filename):
        """Reads the file given by the file name and returns its contents as a list of seperate strings, each string being a line in the file."""
        with open(filename) as fp:
            contents = fp.read().split("\n")

        return contents

    def open_file_single_string(self, filename):
        """Reads the file given by the file name and returns its contents as a list of seperate strings, each string being a line in the file."""
        with open(filename) as fp:
            contents = fp.read()

        fp.close()
        return contents

    def read_folder(self, folder_name, number_of_files_to_read=10000):
        """
        Reads all files in a directory, splits them into sentences and puts these sentences in a list to return.
        Args:
            folder_name = the name of the folder to read files from
            number_of_files_to_read = optional parameter for how many files in a directory to read
        Returns:
            A list of all sentences from all text files in the folder
        """
        count = 0
        all_sentences = []
        for filename in os.listdir(folder_name):
            if filename.endswith(".txt") and count < number_of_files_to_read:
                main_text_to_open = folder_name + "/" + filename
                main_text = self.open_file_single_string(main_text_to_open)
                udata = main_text.decode("utf-8")
                main_text = udata.encode("ascii", "ignore")
                sentences = sent_tokenize(main_text)
                for sentence in sentences:
                    all_sentences.append(sentence)
            count += 1
        return all_sentences
