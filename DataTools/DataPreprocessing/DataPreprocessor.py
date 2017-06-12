import abc
from Dev.DataTools import useful_functions
from operator import itemgetter


class DataPreprocessor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prepare_data(self):
        """
        Prepares the data from papers to be used in the classification algorithms defined by the concrete
        implementations of this class.
        :return: the data in a form suitable for learning with each of the different algorithms.
        """
        pass

    @abc.abstractmethod
    def prepare_for_summarisation(self, filename):
        """
        Prepares a single paper for summarisation as opposed to every paper. Also puts the sentences in the correct
        order according to their sequential appearance in the paper.
        :return: the sentences ready to be summarised.
        """
        pass

    def paper2orderedlist(self, filename):
        """
        Performs the first task necessary to summarise a paper: turning it into an ordered list of sentences which
        doesn't include the highlights or abstract section of the paper (as these are already summaries).
        :param filename: the filename to summarise.
        :return: the paper as an ordered list of sentences, not including abstract or highlights.
        """
        paper = useful_functions.read_in_paper(filename, sentences_as_lists=True, preserve_order=True)

        # We don't want to make any predictions for the Abstract or Highlights as these are already summaries.
        sections_to_predict_for = []
        for section, text in paper.iteritems():

            if section != "ABSTRACT" and section != "HIGHLIGHTS":
                sections_to_predict_for.append(text)

        # Sorts the sections according to the order in which they appear in the paper.
        sorted_sections_to_predict_for = sorted(sections_to_predict_for, key=itemgetter(1))

        # Creates an ordered list of the sentences in the paper
        sentence_list = []
        for sentence_text, section_position_in_paper in sorted_sections_to_predict_for:
            section_sentences = sentence_text
            for sentence in section_sentences:
                sentence_list.append(sentence)

        return sentence_list