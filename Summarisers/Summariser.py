import abc

class Summariser:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def summarise(self, filename):
        """
        Generates a summary of a paper.
        :param filename: the filename of the paper to summarise. The paper is in a specific form: it is a text file
                         where each section is delineated by @&#SECTION-TITLE@&#.
        :return: a summary of the paper.
        """
        pass

    @abc.abstractmethod
    def prepare_paper(self, filename):
        """
        Takes the filename of the paper to summarise and reads the paper into memory. It also puts it into the requisite
        form for summarising the paper, that is it splits the paper on the symbol "@&#" and then puts the paper into a
        dictionary where the keys are the section titles and the values are the text in that section. The values, i.e.
        the section text, will be in the form of a list of lists, where each list is a list of words corresponding to
        a sentence in that section. Depending on the model, this may be augmented in some way e.g. the sentences may
        be read in as averaged word vectors or feature vectors rather than raw words.
        :return: the scientific paper in a form suitable fo the summarisation algorithm to run on.
        """
        pass