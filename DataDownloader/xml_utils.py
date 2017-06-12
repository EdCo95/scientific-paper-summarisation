#!/usr/bin/python3

# ======================================================================================================================
# ====> XML PARSING CLASS

import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import xml.sax
import collections
import os
from operator import itemgetter

section_title_counts = collections.defaultdict(float)

class PubHandler(xml.sax.ContentHandler):
   def __init__(self):
       # content of each publication document
        self.id = ""
        self.journalname = ""  # journal name
        self.openaccess = ""   # open access or not
        self.pubdate = ""      # publication date date
        self.title = ""        # title
        self.authors = []      # list of authors
        self.keyphrases = []   # author-defined keyphrases
        self.abstract = ""     # abstract
        self.highlights = []   # highlight (i.e. author-defined summary statements)
        self.text = collections.OrderedDict()  # text
        self.captions = []     # image and table captions
        self.bib_entries = []  # bib entries

        # Test
        self.section_title = ""

        # temporary variables, ignore
        self.CurrentData = ""
        self.highlightflag = False
        self.textbuilder_highlight = []
        self.inpara = False
        self.textbuilder = []
        self.textbuilder_abstract = []
        self.paraid = 0
        self.inabstract = False
        self.textbuilder_title = []
        self.intitle = False
        self.textbuilder_captions = []
        self.incaption = False
        self.textbuilder_bib = []
        self.inbib = False

        # Class settings
        self.DEBUG = False

        # Test
        self.inkeywords = False
        self.insectiontitle = False
        self.textbuilder_sectiontitle = []
        # Used to check whether to include a section title
        self.permitted_titles = {"introduction",
                                 "method", 
                                 "methods", 
                                 "results", 
                                 "discussion",
                                 "discussions", 
                                 "conclusion", 
                                 "conclusions",
                                 "acknowledgements",
                                 "acknowledgments",
                                 "references",
                                 "results and discussion",
                                 "related work",
                                 "experimental results",
                                 "acknowledgment",
                                 "acknowledgement",
                                 "literature review",
                                 "experiments",
                                 "background",
                                 "methodology",
                                 "conclusions and future work",
                                 "related works",
                                 "summary",
                                 "limitations",
                                 "procedure",
                                 "material and methods",
                                 "discussion and conclusion",
                                 "implementation",
                                 "evaluation",
                                 "performance evaluation",
                                 "experiments and results",
                                 "overview",
                                 "experimental design",
                                 "discussion and conclusions",
                                 "results and discussions",
                                 "motivation",
                                 "proposed method",
                                 "analysis",
                                 "future work",
                                 "results and analysis",
                                 "implementation details"}

   def startElement(self, tag, attributes):
       # Call when an element starts
       self.CurrentData = tag
       v = attributes.get("class")  # returns "None" if not contained
       if v == "author-highlights":
           self.highlightflag = True
       if tag == "ce:para":
           self.inpara = True
           self.paraid += 1  # attributes.get("id")  # there isn't always one, so let's use a counter instead
       if tag == "ce:abstract" and self.highlightflag == False:
           self.inabstract = True
       if tag == "ce:title" or tag == "dc:title":
           self.intitle = True
       if tag == "ce:caption":
           self.incaption = True
       if tag == "ce:bib-reference":
           self.inbib = True

        # Test
       if tag == "ce:section-title":
          self.insectiontitle = True
          self.paraid += 1

   def endElement(self, tag):
       # Call when an elements ends
       self.CurrentData = ""
       if tag == "ce:abstract" or tag == "dc:description":
           self.highlightflag = False
           self.inabstract = False
           if len(self.textbuilder_abstract) > 0:
               para = "".join(self.textbuilder_abstract)
               self.abstract = para
               self.textbuilder_abstract = []
       elif tag == "ce:para":
           if len(self.textbuilder_highlight) > 0:
               para = "".join(self.textbuilder_highlight)
               self.highlights.append(para)
               self.textbuilder_highlight = []
           self.inpara = False
           if len(self.textbuilder) > 0:
               para = "".join(self.textbuilder)
               self.text[self.paraid] = para
               self.textbuilder = []
       if tag == "ce:title" or tag == "dc:title":
           self.intitle = False
           if len(self.textbuilder_title) > 0:
               para = "".join(self.textbuilder_title)
               self.title = para
               self.textbuilder_title = []
       if tag == "ce:caption":
           self.incaption = False
           if len(self.textbuilder_captions) > 0:
               caption = " ".join(self.textbuilder_captions)
               self.captions.append(caption)
               self.textbuilder_captions = []
       if tag == "ce:bib-reference":
           if len(self.textbuilder_bib) > 0:
               bibref = " ".join(self.textbuilder_bib)
               self.bib_entries.append(bibref)
               self.textbuilder_bib = []

        # Test
       if tag == "ce:section-title":
           self.insectiontitle = False
           if len(self.textbuilder) > 0:
               para = "".join(self.textbuilder)
               self.text[self.paraid] = para
               self.textbuilder = []


   def characters(self, content):
       # Call when a character is read
       if self.CurrentData == "dc:identifier":
           self.id = content
       elif self.CurrentData == "prism:publicationName":
           self.journalname = content
       elif self.CurrentData == "openaccess":
           self.openaccess = content
       elif self.CurrentData == "prism:coverDate":
           self.pubdate = content
       elif self.intitle == True:
           self.textbuilder_title.append(content)
       elif self.CurrentData == "dc:creator":
           self.authors.append(content)
       elif self.CurrentData == "dcterms:subject":
           self.keyphrases.append(content)
       elif (self.CurrentData == "dc:description") or (self.inabstract == True and self.highlightflag == False):
           if content.startswith("Abstract"):
               content = content.replace("Abstract", "", 1)
           self.textbuilder_abstract.append(content)
       elif self.highlightflag == True:
           if content.startswith("Highlights"):
               content = content.replace("Highlights", "", 1)
           if content.startswith("•"):
               content = content.replace("•", "", 1)
           self.textbuilder_highlight.append(content)
       elif self.inpara == True and self.highlightflag == False:
           self.textbuilder.append(content)
       elif self.incaption == True:
           self.textbuilder_captions.append(content)
       elif self.inbib == True:
           self.textbuilder_bib.append(content)

       # Test
       elif self.insectiontitle == True:
           section_title_counts[content] += 1.0
           if content.lower() in self.permitted_titles:
              if self.DEBUG:
                print("Adding content: ", content)
              self.textbuilder.append(("@&#" + content.upper() + "@&#"))

# ======================================================================================================================

def parseXML(fpath, write_loc):
    '''
    Parse XML files to retrieve full publication text
    :param fpath: full path to the XML file to read
    :param write_loc: directory location to write the parsed XML to
    :return:
    '''

    OUTPUT = True
    DEBUG = False

    if not os.path.isdir(write_loc):
        print(">>>> The write location needs to be a directory <<<<")
        print("Please change write path to a directory and try again.")
        print(">> EXITING <<")
        exit()

    FILE_TO_WRITE = write_loc

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namespaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = PubHandler()
    parser.setContentHandler(Handler)

    # parse document
    parser.parse(fpath)

    if DEBUG:
      # access the different parts of the publication
      print("Title:", Handler.title)
      print("\n")
      for h in Handler.highlights:
          print(h, "\n")
      print("\n")
      print("Keyphrases: ")
      for k in Handler.keyphrases:
          print(k, "\n")
      print("\n")
      print("Abstract:", Handler.abstract, "\n\n")
      for n, t in Handler.text.items():
          print("Text:", t, "\n\n")

      #print(Handler.section_title)
      input("Press enter to continue...")

    if OUTPUT:
      # Open the file for writing

      # Get rid of ".xml" and add ".txt"
      filename = fpath.split("/")[-1]
      temp = filename.split(".")
      filename = temp[0] + ".txt"
      name = filename

      # Open the file
      f = open(write_loc + name, "w")

      # Write the title
      title = "@&#MAIN-TITLE@&#" + Handler.title + "\n\n"
      f.write(title)

      # Write the highlights
      highlights_title = "@&#HIGHLIGHTS@&#\n\n"
      f.write(highlights_title)
      for h in Handler.highlights:
          string = h + "\n\n"
          f.write(string)

      # Write the keyphrases
      keyphrases_title = "@&#KEYPHRASES@&#\n\n"
      f.write(keyphrases_title)
      for k in Handler.keyphrases:
          string = k + "\n\n"
          f.write(string)

      # Write the abstract
      abstract_title = "@&#ABSTRACT@&#\n\n"
      f.write(abstract_title)
      string = Handler.abstract + "\n\n"
      f.write(string)

      # Write the main test
      for n, t in Handler.text.items():
        string = t + "\n\n"
        f.write(string)
      
      # Close the file
      f.close()

      if DEBUG:
        input("Press enter to continue...")

def parseXMLAll(dirpath, write_loc):
    """
    Takes a directory full of XML paper files, then parses all of them into the form used by the summariser - namely
    where each section title is delineated by the marker "@&#". This means we can simply read the whole file as a string,
    then split it on "@&#" - a symbol very unlikely to occur in the paper itself. This will then give a list of strings.
    It is trivial to then parse this into a dictionary with the keys as section titles and the values as the section
    text. See the file useful_functions.py for a function which does this.
    :param dirpath: the path to the directory containing all of the XML files to parse.
    :param write_loc: the location to write the parsed files.
    """

    directory = os.listdir(dirpath)
    num_files = len([x for x in directory])
    for i, f in enumerate(directory):
        if f.endswith(".xml"):
            print("Parsing file %d of %d"%(i+1, num_files), end="\r")
            parseXML(os.path.join(dirpath, f), write_loc)
    print()

if __name__ == '__main__':
    SOURCE_DIR = "/DataDownloader/Test2/"
    WRITE_DIR = "/DataDownloader/Test3/"
    BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"
    parseXMLAll(BASE_DIR + SOURCE_DIR, BASE_DIR + WRITE_DIR)
    exit()


    count = 0
    num_files = len([name for name in os.listdir(SOURCE_DIR) if name.endswith(".xml")])
    loading_section_size = num_files / 30
    for filename in os.listdir(SOURCE_DIR):

        # Pretty loading bar
        print("Processing Files: [", end="")
        for i in range(31, -1, -1):
          if count > i * loading_section_size:
            for j in range(0, i):
              print("-", end="")
            for j in range(i, 30):
              print(" ", end="")
            break;
        print("] ", count, end="\r")

        # Parse the file
        if filename.endswith(".xml"):
            parseXML(filename)
            count += 1

    print(count)
    # Check the most commonly used section titles
    for key, value in sorted(section_title_counts.items(), key=itemgetter(1), reverse=True):
      print(key, " ", value)


"""
if self.insectiontitle:
     print("Content is: ", content)
     #print("Tag Is: ", tag)
     print("Highlight Flag: ", self.highlightflag)
     print("Para Flag: ", self.inpara)
     print("Abstract Flag: ", self.inabstract)
     print("Title Flag: ", self.intitle)
     print("Caption Flag: ", self.incaption)
     print("Bib Flag: ", self.inbib)
     print("Section Title Flag: ", self.insectiontitle)
     print("\n\n")
"""           