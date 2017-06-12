# ======== IMPORTS ========

import os
import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
from Dev.DataDownloader.sciencedirect_collect import downloadArticlesFromList
from Dev.DataDownloader.xml_utils import parseXMLAll

# =========================

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!! THINGS YOU NEED TO CHECK & CHANGE !!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ALL THE CONFIG VARIABLES YOU NEED TO SET TO DOWNLOAD THE DATA ARE HERE

# ====> (1) : The List of Papers to Download
#       The location of the text file containing the URLs of the papers to download. This will be read and each of the
#       papers downloaded. The default setting assumes you are running this code from the main project folder.
LOCATION_OF_PAPER_URLS_LIST = "DataDownloader/cspubsum_test_ids.txt"

# ====> (2) : The Place to Store the Papers
#       The directory in which to stored the downloaded papers, which will be in XML formet. The default assumes you are
#       running the code from the main project folder.
LOCATION_TO_STORE_XML_PAPERS = "DataDownloader/XML_Papers/"

# ====> (3) : Directory of Existing Papers
#       If you have any papers which are already downloaded in XML format, enter the directory here to combine them with
#       the newly downloaded papers. The default is set to the storage location for the downloaded papers, but setting
#       this to another directory will combine the two sets of papers. Unless you have already downloaded some papers
#       in XML format, you can ignore this.
LOCATION_OF_EXISTING_XML_PAPERS = LOCATION_TO_STORE_XML_PAPERS

# ====> (4) : The Place to Store the Parsed Papers
#       The XML papers will be parsed into ".txt" format papers so that functions provided in useful_functions.py can
#       parse them easily. This variable gives the location to store the parsed papers.
LOCATION_TO_STORE_PARSED_PAPERS = "DataDownloader/Parsed_Papers/"

# ====> (5) : An API Key
#       You will need an API key for ScienceDirect in order to access the papers. To acquire one, you will need to visit
#       the following website: https://dev.elsevier.com/user/login
#       Also note that you must run this code on the network of an institution which is subscribed to ScienceDirect
#       (most universities) otherwise only the title, abstract and keywords of each paper will be downloaded rather than
#       the whole paper.
API_KEY = ""

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ======== SANITY CHECKS ========

# Check that the place to store the XML papers exists
if not os.path.exists(LOCATION_TO_STORE_XML_PAPERS):
    os.makedirs(LOCATION_TO_STORE_XML_PAPERS)

# Check that the place to store the parsed papers exists
if not os.path.exists(LOCATION_TO_STORE_PARSED_PAPERS):
    os.makedirs(LOCATION_TO_STORE_PARSED_PAPERS)

# Check the API key is not blank
try:
    assert(API_KEY is not "")
except AssertionError:
    exit('\033[91m You must provide an API key. You can obtain one from: https://dev.elsevier.com/user/login \033[0m')

# ===============================

# ======== GET SOME DATA ========

print()
print(">>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
print(">>>>>>>> COMMENCING DOWNLOAD <<<<<<<<")
print(">>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
print()

downloadArticlesFromList(
    inlist=LOCATION_OF_PAPER_URLS_LIST,
    exarticles=LOCATION_OF_EXISTING_XML_PAPERS,
    outfpath=LOCATION_TO_STORE_XML_PAPERS,
    apikey=API_KEY
)

print(">>>> Download Complete <<<<")

print()
print(">>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
print(">>>>>>>> COMMENCING PARSING <<<<<<<<")
print(">>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
print()

parseXMLAll(
    dirpath=LOCATION_TO_STORE_XML_PAPERS,
    write_loc=LOCATION_TO_STORE_PARSED_PAPERS
)

print(">>>> Parsing Complete <<<<")

print()
print("==== DATA IS READY FOR USE ====")
print()

# ===============================