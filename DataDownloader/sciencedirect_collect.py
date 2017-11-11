import sys
sys.path.insert(0, "/Users/edcollins/Documents/CS/4thYearProject/Code")
import os
import urllib
import urllib.request
#import feedparser
from bs4 import BeautifulSoup, SoupStrainer
import re
import os
import shutil
import argparse
import io
import time
import numpy as np

# Guide to Elsevier article retrieval: http://dev.elsevier.com/tecdoc_text_mining.html
# Step 0: obtain API key: https://dev.elsevier.com/user/login
# Step 1: explore ScienceDirect journals by subject: http://www.sciencedirect.com/science/journals/ (The developer search API doesn't seem to have the funcionality of retrieving journals by subject)
# Step 2: retrieve XML journal sitemap: http://api.elsevier.com/sitemap/page/sitemap/index.html
# Step 3: collect article PIIs (identifiers)
# Step 4: downlad articles with Elsevier article retrieval API

def collectArticles(urlstr):
# get article PIIs
    retl = []
    with urllib.request.urlopen(urlstr) as url:
        response = url.read()
        linkcnt = 0
        for link in BeautifulSoup(response, parse_only=SoupStrainer("a")):
            if linkcnt == 0:
                linkcnt += 1
                continue
            if link.has_attr("href"):
                #print(link["href"])
                retl.append(link["href"])
            linkcnt += 1
    return retl



def downloadArticles(outfpath, p, apikey):

    if not os.path.exists(outfpath):
        os.makedirs(outfpath)

    flist = os.listdir(outfpath)

    if p.startswith("http:"): # full article url
        volumes = collectArticles(p)
    else:
        volumes = collectArticles("http://api.elsevier.com/sitemap/page/sitemap/" + p)
    for i, v in enumerate(volumes):
        piis = collectArticles(v)
        for piil in piis:
            pii = str(piil).replace("http://api.elsevier.com/content/article/pii/", "")
            if pii + ".xml" in flist:
                flist.remove(pii + ".xml")
                continue
            print(pii)
            piir = "http://api.elsevier.com/content/article/PII:" + pii + "?httpAccept=text/xml&APIKey=" + apikey


            local_filename, headers = urllib.request.urlretrieve(piir)
            shutil.copy(local_filename, outfpath + pii + ".xml")
        rem = len(volumes) - i
        print(rem, "volumes out of", len(volumes), "volumes remaining")


def downloadArticlesFromList(inlist, exarticles, outfpath, apikey):

    existarts = []
    for f in os.listdir(exarticles):
        if f.strip().endswith(".txt"):
            existarts.append(f.strip().replace(".txt", ""))

    inf = open(inlist, "r")
    infl = []
    for l in inf:
        if l.startswith("(http://www.sciencedirect.com"):
            l = l.replace("(http://www.sciencedirect.com/science/article/pii/", "").replace(")", "").replace("\n", "")
            if not l in infl and l not in existarts:
                infl.append(l)

    start_time = time.time()
    download_times = []
    for i, pii in enumerate(infl):

        #pii = str(art).replace("http://api.elsevier.com/content/article/pii/", "")
        #if pii + ".xml" in flist:
        #    flist.remove(pii + ".xml")
        #    continue

        if len(download_times) > 0:
            mean_download_time = np.mean(download_times)
            remaining_files = len(infl) - i
            remaining_time = mean_download_time * remaining_files

            print(pii, i+1, len(infl), "Est. Time Remaining: {} minutes".format(remaining_time/60), end="\r")
        else:
            print(pii, i+1, len(infl), end="\r")

        piir = "http://api.elsevier.com/content/article/PII:" + pii + "?httpAccept=text/xml&APIKey=" + apikey

        download_start_time = time.time()
        local_filename, headers = urllib.request.urlretrieve(piir)
        shutil.copy(local_filename, outfpath + pii + ".xml")
        download_time = time.time() - download_start_time
        if len(download_times) < 50:
            download_times.append(download_time)
        else:
            download_times.pop(0)
            download_times.append(download_time)
    print()



def getJournalURL(jname):
# get journal URL given the journal name for retrieving article PIIs
    urlstr = "http://api.elsevier.com/sitemap/page/sitemap/" + jname[0].lower() + ".html"
    retl = ""
    with urllib.request.urlopen(urlstr) as url:
        response = url.read()
        linkcnt = 0
        for link in BeautifulSoup(response, parse_only=SoupStrainer("a")):
            if linkcnt == 0:
                linkcnt += 1
                continue
            if link.has_attr("href"):
                if link.text.lower() == jname.lower():
                    #print(link["href"])
                    retl = link["href"]
                    break
            linkcnt += 1
    return retl


def downloadAllJournalArticles(skip = False, dontskip = ""):
# download journal articles from a list of journal names
    if dontskip != "":
        skip = True
    f = io.open("../compsci_journals.txt")  # all Elsevier CS journal names
    for l in f:
        if len(l) > 2:
            j = l.strip("\n")
            # skip the ones for which there are already folders, we assume for those downloading has finished
            if skip == True:
                if j.lower().replace(" ", "_") in os.listdir("../elsevier_papers_xml"):
                    if not j == dontskip:
                        print("Skipping journal:", j)
                        continue
            print("Downloading articles for journal:", j)
            jurl = getJournalURL(j)
            downloadArticles("../elsevier_papers_xml/" + j.lower().replace(" ", "_") + "/", jurl)


if __name__ == '__main__':

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("skip", type=bool)
    parser.add_argument("skipexcept", type=str)
    args = parser.parse_args()

    if args.skip:
        skip = args.skip
    if args.skipexcept:
        dontskip = args.skipexcept

    #downloadAllJournalArticles(skip=True, dontskip="AEU - International Journal of Electronics and Communications")
    downloadAllJournalArticles(skip=skip, dontskip=dontskip)

    '''

    # The base directory of the project, from the root directory
    BASE_DIR = "/Users/edcollins/Documents/CS/4thYearProject/Code/Dev"

    #downloadArticlesFromList("/Users/Isabelle/Documents/UCLMR/articles_todownload_oa_cs2.txt", "/Users/Isabelle/Documents/UCLMR/ccby-articles_text", "/Users/Isabelle/Documents/UCLMR/oa-articles-cs_xml/")

    downloadArticlesFromList(BASE_DIR + "/DataDownloader/cspubsum_test_ids.txt", BASE_DIR + "/DataDownloader/Test2/",
                             BASE_DIR + "/DataDownloader/Test2/")


