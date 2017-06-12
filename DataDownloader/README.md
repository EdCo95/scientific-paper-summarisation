# DataDownloader

**N.B. All code in this folder runs with Python 3. The rest of the project runs on Python 2. Please ensure you run the file with `python3 acquire_data.py`, NOT `python acquire_data.py`. Requirements for Python 3 are in `python3_requirements.txt`, which you should install with `pip3 install -r python3_requirements.txt` if you need to.**

All code required to download the data from ScienceDirect is contained here. The only things you need to do to make this run are to change the `sys.path.insert(0, PROJECT_PARENT_DIR)` at the top of each of the code files in this folder to match where this code is stored on your system. You then only need concern yourself with `acquire_data.py`. In this file, there are several config variables which you should be sure to set correctly, they are described in the file and the parts you should change are clearly indicated.

`cspubsum_ids.txt` and `cspubsum_test_ids.txt` contain lists of file URLs from which each of the papers used in the original research can be accessed. These URLs are processed automatically by the code here. If you wish to add more papers here for download, you should add URLs to these lists in the same format.

In order to download the data from ScienceDirect, you will need to be connected to a network from an organisation which is subscribed to ScienceDirect (this is most universities). You will also need to obtain an API key from this location: https://dev.elsevier.com/user/login and set it in `acquire_data.py` to access the data.

Once you run the code, it will automatically download all of the papers in the list in XML format and store them in the directory you specify. It will then parse all of these documents into `.txt` format - where each of the section headings in each paper is separated by the symbol "@&#" - a symbol never seen in the paper text. This makes it easy to read in the papers - simply read them as a string then split the string on "@&#". The resulting list can be parsed into a dictionary where the keys are section headings and values are the text of that section. All of this reading in of papers is handled by the function `read_in_paper()` in `useful_functions.py` in the "DataTools/" directory.

If you elect to download all 10148 papers, then they will take up 3.38 GB of space. The parsed papers in `.txt` format will require 438.8 MB of space.