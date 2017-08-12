Utility data is data unrelated to the actual papers to be summarised, but is useful in processing them. These include things like stopword data. Current utility data files include:

- common_words.txt - stop words
- permitted_titles.txt - section titles which are allowed to be separated into their own section in the papers data. Many papers include section titles that are very specific to that paper only, whereas we only want general titles included such as "Introduction", "Method" and "Conclusion".
- definite_non_summary_titles.txt - section titles which hardly ever contain summary statements, according to the data
 that was collected about where information came from within the papers.