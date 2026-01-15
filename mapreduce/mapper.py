#!/usr/bin/env python2
import sys
import csv
import re

# read CSV
reader = csv.reader(sys.stdin, quotechar='"', delimiter=',', escapechar='\\')
for row in reader:
    polarity, text = row[0], row[1]

    # choose score: positive=+1, negative=-1
    if polarity == "2":
        score = 1
    else:
        score = -1

    # get single words, keep stuff like "-" and "'", but split elsewhere.
    words = re.findall(r"\b\w+(?:['-]\w+)*\b", text.lower())

    for word in words:
        print "{0}\t{1}".format(word, score)
