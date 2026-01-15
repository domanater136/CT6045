#!/usr/bin/env python2
import sys

current_word = None
current_score = 0

for line in sys.stdin:
    word, score = line.strip().split('\t')
    score = int(score)

    if word == current_word:
        current_score += score
    else:
        if current_word:
            print "{0}\t{1}".format(current_word, current_score)
        current_word = word
        current_score = score
