import sys
import os
import collections
import math

class PMICalculator:

    def __init__(self):
        self.label_counts = None
        self.word_counts = None
        self.joint_counts = None
        self.num_pairs = None

    def train(self, corpus, smoothing_factor=0.0):
        self.label_counts = collections.defaultdict(lambda: smoothing_factor)
        self.word_counts = collections.defaultdict(lambda: smoothing_factor)
        self.joint_counts = collections.defaultdict(lambda: collections.defaultdict(lambda: smoothing_factor))
        self.num_pairs = 0

        for label, document in corpus:
            for word in document:
                weight = 1.0
                self.label_counts[label] += 1
                self.word_counts[word] += 1
                self.joint_counts[label][word] += 1
                self.num_pairs += 1

    def key_set(self, label):
        return self.joint_counts[label].keys()

    def pmi(self, label, word):
        joint_prob = float(self.joint_counts[label][word]) / float(self.num_pairs)
        label_prob = float(self.label_counts[label]) / float(self.num_pairs)
        word_prob = float(self.word_counts[word]) / float(self.num_pairs)
        return joint_prob / (label_prob * word_prob)

    def count(self, word):
        return self.word_counts[word]
