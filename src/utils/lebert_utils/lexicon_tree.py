# -*- coding: utf-8 -*-

import collections


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self, use_single=True):
        self.root = TrieNode()
        self.max_depth = 0
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        current = self.root
        deep = 0
        for letter in word:
            current = current.children[letter]
            deep += 1
        current.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, str, space=""):
        matched = []
        while len(str) > self.min_len:
            if self.search(str):
                matched.insert(0, space.join(str[:]))  # 短的词总是在最前面
            del str[-1]

        if len(matched) > 1 and len(matched[0]) == 1:  # filter single character word
            matched = matched[1:]

        return matched
