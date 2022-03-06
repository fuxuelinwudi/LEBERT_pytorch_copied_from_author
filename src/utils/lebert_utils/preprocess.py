# -*- coding: utf-8 -*-

import json
from tqdm import trange
from src.utils.lebert_utils.lexicon_tree import Trie


def sent_to_matched_words_boundaries(sent, lexicon_tree, max_word_num=None):
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    sent_boundaries = [[] for _ in range(sent_length)]  # each char has a boundary

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0 and len(sent_boundaries[idx]) == 0:
            sent_boundaries[idx].append(3)  # S-
        else:
            if len(words) == 1 and len(words[0]) == 1:  # single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
                    sent_boundaries[idx].append(3)  # S-
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    if 0 not in sent_boundaries[idx]:
                        sent_boundaries[idx].append(0)  # S-
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        if 1 not in sent_boundaries[tmp_j]:
                            sent_boundaries[tmp_j].append(1)  # M-
                        sent_words[tmp_j].append(word)
                    if 2 not in sent_boundaries[end_pos]:
                        sent_boundaries[end_pos].append(2)  # E-
                    sent_words[end_pos].append(word)

    assert len(sent_words) == len(sent_boundaries)

    new_sent_boundaries = []
    idx = 0
    for boundary in sent_boundaries:
        if len(boundary) == 0:
            print("Error")
            new_sent_boundaries.append(0)
        elif len(boundary) == 1:
            new_sent_boundaries.append(boundary[0])
        elif len(boundary) == 2:
            total_num = sum(boundary)
            new_sent_boundaries.append(3 + total_num)
        elif len(boundary) == 3:
            new_sent_boundaries.append(7)
        else:
            print(boundary)
            print("Error")
            new_sent_boundaries.append(8)
    assert len(sent_words) == len(new_sent_boundaries)

    return sent_words, new_sent_boundaries


def sent_to_distinct_matched_words(sent, lexicon_tree):
    sent_length = len(sent)
    sent_words = [[[], [], [], []] for _ in range(sent_length)]  # 每个字都有对应BMES
    sent_group_mask = [[0, 0, 0, 0] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]
        words = lexicon_tree.enumerateMatch(sub_sent)
        if len(words) == 0:
            continue
        else:
            for word in words:
                word_length = len(word)
                if word_length == 1:
                    sent_words[idx][3].append(word)
                    sent_group_mask[idx][3] = 1
                else:
                    sent_words[idx][0].append(word)  # begin
                    sent_group_mask[idx][0] = 1
                    for pos in range(1, word_length - 1):
                        sent_words[idx + pos][1].append(word)  # middle
                    sent_words[idx + word_length - 1][2].append(word)  # end
        if len(sent_words[idx][1]) > 0:
            sent_group_mask[idx][1] = 1
        if len(sent_words[idx][2]) > 0:
            sent_group_mask[idx][2] = 1

    return sent_words, sent_group_mask


def sent_to_matched_words(sent, lexicon_tree, max_word_num=None):
    """same to sent_to_matched_words_boundaries, but only return words"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0:
            continue
        else:
            if len(words) == 1 and len(words[0]) == 1:  # single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        sent_words[tmp_j].append(word)
                    sent_words[end_pos].append(word)

    return sent_words


def sent_to_matched_words_set(sent, lexicon_tree, max_word_num=None):
    """return matched words set"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    matched_words_set = set()
    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        _ = [matched_words_set.add(word) for word in words]
    matched_words_set = list(matched_words_set)
    matched_words_set = sorted(matched_words_set)
    return matched_words_set


def get_corpus_matched_word_from_vocab_files(files, vocab_files, scan_nums=None):
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    total_matched_words = get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree)
    return total_matched_words, lexicon_tree


def get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree):
    total_matched_words = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                words, labels = line.strip('\n').split('\t')
                text = words.split('\002')
                sent = [ch for ch in text]
                sent_matched_words = sent_to_matched_words_set(sent, lexicon_tree)
                _ = [total_matched_words.add(word) for word in sent_matched_words]

    total_matched_words = list(total_matched_words)
    total_matched_words = sorted(total_matched_words)

    return total_matched_words


def insert_seg_vocab_to_lexicon_tree(seg_vocab, word_vocab, lexicon_tree):
    seg_words = set()
    whole_words = set()
    with open(seg_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)

        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            if line:
                seg_words.add(line)

    with open(word_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)

        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            if line:
                whole_words.add(line)

    overleap_words = seg_words & whole_words
    overleap_words = list(overleap_words)
    overleap_words = sorted(overleap_words)
    print("Overleap words number is: \n", len(overleap_words))

    for word in overleap_words:
        lexicon_tree.insert(word)

    return lexicon_tree


def build_lexicon_tree_from_vocabs(vocab_files, scan_nums=None):
    # 1.获取词汇表
    print(vocab_files)
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    return lexicon_tree


def get_all_labels_from_corpus(files, label_file, defalut_label='O'):
    labels = [defalut_label]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    label = sample['label']
                    if isinstance(label, list):
                        for l in label:
                            if l not in labels:
                                labels.append(l)
                    else:
                        labels.append(label)

    with open(label_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write("%s\n" % (label))

