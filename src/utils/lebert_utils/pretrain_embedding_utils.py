# -*- coding: utf-8 -*-


import os
import json
import numpy as np
from tqdm import tqdm, trange
import torch
import pickle


def load_pretrain_embed(embedding_path, max_scan_num=1000000, add_seg_vocab=False):
    if add_seg_vocab:
        max_scan_num = -1

    embed_dict = dict()
    embed_dim = -1
    with open(embedding_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if max_scan_num == -1:
            max_scan_num = len(lines)
        max_scan_num = min(max_scan_num, len(lines))
        line_iter = trange(max_scan_num)
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            items = line.split()
            if len(items) == 2:
                embed_dim = int(items[1])
                continue
            elif len(items) == 201:
                embed_dim = 200
                token = items[0]
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[1:]
                embed_dict[token] = embedd
            elif len(items) > 201:
                print("++++longer than 201+++++, line is: %s\n" % (line))
                token = items[0:-200]
                token = "".join(token)
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[-200:]
                embed_dict[token] = embedd
            else:
                print("-------error word-------, line is: %s\n" % (line))

    return embed_dict, embed_dim


def build_pretrained_embedding_for_corpus(
        embedding_path,
        word_vocab,
        embed_dim=200,
        max_scan_num=1000000,
        saved_corpus_embedding_dir=None,
        add_seg_vocab=False
):
    saved_corpus_embedding_file = os.path.join(saved_corpus_embedding_dir,
                                               'saved_word_embedding_{}.pkl'.format(max_scan_num))

    if os.path.exists(saved_corpus_embedding_file):
        with open(saved_corpus_embedding_file, 'rb') as f:
            pretrained_emb = pickle.load(f)
        return pretrained_emb, embed_dim

    embed_dict = dict()
    if embedding_path is not None:
        embed_dict, embed_dim = load_pretrain_embed(embedding_path, max_scan_num=max_scan_num,
                                                    add_seg_vocab=add_seg_vocab)

    scale = np.sqrt(3.0 / embed_dim)
    pretrained_emb = np.empty([word_vocab.item_size, embed_dim])

    matched = 0
    not_matched = 0

    for idx, word in enumerate(word_vocab.idx2item):
        if word in embed_dict:
            pretrained_emb[idx, :] = embed_dict[word]
            matched += 1
        else:
            pretrained_emb[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_matched += 1

    pretrained_size = len(embed_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, matched, not_matched, (not_matched + 0.) / word_vocab.item_size))

    with open(saved_corpus_embedding_file, 'wb') as f:
        pickle.dump(pretrained_emb, f, protocol=4)

    return pretrained_emb, embed_dim
