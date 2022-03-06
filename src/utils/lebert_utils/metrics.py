# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score


# 字符级验证方式
def seq_f1_with_mask(all_true_labels, all_pred_labels, all_label_mask, label_vocab):

    assert len(all_true_labels) == len(all_pred_labels), (len(all_true_labels), len(all_pred_labels))
    assert len(all_true_labels) == len(all_label_mask), (len(all_true_labels), len(all_label_mask))

    true_labels = []
    pred_labels = []

    sample_num = len(all_true_labels)
    for i in range(sample_num):
        tmp_true = []
        tmp_predict = []

        assert len(all_true_labels[i]) == len(all_pred_labels[i]), (len(all_true_labels[i]), len(all_pred_labels[i]))
        assert len(all_true_labels[i]) == len(all_label_mask[i]), (len(all_true_labels[i]), len(all_label_mask[i]))

        real_seq_length = np.sum(all_label_mask[i])
        for j in range(1, real_seq_length - 1):  # skip [CLS] and [SEP]
            if all_label_mask[i][j] == 1:
                tmp_true.append(label_vocab.convert_id_to_item(all_true_labels[i][j]).replace("M-", "I-"))
                tmp_predict.append(label_vocab.convert_id_to_item(all_pred_labels[i][j]).replace("M-", "I-"))

        true_labels.append(tmp_true)
        pred_labels.append(tmp_predict)

    acc = accuracy_score(true_labels, pred_labels)
    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    acc, p, r, f1 = round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)

    return acc, p, r, f1, true_labels, pred_labels


# 实体级评测方式
def get_entity_bio(seq):
    chunks = []
    chunk = [-1, -1, -1]
    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = i
            chunk[0] = tag.split('-')[1]
            chunk[2] = i
            if i == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = i
            if i == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq):
    return get_entity_bio(seq)


class SeqEntityScore(object):
    def __init__(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path)
            pre_entities = get_entities(pre_path)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


