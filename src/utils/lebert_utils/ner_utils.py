# coding:utf-8


import os
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertConfig, AdamW

from src.model.LEBERT_model import LEBertCRFForNer
from src.utils.bert_utils import Lookahead, WarmupLinearSchedule
from src.utils.lebert_utils.metrics import seq_f1_with_mask, SeqEntityScore
from src.utils.lebert_utils.preprocess import sent_to_matched_words_boundaries


def build_bert_inputs(args, inputs, sentence, label, tokenizer, lexicon_tree, label_vocab):

    token_list = sentence
    label_list = label

    assert len(token_list) == len(label_list), (len(token_list), len(label_list))

    tokens, labels = [], []
    for i, word in enumerate(token_list):

        if word == ' ' or word == '':
            word = '-'

        token = tokenizer.tokenize(word)

        if len(token) > 1:
            token = [tokenizer.unk_token]

        tokens.extend(token)
        labels.append(label_list[i])

    assert len(tokens) == len(labels), (len(tokens), len(labels))

    inputs_dict = tokenizer.encode_plus(tokens, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    label_ids = []
    label_ids.extend(label_vocab.convert_items_to_ids(['O']))
    label_ids.extend(label_vocab.convert_items_to_ids(labels))
    label_ids.extend(label_vocab.convert_items_to_ids(['O']))

    assert len(input_ids) == len(label_ids), (len(input_ids), len(label_ids))

    sentence = ['[CLS]'] + sentence + ['[SEP]']

    matched_words, sent_boundaries = sent_to_matched_words_boundaries(sentence, lexicon_tree, args.max_word_num)

    max_len = len(input_ids)
    if len(sent_boundaries) < max_len:
        boundary_ids = sent_boundaries + [tokenizer.pad_token_id] * (max_len - len(sent_boundaries))
    else:
        boundary_ids = sent_boundaries[:max_len]

    assert len(input_ids) == len(matched_words), (len(input_ids), len(matched_words))
    assert len(input_ids) == len(boundary_ids), (len(input_ids), len(boundary_ids))

    inputs['input_ids'].append(input_ids)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)
    inputs['labels'].append(label_ids)
    inputs['boundary_ids'].append(boundary_ids)
    inputs['matched_words'].append(matched_words)
    inputs['input_length'].append(len(input_ids))


class NerDataset(Dataset):
    def __init__(self, data_dict):
        super(NerDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['labels'][index],
            self.data_dict['boundary_ids'][index],
            self.data_dict['matched_words'][index],
            self.data_dict['input_length'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len, max_word_num, tokenizer, word_vocab):
        self.max_seq_len = max_seq_len
        self.max_word_num = max_word_num
        self.tokenizer = tokenizer
        self.word_vocab = word_vocab

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list, labels_list,
                         boundary_ids_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.zeros_like(input_ids)
        boundary_ids = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):

            seq_len = len(input_ids_list[i])

            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
                labels[i, :seq_len] = torch.tensor(labels_list[i], dtype=torch.long)
                boundary_ids[i, :seq_len] = torch.tensor(boundary_ids_list[i], dtype=torch.long)

            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
                labels[i] = torch.tensor(labels_list[i][:max_seq_len], dtype=torch.long)
                boundary_ids[i] = torch.tensor(boundary_ids_list[i][:max_seq_len], dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, labels, boundary_ids

    def pad_and_truncate_matched_words(self, matched_words_list, max_seq_len):

        matched_words_list = list(matched_words_list)

        matched_word_ids = torch.zeros((len(matched_words_list)), max_seq_len, self.max_word_num, dtype=torch.long)
        matched_word_mask = torch.zeros_like(matched_word_ids)

        for i in range(len(matched_words_list)):
            seq_len = len(matched_words_list[i])
            if seq_len <= max_seq_len:
                for k in range(max_seq_len - seq_len):
                    matched_words_list[i].append([])
            else:
                tmp = []
                for k in range(max_seq_len):
                    tmp.append(matched_words_list[i][k])
                matched_words_list[i] = tmp

        for i, matched_words in enumerate(matched_words_list):
            for j, words in enumerate(matched_words):
                words_ids = self.word_vocab.convert_items_to_ids(words)
                words_ids = words_ids if len(words_ids) != 0 else [0]
                matched_word_ids[i, j, :len(words_ids)] = torch.tensor(words_ids, dtype=torch.long)
                matched_word_mask[i, j, :len(words_ids)] = 1

        return matched_word_ids, matched_word_mask

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list, boundary_ids_list, matched_words_list, \
        input_length_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels, boundary_ids = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list, labels_list,
                                  boundary_ids_list,  max_seq_len)

        matched_word_ids, matched_word_mask = self.pad_and_truncate_matched_words(matched_words_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'boundary_ids': boundary_ids,
            'matched_word_ids': matched_word_ids,
            'matched_word_mask': matched_word_mask,
            'input_length': input_length_list
        }

        return data_dict


def load_data(args, tokenizer,  word_vocab):
    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    with open(train_cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(dev_cache_pkl_path, 'rb') as f:
        dev_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, args.max_word_num, tokenizer, word_vocab)

    train_dataset = NerDataset(train_data)
    dev_dataset = NerDataset(dev_data)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, dev_dataloader


def build_optimizer(args, model, train_steps):

    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    classifier_param_optimizer = []
    other_param_optimizer = []

    for name, param in model_param:
        space = name.split('.')
        if space[0] == 'bert':
            bert_param_optimizer.append((name, param))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, param))
        elif space[0] == 'classifier':
            classifier_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [

        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate},

        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},

        {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.classifier_learning_rate},
        {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.classifier_learning_rate},

        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.use_lookahead:
        optimizer = Lookahead(optimizer, 5, 1)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


def build_model_and_tokenizer(args, num_labels, pretrained_word_embedding):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.model_name_or_path)
    model = LEBertCRFForNer.from_pretrained(args.model_name_or_path, config=config,
                                            pretrained_embeddings=pretrained_word_embedding,
                                            num_labels=num_labels)
    model.to(args.device)

    return tokenizer, model


def batch2cuda(args, batch):
    input_ids, token_type_ids, attention_mask, labels, matched_word_ids, matched_word_mask, boundary_ids = \
        batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels'], \
        batch['matched_word_ids'], batch['matched_word_mask'], batch['boundary_ids']
    input_ids, token_type_ids, attention_mask, labels, matched_word_ids, matched_word_mask, boundary_ids = \
        input_ids.to(args.device), token_type_ids.to(args.device), attention_mask.to(args.device), \
        labels.to(args.device), matched_word_ids.to(args.device), matched_word_mask.to(args.device), \
        boundary_ids.to(args.device)

    batch_cuda = {}
    batch_cuda['input_ids'], batch_cuda['token_type_ids'], batch_cuda['attention_mask'], batch_cuda['labels'], \
    batch_cuda['matched_word_ids'], batch_cuda['matched_word_mask'], batch_cuda['boundary_ids'] = \
        input_ids, token_type_ids, attention_mask, labels, matched_word_ids, matched_word_mask, boundary_ids

    return batch_cuda


# ====================== evaluation ============================
def evaluate(args, model, dev_dataloader, label_vocab, entity_level=True):

    val_iterator = tqdm(dev_dataloader, desc='Evaluation', total=len(dev_dataloader))
    val_loss = 0.

    if entity_level:
        eval_metric = SeqEntityScore()

        entity_all_label_ids = []
        entity_all_predict_ids = []

        with torch.no_grad():
            for batch in val_iterator:
                batch_cuda = batch2cuda(args, batch)
                loss, predict = model(**batch_cuda)
                val_loss += loss.item()

                label_ids = batch_cuda['labels'].detach().cpu().numpy().tolist()
                predict_ids = predict.detach().cpu().numpy().tolist()
                input_length = batch['input_length']

                for i in range(len(label_ids)):
                    tmp_label, tmp_predict = [], []
                    for j in range(1, input_length[i] - 1):  # skip [CLS] and [SEP]
                        tmp_label.append(label_vocab.convert_id_to_item(label_ids[i][j]))
                        tmp_predict.append(label_vocab.convert_id_to_item(predict_ids[i][j]))
                    entity_all_label_ids.append(tmp_label)
                    entity_all_predict_ids.append(tmp_predict)

        eval_metric.update(entity_all_label_ids, entity_all_predict_ids)

        entity_metrics, entity_info = eval_metric.result()
        precision, recall, f1 = entity_metrics['precision'], entity_metrics['recall'], entity_metrics['f1']

    else:
        char_all_input_ids = []
        char_all_label_ids = []
        char_all_predict_ids = []
        char_all_attention_mask = []

        with torch.no_grad():
            for batch in val_iterator:
                batch_cuda = batch2cuda(args, batch)
                loss, predict = model(**batch_cuda)
                val_loss += loss.item()

                input_ids = batch_cuda['input_ids'].detach().cpu().numpy().tolist()
                label_ids = batch_cuda['labels'].detach().cpu().numpy().tolist()
                attention_mask = batch_cuda['attention_mask'].detach().cpu().numpy().tolist()
                predict_ids = predict.detach().cpu().numpy().tolist()

                for i in range(len(input_ids)):
                    char_all_input_ids.append(input_ids[i])
                    char_all_label_ids.append(label_ids[i])
                    char_all_predict_ids.append(predict_ids[i])
                    char_all_attention_mask.append(attention_mask[i])

        acc, precision, recall, f1, all_true_labels, all_predict_labels = seq_f1_with_mask(
            char_all_label_ids, char_all_predict_ids, char_all_attention_mask, label_vocab)

        entity_info = None

    avg_dev_loss = val_loss / len(dev_dataloader)

    metrics = {}
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1

    metrics['avg_dev_loss'] = avg_dev_loss

    return metrics, entity_info
