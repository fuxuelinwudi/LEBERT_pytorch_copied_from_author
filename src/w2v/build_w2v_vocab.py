# coding:utf-8

from gensim.models import Word2Vec
from argparse import ArgumentParser


def build_w2v_vocab_tencent(args):

    all_words = []

    with open(args.tencent_model_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            data = line.strip().split()
            word, embedding = data[0], data[1:]
            all_words.append(word)

    with open(args.out_tencent_word_vocab_path, 'w', encoding='utf-8') as f:
        for word in all_words:
            f.writelines(word + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--tencent_model_path', type=str,
                        default='../../user_data/w2v_model/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt')
    parser.add_argument('--out_tencent_word_vocab_path', type=str,
                        default='../../user_data/w2v_model/tencent_word_vocab.txt')

    args = parser.parse_args()
    build_w2v_vocab_tencent(args)

