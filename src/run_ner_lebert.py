# -*- coding: utf-8 -*-

import gc
import time
import warnings
from argparse import ArgumentParser
from torch.cuda.amp import autocast

from src.utils.bert_utils import *
from src.utils.utils import save_pickle
from src.utils.lebert_utils.ner_utils import *
from src.utils.lebert_utils.vocab import ItemVocabArray, ItemVocabFile
from src.utils.lebert_utils.pretrain_embedding_utils import build_pretrained_embedding_for_corpus
from src.utils.lebert_utils.preprocess import build_lexicon_tree_from_vocabs, \
    get_corpus_matched_word_from_lexicon_tree


def read_data(args, tokenizer, lexicon_tree, label_vocab):
    train_inputs, dev_inputs = defaultdict(list), defaultdict(list)

    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            text = words.split('\002')
            label = labels.split('\002')
            build_bert_inputs(args, train_inputs, text, label, tokenizer, lexicon_tree, label_vocab)

    with open(args.dev_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            words, labels = line.strip('\n').split('\t')
            text = words.split('\002')
            label = labels.split('\002')
            build_bert_inputs(args, dev_inputs, text, label, tokenizer, lexicon_tree, label_vocab)

    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    save_pickle(train_inputs, train_cache_pkl_path)
    save_pickle(dev_inputs, dev_cache_pkl_path)


def build_lebert_files(args):

    # 1.prepare data
    # 1.1. lexicon tree
    lexicon_tree = build_lexicon_tree_from_vocabs([args.word_vocab_file], scan_nums=[args.max_scan_num])
    embed_lexicon_tree = lexicon_tree

    # 1.2. word vocab, label vocab
    data_files = [args.train_path, args.dev_path]
    matched_words = get_corpus_matched_word_from_lexicon_tree(data_files, embed_lexicon_tree)

    word_vocab, label_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5), \
                              ItemVocabFile(files=[args.label_file], is_word=False)

    # 1.3. prepare embedding
    pretrained_word_embedding, embed_dim = build_pretrained_embedding_for_corpus(
        embedding_path=args.word_embedding,
        word_vocab=word_vocab,
        embed_dim=args.word_embed_dim,
        max_scan_num=args.max_scan_num,
        saved_corpus_embedding_dir=args.saved_embedding_dir
    )

    return lexicon_tree, pretrained_word_embedding, label_vocab, word_vocab


def train(args):

    lexicon_tree, pretrained_word_embedding, label_vocab, word_vocab = build_lebert_files(args)

    num_labels = label_vocab.get_item_size()

    tokenizer, model = build_model_and_tokenizer(args, num_labels, pretrained_word_embedding)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer, lexicon_tree, label_vocab)

    train_dataloader, dev_dataloader = load_data(args, tokenizer, word_vocab)

    total_steps = args.num_epochs * len(train_dataloader)
    optimizer, scheduler = build_optimizer(args, model, total_steps)

    scaler = None
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    global_steps, total_loss, cur_avg_loss, best_f1 = 0, 0., 0., 0.

    print("\n >> Start training ... ... ")
    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:

            model.zero_grad()

            batch_cuda = batch2cuda(args, batch)

            if args.use_fp16:
                with autocast():
                    loss = model(**batch_cuda)[0]
            else:
                loss = model(**batch_cuda)[0]

            if args.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if args.use_fgm:
                model.zero_grad()
                fgm = FGM(args, model)
                fgm.attack()

                if args.use_fp16:
                    with autocast():
                        adv_loss = model(**batch_cuda)[0]
                else:
                    adv_loss = model(**batch_cuda)[0]

                if args.use_fp16:
                    scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()

                fgm.restore()

            if args.use_pgd:
                model.zero_grad()
                pgd = PGD(args, model)
                pgd.backup_grad()
                for t in range(args.adv_k):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()

                    if args.use_fp16:
                        with autocast():
                            adv_loss = model(**batch_cuda)[0]
                    else:
                        adv_loss = model(**batch_cuda)[0]

                    if args.use_fp16:
                        scaler.scale(adv_loss).backward()
                    else:
                        adv_loss.backward()

                pgd.restore()

            if args.use_freelb:
                freelb = FreeLB(args, model, optimizer, batch_cuda, scaler)
                freelb.attack()

            if args.use_fp16:
                scaler.unscale_(optimizer)

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            if args.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            if args.use_ema:
                if args.ema_start:
                    ema.update()

            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_steps == 0:

                epoch_avg_loss = cur_avg_loss / args.logging_steps
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                if args.use_ema:
                    if global_steps >= args.ema_start_step and not args.ema_start:
                        print('\n>>> EMA starting ...')
                        args.ema_start = True
                        ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)

                if args.do_eval:

                    model.eval()

                    if args.use_ema:
                        if args.ema_start:
                            ema.apply_shadow()

                    print("\n >> Start evaluating ... ... ")

                    metric, entity_info = evaluate(args, model, dev_dataloader, label_vocab)

                    f1_score, precision, recall = metric['f1'], metric['precision'], metric['recall']
                    dev_loss = metric['avg_dev_loss']

                    f1_score, precision, recall, dev_loss = round(f1_score, 4), round(precision, 4), \
                                                            round(recall, 4), round(dev_loss, 4)

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        save_model(args, model, tokenizer)

                        print(f"\n >> best model saved ."
                              f"\n >> f1 : {f1_score}, precision : {precision}, recall : {recall}, "
                              f"dev loss : {dev_loss} .")

                        if args.print_entity_info:
                            print("***** Entity results %s *****")
                            for key in sorted(entity_info.keys()):
                                print(f"******* entity:{key} results ********")
                                info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
                                print(info)

                    if args.use_ema:
                        if args.ema_start:
                            ema.restore()

                    model.train()
                    cur_avg_loss = 0.

            global_steps += 1
            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

    if args.use_ema:
        ema.apply_shadow()

    if not args.do_eval:
        save_model(args, model, tokenizer)

    data = time.asctime(time.localtime(time.time())).split(' ')
    now_time = data[-1] + '-' + data[-5] + '-' + data[-3] + '-' + \
               data[-2].split(':')[0] + '-' + data[-2].split(':')[1] + '-' + data[-2].split(':')[2]
    os.makedirs(os.path.join(args.output_path, f'f1-{best_f1}-{now_time}'), exist_ok=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    print('\n >> Finish training .')


def main(ner_type):
    parser = ArgumentParser()

    parser.add_argument('--ner_type', type=str, default=ner_type)

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--output_path', type=str,
                        default=f'../user_data/output_model/{ner_type}')
    parser.add_argument('--train_path', type=str,
                        default=f'../raw_data/train.json')
    parser.add_argument('--dev_path', type=str,
                        default=f'../raw_data/dev.json')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'../user_data/process_data/pkl/{ner_type}')
    parser.add_argument('--model_name_or_path', type=str,
                        default=f'../user_data/pretrain_model/bert-base-chinese')
    parser.add_argument("--label_file", type=str,
                        default="../raw_data/label.txt")

    # use tencent word vectors
    parser.add_argument("--word_vocab_file", type=str,
                        default="../user_data/w2v_model/tencent_word_vocab.txt")
    parser.add_argument("--word_embedding", type=str,
                        default="../user_data/w2v_model/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt")

    parser.add_argument("--saved_embedding_dir", type=str,
                        default="../user_data/w2v_model")

    parser.add_argument('--do_lower_case', type=bool, default=True)

    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--print_entity_info', type=bool, default=False)

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=91)

    parser.add_argument("--word_embed_dim", type=int, default=200)
    parser.add_argument("--max_scan_num", type=int, default=100000)
    parser.add_argument("--max_word_num", type=int, default=5)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--crf_learning_rate', type=float, default=3e-2)
    parser.add_argument('--classifier_learning_rate', type=float, default=3e-4)
    parser.add_argument('--other_learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--use_fgm', type=bool, default=False)
    parser.add_argument('--use_pgd', type=bool, default=False)
    parser.add_argument('--use_freelb', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--use_lookahead', type=bool, default=False)
    parser.add_argument('--use_fp16', type=bool, default=False)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')

    parser.add_argument('--adv-lr', type=float, default=3e-5)
    parser.add_argument('--adv-steps', type=int, default=3, help="should be at least 1")
    parser.add_argument('--adv-init-mag', type=float, default=0)
    parser.add_argument('--norm-type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv-max-norm', type=float, default=0, help="set to 0 to be unlimited")

    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_step', type=int, default=0)

    parser.add_argument('--logging_steps', type=int, default=50)

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    path_list = [args.output_path, args.data_cache_path]
    for i in path_list:
        os.makedirs(i, exist_ok=True)

    seed_everything(args.seed)

    train(args)


if __name__ == '__main__':
    main('lebert')
