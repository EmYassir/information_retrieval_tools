'''
Document ranking prediction using generator
Save the ranked documents in order into a new data file with suffix '_rank.json'.
rank_and_save(args, dev_filename_doc, dev_filename_qa, best_model, dev_filename_doc_ranked)
dev_filename_doc: documents
dev_filename_qa: answers
best_model: model
dev_filename_doc_ranked: output of new documents in ranked order
'''
import torch
from eval import accranker
from utils import data as dtld
from utils import data
import logging
import argparse
from model.genranker import GEN
from functools import partial
from utils import vector
import json
from tqdm import tqdm
import numpy as np
import re
from utils import data_tok
import string
import io
import socket

logger = logging.getLogger()


def set_args():
    parser = argparse.ArgumentParser(
        'Adversarial Question Answering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    args.dataset = 'quasart'
    args.FIX_EMB = True
    args.random_seed = 1
    args.D_WEIGHT_DECAY = 0.05
    args.G_WEIGHT_DECAY = 0.01
    args.D_LEARNING_RATE = 0.01
    args.G_LEARNING_RATE = 0.001
    args.D_MOMENTUM = 0.2
    args.G_MOMENTUM = 0.2
    args.DOC_TEMPERATURE = 0.9
    args.SPN_TEMPERATURE = 0.9
    args.BATCH_SIZE = 24
    args.restrict_vocab = True
    args.type_max = True
    args.D_GRAD_CLIP = 0.5
    args.G_GRAD_CLIP = 0.5
    args.updates = 0
    args.num_sample_doc = 5
    args.num_sample_spn = 3

    args.dis_epochs = 1
    args.gen_epochs = 3
    args.num_epochs = 6

    args.num_span_gen = 15
    args.num_doc_gen = 10
    args.span_len = 5
    args.sent_len = 30
    args.LAMBDA = 0.5
    args.default_num_docs = 50

    args.doc_layers = 1
    args.use_qemb = True
    args.hidden_size = 128
    args.dropout_rnn = 0.3
    args.dropout_rnn_output = True
    args.dropout_emb = 0.3
    args.concat_rnn_layers = True
    args.rnn_padding = False
    args.question_merge = 'self_attn'
    args.fix_embeddings = True
    args.data_workers = 0
    args.num_display =  200
    return args


def evl(args, dev_filename_doc, dev_filename_qa, model_file):
    stats = {'timer': dtld.Timer(), 'epoch': 0, 'best_valid': 0.0, 'best_model_file': model_file}

    # load model and dictionaries
    pre_best_acc, generator, word_dict, feature_dict = load_generator(args, model_file)
    logger.info('Previous best accuracy: %.4f', pre_best_acc)
    # prepare data loader
    num_docs = args.default_num_docs
    dev_docs, dev_questions, _ = dtld.load_data_with_doc(dev_filename_doc, num_docs)
    dev_exs_with_doc = dtld.read_data(dev_filename_qa, dev_questions)
    dev_dataset_with_doc = dtld.ReaderDataset_with_Doc(dev_exs_with_doc, dev_docs, word_dict, feature_dict, num_docs, single_answer=False)
    dev_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(dev_dataset_with_doc)
    dev_loader_with_doc = torch.utils.data.DataLoader(
        dev_dataset_with_doc,
        batch_size=args.BATCH_SIZE,
        sampler=dev_sampler_with_doc,
        num_workers=args.data_workers,
        collate_fn=partial(vector.batchify_with_docs, num_docs=num_docs),
        pin_memory=args.cuda,
    )

    # evaluate
    with torch.no_grad():
        acc_rslt = accranker.topk_eval_unofficial_with_doc(args, dev_loader_with_doc, generator,
                                                           stats, dev_exs_with_doc, dev_docs, 'dev',
                                                           topk=150)
        logger.info('current best accuracy: %.4f', acc_rslt['exact_match'])


def pred(args, dev_filename_doc, dev_filename_qa, model_file, rank_file):
    # load model and dictionaries
    pre_best_acc, generator, word_dict, feature_dict = load_generator(args, model_file)

    logger.info("Predicting ranking scores")
    # === data loader
    num_docs = args.default_num_docs
    dev_docs, dev_questions, _ = dtld.load_data_with_doc(dev_filename_doc, num_docs)
    dev_exs_with_doc = dtld.read_data(dev_filename_qa, dev_questions)
    dev_dataset_with_doc = dtld.ReaderDataset_with_Doc(dev_exs_with_doc, dev_docs, word_dict, feature_dict, num_docs,
                                                       single_answer=False)
    dev_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(dev_dataset_with_doc)
    dev_loader_with_doc = torch.utils.data.DataLoader(
        dev_dataset_with_doc,
        batch_size=args.BATCH_SIZE,
        sampler=dev_sampler_with_doc,
        num_workers=args.data_workers,
        collate_fn=partial(vector.batchify_with_docs, num_docs=num_docs),
        pin_memory=args.cuda,
    )

    # === predict
    examples = 0
    with open(rank_file, 'w+') as rf:
        for ex_with_doc in tqdm(dev_loader_with_doc, total=len(dev_loader_with_doc)):
            ex = ex_with_doc[0]
            batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
            scores_docs = generator.predict(ex_with_doc)
            for bt in scores_docs:
                rf.write(json.dumps(bt.squeeze().cpu().tolist()) + '\n')
            examples += batch_size
        rf.flush()
    logger.info('total %d examples.', examples)

def group_by_len(docfile):
    total = sum(1 for line in open(docfile, 'r'))
    grps_inds = {}
    grps_lins = {}
    with open(docfile, 'r') as df:
        print('Processing file', docfile)
        for idx_line, line in enumerate(tqdm(df, total=total)):
            lexs = len(json.loads(line))
            if lexs not in grps_inds:
                grps_inds.setdefault(lexs, []).append(idx_line)
                grps_lins.setdefault(lexs, []).append(json.loads(line))
            else:
                grps_inds[lexs].append(idx_line)
                grps_lins[lexs].append(json.loads(line))

    return grps_inds, grps_lins

def pred_by_group(args, dev_filename_doc, dev_filename_qa, model_file, rank_file):
    pre_best_acc, generator, word_dict, feature_dict = load_generator(args, model_file)

    logger.info("Predicting ranking scores")

    grps_inds, grps_lines = group_by_len(dev_filename_doc)
    with open(dev_filename_qa, 'r') as df:
        print('Processing file', dev_filename_qa)
        at_total = df.read().splitlines()

    examples = 0
    grps_ranks = {}
    for key, dt_lst in grps_lines.items():
        # ==== prepare dataloader for group key
        logger.info("start processing group with len: " + str(key)  + ' total: '+ str(len(dt_lst)))
        num_docs = key
        at_lst = [json.loads(at_total[i]) for i in grps_inds[key]]
        tmp_train_docs, tmp_train_questions, tmp_train_docs_inds = dtld.load_list_with_doc(dt_lst, num_docs, pred=True)
        # reversed_tmp_train_docs_inds = []
        # for inds in tmp_train_docs_inds:
        #     reversed_tmp_train_docs_inds.append(sorted(range(len(inds)), key=inds.__getitem__))
        # tmp_train_docs = [[old_tmp_train_docs[j][i] for i in reversed_tmp_train_docs_inds[j]] for j in range(len(old_tmp_train_docs))]
        tmp_train_exs_with_doc = dtld.read_list(at_lst, args)

        tmp_train_dataset_with_doc = dtld.ReaderDataset_with_Doc(tmp_train_exs_with_doc, tmp_train_docs, word_dict,
                                                                 feature_dict, num_docs,
                                                                 single_answer=False)
        tmp_train_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(tmp_train_dataset_with_doc)
        tmp_train_loader_with_doc = torch.utils.data.DataLoader(
            tmp_train_dataset_with_doc,
            batch_size=args.BATCH_SIZE,
            sampler=tmp_train_sampler_with_doc,
            num_workers=args.data_workers,
            collate_fn=partial(vector.batchify_with_docs, num_docs=num_docs),
            pin_memory=args.cuda,
        )
        # === predict for group key
        tmp_cnt = 0
        with torch.no_grad():
            for ex_with_doc in tqdm(tmp_train_loader_with_doc, total=len(tmp_train_loader_with_doc)):
                ex = ex_with_doc[0]
                batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
                scores_docs = generator.predict(ex_with_doc)
                for bt in scores_docs:
                    grps_ranks[grps_inds[key][tmp_cnt]] = bt.squeeze().cpu().tolist()
                    tmp_cnt += 1
                examples += batch_size

    # === write to file
    with open(rank_file, 'w+') as rf:
        json.dump(grps_ranks, rf)
        rf.flush()
    logger.info('total %d examples.', examples)

def load_generator(args, filename):
    logger.info('=> loading checkpoint %s' % filename)
    checkpoint = torch.load(filename)
    # start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    # best_f1 = checkpoint['best_f1']
    word_dict = checkpoint['word_dict']
    feature_dict = checkpoint['feature_dict']
    args.num_features = len(feature_dict)
    args.vocab_size = len(word_dict)
    generator = GEN(args).to(args.device)
    generator.load_state_dict(checkpoint['gen_state'])
    logger.info("=> checkpoint loaded, current best EM %.2f" % best_acc)
    return best_acc, generator, word_dict, feature_dict


def rank_and_save(args, dev_filename_doc, dev_filename_qa, model_file, dev_filename_doc_ranked):
    # load model and dictionaries
    stats = {'timer': dtld.Timer(), 'epoch': 0, 'best_valid': 0.0, 'best_model_file': model_file}
    pre_best_acc, generator, word_dict, feature_dict = load_generator(args, model_file)

    logger.info("Predicting ranking scores")
    # === data loader
    num_docs = args.default_num_docs
    dev_docs, dev_questions, dev_docs_inds = dtld.load_data_with_doc(dev_filename_doc, num_docs)
    dev_exs_with_doc = dtld.read_data(dev_filename_qa, dev_questions)
    dev_dataset_with_doc = dtld.ReaderDataset_with_Doc(dev_exs_with_doc, dev_docs, word_dict, feature_dict, num_docs,
                                                       single_answer=False)
    dev_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(dev_dataset_with_doc)
    dev_loader_with_doc = torch.utils.data.DataLoader(
        dev_dataset_with_doc,
        batch_size=args.BATCH_SIZE,
        sampler=dev_sampler_with_doc,
        num_workers=args.data_workers,
        collate_fn=partial(vector.batchify_with_docs, num_docs=num_docs),
        pin_memory=args.cuda,
    )
    dev_docs_ranked = []
    # dev_exs_with_doc_ranked = []
    # evaluate
    max_len = args.default_num_docs
    eval_time = data.Timer()
    exact_matchs = [data.AverageMeter() for i in range(max_len)]
    # tot_len = sum(1 for i in open(dev_filename_qa, 'r'))
    with torch.no_grad():
        examples = 0
        with io.open(dev_filename_doc_ranked, 'w+', encoding='utf-8') as df:
            # for idx, ex_with_doc in enumerate(tqdm(dev_loader_with_doc, total=len(dev_loader_with_doc))):
            for ex_with_doc in tqdm(dev_loader_with_doc):
                ex = ex_with_doc[0]
    
                batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
                scores_docs = generator.predict(ex_with_doc)
                scores_docs = scores_docs.detach()
                _, indices = scores_docs.sort(2, descending=True)
    
                for idx_q in range(batch_size):
                    # # build question answers line for txt file
                    # new_line_dev_exs_with_doc = {}
                    # new_line_dev_exs_with_doc["question"] = dev_exs_with_doc[ex_id[idx_q]]['question']
                    # new_line_dev_exs_with_doc["answers"] = dev_exs_with_doc[ex_id[idx_q]]['answer'][0]
                    # dev_exs_with_doc_ranked.append(json.dumps(new_line_dev_exs_with_doc))
    
                    predictions = []
                    new_line_docs_q_ranked = []
                    for j in range(len(indices[idx_q, 0, :])):
                        idx_doc = indices[idx_q, 0, j]
                        score_doc = scores_docs[idx_q, 0, idx_doc]
                        doc_text = dev_docs[ex_id[idx_q]][idx_doc % len(dev_docs[ex_id[idx_q]])]["document"]
                        predictions.append(" ".join(doc_text))
                        old_doc_q = dev_docs[ex_id[idx_q]][idx_doc % len(dev_docs[ex_id[idx_q]])]
                        old_que_ans = dev_exs_with_doc[ex_id[idx_q]]
                        # build document one by one
                        new_doc_q = {}
                        new_doc_q["question"] = old_doc_q["question"]
                        new_doc_q["answers"] = old_que_ans["answers_orig"]
                        new_doc_q["id"] = old_doc_q["id"]
                        new_doc_q["document"] = old_doc_q["document"]
                        new_doc_q["has_answers"] = data.has_answer(new_doc_q["answers"], new_doc_q["document"])
                        new_doc_q["score"] = score_doc.item()
                        new_line_docs_q_ranked.append(new_doc_q)
                    # dev_docs_ranked.append(json.dumps(new_line_docs_q_ranked))
                    # Write file instead of save in memory
                    df.write(json.dumps(new_line_docs_q_ranked) + '\n')
                    ground_truths = []
                    answer = dev_exs_with_doc[ex_id[idx_q]]['answers']
                    if (args.dataset == "CuratedTrec"):
                        ground_truths = answer
                    else:
                        for a in answer:
                            ground_truths.append(" ".join([w for w in a]))
                    mtsi = topk_metric_max_over_ground_truths(topk_scope_match_score, predictions, ground_truths, max_len)
                    for j in range(max_len):
                        exact_matchs[j].update(mtsi[j])
                examples += batch_size
            logger.info('Eval Results:'
                        + '\n                          E1 = %.4f | E3 = %.4f | E5 = %.4f | E10 = %.4f | E30 = %.4f |' % (
                        exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100,
                        exact_matchs[9].avg * 100, exact_matchs[29].avg * 100)
                        + '\n                          examples = %d | valid time = %.2f (s)' % (examples, eval_time.time())
                        )
            df.flush()
    # write file
    logger.info('Ranked documents wrote to file %s.', dev_filename_doc_ranked)
    logger.info('Success!')

def topk_metric_max_over_ground_truths(metric_fn, predictions, ground_truths, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(predictions, ground_truth, max_len)
        scores_for_ground_truths.append(score)
    scores_for_ground_truths = np.array(scores_for_ground_truths)
    return np.max(scores_for_ground_truths, 0).tolist()

def topk_scope_match_score(predictions, ground_truth, max_len):
    for i, prediction in enumerate(predictions):
        if normalize_answer(ground_truth) in normalize_answer(prediction):
            return [0] * i + [1] * (max_len - i)
    return [0] * max_len

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

if __name__ == '__main__':
    # set logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    args = set_args()

    if 'computecanada.ca' in socket.gethostname():
        workdir = '/home/mutux/projects/def-jynie/mutux/data/openqa'
        best_model = '/home/mutux/projects/def-jynie/mutux/output/best_acc.pth.tar'
    else:
        workdir = '/u/pandu/data/openQA/data'
        best_model = '/u/pandu/data/aqa_output/quasart/aranker/octal18-05-19-13-11-20/checkpoints/best_acc.pth.tar'

    args.embedding_file = workdir + "/embeddings/glove.840B.300d.txt"

    subset = 'debug'
    # subset = 'dev'
    # subset = 'train'
    # subset = 'test'
    # subset = 'local_debug'
    args.dataset = 'nqsub'

    dev_filename_doc = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm.json"
    dev_filename_qa = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm.txt"
    dev_filename_doc_ranked = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_rank_by_quasart.json"

    if subset == 'local_debug':
        dev_filename_doc = 'resources/debug.json'
        dev_filename_qa = 'resources/debug.txt'
        dev_filename_doc_ranked = 'resources/debug_rank.json'
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')
    args.default_num_docs = 50
    args.BATCH_SIZE = 64
    # evl(args, dev_filename_doc, dev_filename_qa, best_model)
    # pred_by_group(args, dev_filename_doc, dev_filename_qa, best_model, rank_file)
    rank_and_save(args, dev_filename_doc, dev_filename_qa, best_model, dev_filename_doc_ranked)
