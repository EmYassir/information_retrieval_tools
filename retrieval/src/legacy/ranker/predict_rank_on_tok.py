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
from utils import data_tok as dtld
import logging
import argparse
from model.genranker import GEN
from functools import partial
from utils import vector
import json
from tqdm import tqdm
import numpy as np
import re
import string
import io
import socket
from datetime import datetime

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

    args.use_qemb = True
    args.doc_layers = 2
    args.hidden_size = 128
    args.dropout_rnn = 0.3
    args.dropout_rnn_output = True
    args.dropout_emb = 0.3
    args.concat_rnn_layers = True
    args.rnn_padding = False
    args.question_merge = 'self_attn'
    args.fix_embeddings = True
    args.data_workers = 0
    args.num_display = 200
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
        logger.info("start processing group with len: " + str(key) + ' total: ' + str(len(dt_lst)))
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
    try:
        generator.load_state_dict(checkpoint['gen_state'])
    except:
        logger.info("Trying to load generator with DataParallel")
        try:
            generator = torch.nn.DataParallel(generator)
            generator.load_state_dict(checkpoint['gen_state'])
        except:
            logger.info("Generator cannot be loaded...")

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
    eval_time = dtld.Timer()
    exact_matchs = [dtld.AverageMeter() for i in range(max_len)]
    # tot_len = sum(1 for i in open(dev_filename_qa, 'r'))
    with torch.no_grad():
        examples = 0
        time_pred = 0
        with io.open(dev_filename_doc_ranked, 'w+', encoding='utf-8') as df:
            # for idx, ex_with_doc in enumerate(tqdm(dev_loader_with_doc, total=len(dev_loader_with_doc))):

            for ex_with_doc in tqdm(dev_loader_with_doc):
                ex = ex_with_doc[0]
                st = datetime.now()
                batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
                if isinstance(generator, torch.nn.DataParallel):
                    scores_docs = generator.module.predict(ex_with_doc)
                else:
                    scores_docs = generator.predict(ex_with_doc)
                time_pred += (datetime.now()-st).microseconds

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
                        new_doc_q["answers_tok"] = old_doc_q["answers_tok"]
                        new_doc_q["has_answers"] = old_doc_q["has_answers"]
                        new_doc_q["score"] = score_doc.item() * 1000
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
                            # ground_truths.append(" ".join([w for w in a]))
                            ground_truths.append(a)
                    mtsi = topk_metric_max_over_ground_truths(topk_scope_match_score, predictions, ground_truths,
                                                              max_len)
                    for j in range(max_len):
                        exact_matchs[j].update(mtsi[j])
                examples += batch_size
            logger.info('Eval Results:'
                        + '\n                          E1 = %.4f | E3 = %.4f | E5 = %.4f | E20 = %.4f | E50 = %.4f |' % (
                            exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100,
                            exact_matchs[19].avg * 100, exact_matchs[49].avg * 100)
                        + '\n                          examples = %d | valid time = %.2f (s)' % (
                        examples, eval_time.time())
                        )
            df.flush()
        logger.info("Prediction time used: %d." % time_pred)
    # write file
    logger.info('Ranked documents wrote to file %s.', dev_filename_doc_ranked)
    logger.info('Success!')


def eval_rank(args, dev_filename_doc, dev_filename_qa, model_file):
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

    # evaluate
    max_len = args.default_num_docs
    eval_time = dtld.Timer()
    exact_matchs = [dtld.AverageMeter() for i in range(max_len)]
    # tot_len = sum(1 for i in open(dev_filename_qa, 'r'))
    with torch.no_grad():
        examples = 0
        for ex_with_doc in tqdm(dev_loader_with_doc):
            ex = ex_with_doc[0]

            batch_size, _, ex_id = ex[0].size(0), ex[3], ex[-1]
            if isinstance(generator, torch.nn.DataParallel):
                scores_docs = generator.module.predict(ex_with_doc)
            else:
                scores_docs = generator.predict(ex_with_doc)
            scores_docs = scores_docs.detach()
            _, indices = scores_docs.sort(2, descending=True)

            for idx_q in range(batch_size):

                predictions = []
                for j in range(len(indices[idx_q, 0, :])):
                    idx_doc = indices[idx_q, 0, j]
                    doc_text = dev_docs[ex_id[idx_q]][idx_doc % len(dev_docs[ex_id[idx_q]])]["document"]
                    predictions.append(" ".join(doc_text))

                ground_truths = []
                answer = dev_exs_with_doc[ex_id[idx_q]]['answers']
                if (args.dataset == "CuratedTrec"):
                    ground_truths = answer
                else:
                    for a in answer:
                        ground_truths.append(a)
                mtsi = topk_metric_max_over_ground_truths(topk_scope_match_score, predictions, ground_truths,
                                                          max_len)
                for j in range(max_len):
                    exact_matchs[j].update(mtsi[j])
            examples += batch_size
        logger.info('Eval Results:'
                    + '\n                          E1 = %.2f | E3 = %.2f | E5 = %.2f | E20 = %.2f | E50 = %.2f |' % (
                        exact_matchs[0].avg * 100, exact_matchs[2].avg * 100, exact_matchs[4].avg * 100,
                        exact_matchs[19].avg * 100, exact_matchs[49].avg * 100)
                    + '\n                          examples = %d | valid time = %.2f (s)' % (
                    examples, eval_time.time())
                    )

    logger.info('Success!')

def topk_metric_max_over_ground_truths(metric_fn, predictions, ground_truths, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    if len(ground_truths) == 0:
        return [0.0] * max_len
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

    rankers = {
        "octal": {
            "quasart": "/u/pandu/pyspace/sigir/ranker/output/quasart/aranker/12-04-13-46-44-octal18/checkpoints/best_acc.pth.tar",
            "trec": "/u/pandu/pyspace/sigir/ranker/output/trec/aranker/12-25-00-32-12-Thu-407-X299/checkpoints/best_acc.pth.tar",
            "webquestions": "/u/pandu/pyspace/sigir/ranker/output/webquestions/aranker/12-23-00-54-09-Thu-407-X299/checkpoints/best_acc.pth.tar",
            "searchqa": "/u/pandu/pyspace/sigir/ranker/output/searchqa/aranker/12-10-14-41-34-cdr2496.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "unftriviaqa": "/u/pandu/pyspace/sigir/ranker/output/unftriviaqa/aranker/12-26-17-18-01-cdr2513.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "nqsub": "/u/pandu/pyspace/sigir/ranker/output/nqsub/aranker/cedar-11-03-22-44-02/best_acc.pth.tar"
        },
        "computecanada": {
            "quasart": "/home/mutux/projects/def-jynie/mutux/pyspace/aqa/aranker/output/quasart/aranker/cdr209.int.cedar.computecanada.ca-07-21-21-00-38/checkpoints/best_acc.pth.tar",
            "trec": "",
            "webquestions":"",
            "searchqa": "/home/mutux/projects/def-jynie/mutux/pyspace/sigir2021/ranker/output/searchqa/aranker/12-10-14-41-34-cdr2496.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "unftriviaqa": "/home/mutux/projects/def-jynie/mutux/pyspace/sigir2021/ranker/output/unftriviaqa/aranker/12-26-17-18-01-cdr2513.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "nqsub": "/home/mutux/projects/def-jynie/mutux/pyspace/aqa/aranker/output/nqsub/aranker/11-03-22-44-02-cdr2500.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar"
        },
        "thu": {
            "quasart": "",
            "trec": "",
            "webquestions": "",
            "searchqa": "/home/zlx/ranker/output/searchqa/aranker/12-10-14-41-34-cdr2496.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "unftriviaqa": "/home/zlx/ranker/output/unftriviaqa/aranker/12-26-17-18-01-cdr2513.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "nqsub": "/home/zlx/ranker/output/nqsub/aranker/cedar-11-03-22-44-02/best_acc.pth.tar"
        },
    }


    # subset = 'debug'
    # subset = 'dev'
    # subset = 'train'
    subset = 'test'
    # subset = 'testcase'

    # args.dataset = 'nqsub'
    # args.dataset = 'searchqa'
    # args.dataset = 'unftriviaqa'
    # args.dataset = 'quasart'
    args.dataset = 'trec' #0.8775
    # args.dataset = 'trec_d100_p500' # 0.9395
    args.dataset = 'trec100_full' #@50 E1 = 33.29 | E3 = 54.61 | E5 = 64.12 | E20 = 80.98 | E50 = 84.58 |
    # args.dataset = 'webquestions'
    # args.dataset = 'webquestions100_full'
    # args.dataset = 'webquestions_d100_p500'
    # args.dataset = 'unftriviaqa100'
    # args.dataset = 'wqsub'
    # args.dataset = 'nqsub200'
    # args.dataset = 'nqsub'
    # args.dataset = 'quasart'

    if 'computecanada.ca' in socket.gethostname():
        # E1 = 38.03 > 35.0
        args.doc_layers = 1
        workdir = '/home/mutux/projects/def-jynie/mutux/data/openqa'
        best_model = rankers["computecanada"][args.dataset]
        # best_model = '/home/mutux/projects/def-jynie/mutux/pyspace/aqa/aranker/output/nqsub/aranker/11-03-22-44-02-cdr2500.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar'
    elif 'octal' in socket.gethostname():
        # print(args.dataset)
        args.doc_layers = 1
        workdir = '/u/pandu/data/openQA/data'
        # best_model = '/u/pandu/data/aqa_output/quasart/aranker/cedar-07-21-21-00-38/best_acc.pth.tar'
        if args.dataset.startswith('trec'):
            mdl_key = 'trec'
        elif args.dataset.startswith('webquestions') or args.dataset.startswith('wqsub'):
            mdl_key = 'webquestions'
        elif args.dataset.startswith('unftriviaqa'):
            mdl_key = 'unftriviaqa'
        elif args.dataset.startswith('nqsub'):
            mdl_key = 'nqsub'
        else:
            mdl_key = args.dataset
        best_model = rankers['octal'][mdl_key]
        # best_model = '/u/pandu/data/aqa_output/nqsub/aranker/cedar-11-03-22-44-02/best_acc.pth.tar'
    elif socket.gethostname().lower().startswith('thu-'):
        args.doc_layers = 1
        workdir = '/home/zlx/data'
        # best_model = '/u/pandu/data/aqa_output/quasart/aranker/cedar-07-21-21-00-38/best_acc.pth.tar'
        best_model = rankers['thu'][args.dataset]
    args.embedding_file = workdir + "/embeddings/glove.840B.300d.txt"

    dev_filename_doc = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_tok_orig.json"
    dev_filename_qa = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm.tok.txt"
    # dev_filename_doc_ranked = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_tok_orig_rank_by_quasart.json"
    dev_filename_doc_ranked = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_tok_rank.json"

    sst = datetime.now()
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
    # rank_and_save(args, dev_filename_doc, dev_filename_qa, best_model, dev_filename_doc_ranked)
    logger.info("Total time used %d seconds" % (datetime.now() - sst).seconds)
    # evaluation for report only
    eval_rank(args, dev_filename_doc_ranked, dev_filename_qa, best_model)

# wqsub on dpr+gan
# E1 = 20.96 | E3 = 35.93 | E5 = 43.50 | E20 = 59.15 | E50 = 65.26 |
# wqsub on dpr
# E1 = 35.8268 | E3 = 46.7520 | E5 = 50.5413 | E10 = 55.3642 | E20 = 59.7441 | E50 = 65.2559 | E100 = 68.1594 |