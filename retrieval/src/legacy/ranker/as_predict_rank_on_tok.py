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
        with io.open(dev_filename_doc_ranked, 'w+', encoding='utf-8') as df:
            # for idx, ex_with_doc in enumerate(tqdm(dev_loader_with_doc, total=len(dev_loader_with_doc))):
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

                examples += batch_size

            df.flush()
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
                hits = []
                for j in range(len(indices[idx_q, 0, :])):
                    idx_doc = indices[idx_q, 0, j]
                    has_ans = dev_docs[ex_id[idx_q]][idx_doc % len(dev_docs[ex_id[idx_q]])]["has_answers"]
                    hits.append(1 if has_ans[0] else 0)

                mtsi = topk_hits_max_over_ground_truths(hits, max_len)
                for j in range(max_len):
                    exact_matchs[j].update(mtsi[j])
            examples += batch_size
        rslt_str = ""
        for i, ele in enumerate(exact_matchs):
            if i%10 == 0:
                rslt_str += '\n\t\t'
            rslt_str += 'E'+str(i+1)+ ": "+"{:.3f}".format(ele.avg*100)+'\t| '
        logger.info("Eval Results: \t"+ rslt_str)
        logger.info('examples = %d | valid time = %.2f (s)' % (examples, eval_time.time()))

    logger.info('Success!')


def topk_hits_max_over_ground_truths(hit_order_over_ground, max_len):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    for i, hit in enumerate(hit_order_over_ground):
        if hit:
            return [0] * i + [1] * (max_len - i)
    return [0] * max_len



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
    subset = 'train'
    # subset = 'test'

    args.dataset = 'trecqa_as_abl'

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
        mdl_key = 'trec'
        # mdl_key = 'webquestions'
        # mdl_key = 'quasart'
        # mdl_key = 'searchqa'
        # mdl_key = 'unftriviaqa'
        # mdl_key = 'nqsub'

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


    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')
    args.default_num_docs = 100
    args.BATCH_SIZE = 128
    # evl(args, dev_filename_doc, dev_filename_qa, best_model)
    # pred_by_group(args, dev_filename_doc, dev_filename_qa, best_model, rank_file)
    # rank_and_save(args, dev_filename_doc, dev_filename_qa, best_model, dev_filename_doc_ranked)

    # evaluation for report only
    eval_rank(args, dev_filename_doc, dev_filename_qa, best_model)

# test_rank 55
# E1 = 58.82 | E3 = 60.29 | E5 = 64.71 | E20 = 88.24 | E50 = 100.00
# test_rank 50
# E1 = 58.82 | E3 = 63.24 | E5 = 67.65 | E20 = 88.24 | E50 = 100.00
# test_rank 100
# E1 = 58.82 | E3 = 58.82 | E5 = 63.24 | E20 = 76.47 | E50 = 89.71
# test_rank 20
# E1 = 57.35 | E3 = 69.12 | E5 = 80.88 | E20 = 100.00 |

# dev_rank 50
# E1 = 49.28 | E3 = 52.17 | E5 = 63.77 | E20 = 88.41 | E50 = 100.00 |
# dev_rank 100
# E1 = 50.73 | E3 = 52.17 | E5 = 53.62 | E20 = 79.71 | E50 - 88.41
# dev_rank 20
# E1 = 44.93 | E3 = 63.77 | E5 = 75.36 | E20 = 100.00

# train_rank 50
# E1 = 71.91 | E3 = 78.65 | E5 = 85.39 | E20 = 93.63 | E50 = 98.88 |
# train_rank 100
# E1 = 73.03 | E3 = 74.16 | E5 = 78.65 | E20 = 93.26 | E50 = 97.75 |
# train_rank 20
# E1 = 57.30 | E3 = 75.28 | E5 = 84.27 | E20 = 94.38
