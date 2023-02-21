from normdata import norm_data, norm_data_with_doc
from pre_docrank import baseline_rank_data
from pre_txttoken import tokenize_txt, tokenize_json
from nq_format import nq2quasart, quasart2nq
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

def rank_data(ifile, ofn_json, ofn_txt):
    nq2quasart(ifile, ofn_json, ofn_txt)
    norm_data(ofn_txt, ofn_txt_norm)
    norm_data_with_doc(ofn_json, ofn_json_norm)
    baseline_rank_data(ofn_json_norm, ofn_txt_norm, ofn_json_norm_orig, mtype='orig', top_n=100)
    tokenize_json(ofn_json_norm_orig, ofn_json_norm_tok_orig)
    tokenize_txt(ofn_txt_norm, ofn_txt_norm_tok)
    rank_and_save()
    quasart2nq(ifn_json_norm_tok_rank, ifile, ofile)


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
    generator.load_state_dict(checkpoint['gen_state'])
    logger.info("=> checkpoint loaded, current best EM %.2f" % best_acc)
    return best_acc, generator, word_dict, feature_dict


def rank_and_save(args, dev_filename_doc, dev_filename_qa, model_file, dev_filename_doc_ranked):
    # load model and dictionaries
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
                        new_doc_q["answers_tok"] = old_doc_q["answers_tok"]
                        new_doc_q["has_answers"] = old_doc_q["has_answers"]
                        new_doc_q["score"] = score_doc.item()
                        new_line_docs_q_ranked.append(new_doc_q)
                    # dev_docs_ranked.append(json.dumps(new_line_docs_q_ranked))
                    # Write file instead of save in memory
                    df.write(json.dumps(new_line_docs_q_ranked) + '\n')

                examples += batch_size

            df.flush()
    # write file
    logger.info('Ranked documents wrote to file %s.', dev_filename_doc_ranked)
    logger.info('Success!')


if __name__ == '__main__':
    # set logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    args = set_args()

    args.doc_layers = 1
    workdir = '/u/pandu/data/openQA/data'
    # best_model = '/u/pandu/data/aqa_output/quasart/aranker/cedar-07-21-21-00-38/best_acc.pth.tar'
    best_model = '/u/pandu/data/aqa_output/nqsub/aranker/cedar-11-03-22-44-02/best_acc.pth.tar'
    args.embedding_file = workdir + "/embeddings/glove.840B.300d.txt"

    # subset = 'debug'
    # subset = 'dev'
    subset = 'train'
    # subset = 'test'
    # subset = 'local_debug'
    args.dataset = 'nqsub'

    dev_filename_doc = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_tok_orig.json"
    dev_filename_qa = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm.tok.txt"
    # dev_filename_doc_ranked = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_tok_orig_rank_by_quasart.json"
    dev_filename_doc_ranked = workdir + "/datasets/" + args.dataset + "/" + subset + "_utf8.norm_tok_rank.json"

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

    rank_and_save(args, dev_filename_doc, dev_filename_qa, best_model, dev_filename_doc_ranked)
