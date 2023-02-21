import logging
from model import ranker
from utils import data_tok as data
import argparse
import os
import torch
import socket
from datetime import datetime
import json
import copy
import sys

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger()


def set_args():
    parser = argparse.ArgumentParser(
        'Adversarial Question Answering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, default='nqsub50',
                        help='datasset names in quasart,searchqa,unftriviaqa,webquestions,trec,nqsub, trec100, webquestions100')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size in 4, 8, 12, 16, 32')
    parser.add_argument('--num_epochs', type=int, default=8,
                        help='batch size in 4, 8, 12, 16, 32')
    parser.add_argument('--span_len', type=int, default=5,
                        help='batch size in 5, 10, 15')
    parser.add_argument('--default_num_docs', type=int, default=50,
                        help='num of docs for training in 50, 100 for trec100 and webquestions100, possibly nqsub100')
    parser.add_argument('--data_workers', type=int, default=0,
                        help='num of workers used by dataloader, 0, 1, 2, 3, 4, 5')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # args.cuda = False  # debug
    # args.device = torch.device("cuda:0" if args.cuda else "cpu")
    # args.device = torch.device('cpu')
    # args.dataset = 'searchqa'
    # args.dataset = 'nqsub'
    # args.dataset = 'unftriviaqa'
    # args.dataset = 'webquestions'
    # args.dataset = 'trec'
    # args.dataset = 'quasart'
    args.pt_str = datetime.now().strftime("%m-%d-%H-%M-%S") + '-' + socket.gethostname()
    args.FIX_EMB = True
    args.random_seed = 1
    args.D_WEIGHT_DECAY = 0.001
    args.C_WEIGHT_DECAY = 0.001
    args.G_WEIGHT_DECAY = 0.001
    args.D_LEARNING_RATE = 0.1
    args.C_LEARNING_RATE = 0.1
    args.G_LEARNING_RATE = 0.01
    args.D_MOMENTUM = 0.9
    args.C_MOMENTUM = 0.9
    args.G_MOMENTUM = 0.9
    args.DOC_TEMPERATURE = 0.9
    args.SPN_TEMPERATURE = 0.9
    # args.BATCH_SIZE = 4
    args.restrict_vocab = True
    args.type_max = True
    args.D_GRAD_CLIP = 0.5
    args.C_GRAD_CLIP = 0.5
    args.G_GRAD_CLIP = 0.5
    args.updates = 0
    args.num_sample_doc = 5
    args.num_sample_spn = 3


    args.dis_epochs = 3
    args.gen_epochs = 4
    # args.num_epochs = 6

    args.log_file = None
    args.num_span_gen = 15
    args.num_doc_gen = 10
    # args.span_len = 10
    args.parag_len = 100 # max len of paragraphs
    args.LAMBDA = 0.5
    # args.default_num_docs = 50

    args.use_qemb = True
    args.hidden_size = 128
    args.dropout_rnn = 0.3
    args.dropout_rnn_output = True
    args.dropout_emb = 0.3
    args.concat_rnn_layers = True
    args.rnn_padding = False
    args.question_merge = 'self_attn'
    args.fix_embeddings = True
    args.doc_layers = 1

    args.out_dir = "./output/" + args.dataset + "/aranker"
    args.chp_dir = args.out_dir + '/' + args.pt_str + '/' + 'checkpoints'
    args.run_dir = args.out_dir + '/' + args.pt_str + '/' + 'runs'
    args.log_file = args.out_dir + '/' + args.pt_str + '/' + args.dataset + '.log'
    args.conf_file = args.out_dir + '/' + args.pt_str + '/conf.json'
    args.stats_file = args.out_dir + '/' + args.pt_str + '/final_stats.json'
    # args.data_workers = 2
    args.num_display = (37012 // args.batch_size + 1) // 2 + 1
    return args


def set_out(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.out_dir + '/' + args.pt_str):
        os.mkdir(args.out_dir + '/' + args.pt_str)
    if not os.path.exists(args.out_dir + '/' + args.pt_str + '/' + 'runs'):
        os.mkdir(args.out_dir + '/' + args.pt_str + '/' + 'runs')
    if not os.path.exists(args.out_dir + '/' + args.pt_str + '/' + 'checkpoints'):
        os.mkdir(args.out_dir + '/' + args.pt_str + '/' + 'checkpoints')

    with open(args.conf_file, 'w') as af:
        tmp = vars(copy.deepcopy(args))
        # if 'device' in tmp:
        #     tmp['device'] = "cuda:0" if args.cuda else "cpu"
        json.dump(tmp, af)


def set_logger(args):
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)


if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    args = set_args()
    stage = 'debug'

    # Set data
    if socket.gethostname() == 'mutux':
        stage = 'localdebug'
    elif sys.gettrace() is not None: # or socket.gethostname() == 'octal20':
        # stage = 'localdebug'
        stage = 'debug'
        # stage = 'train'
    else:
        stage = 'train'
        # stage = 'test'

    if stage == 'localdebug':
        workdir = './resources'
        args.embedding_file = workdir + '/glove.6B.50d.txt'
        filename_doc = workdir + "/debug_tok_orig.json"
        filename_qa = workdir + '/debug.tok.txt'
        dev_filename_doc = workdir + '/debug_tok_orig.json'
        dev_filename_qa = workdir + '/debug.tok.txt'
        args.out_dir = "output/localdebug" + '/' + args.dataset + '/aranker'
        args.chp_dir = args.out_dir + '/' + args.pt_str + '/' + 'checkpoints'
        args.run_dir = args.out_dir + '/' + args.pt_str + '/' + 'runs'
        args.log_file = args.out_dir + '/' + args.pt_str + '/' + args.dataset + '.log'
        args.conf_file = args.out_dir + '/' + args.pt_str + '/conf.json'
        args.stats_file = args.out_dir + '/' + args.pt_str + '/final_stats.json'
        args.data_workers = 0
        args.dis_epochs = 1
        args.gen_epochs = 2
        args.num_epochs = 2
    elif stage == 'debug':
        workdir = '/u/pandu/data/openQA/data'
        args.embedding_file = workdir + "/embeddings/glove.840B.300d.txt"
        filename_doc = workdir + "/datasets/" + args.dataset + "/debug_utf8.norm_tok_orig.json"
        filename_qa = workdir + "/datasets/" + args.dataset + "/debug_utf8.norm.tok.txt"
        dev_filename_doc = workdir + "/datasets/" + args.dataset + "/debug_utf8.norm_tok_orig.json"
        dev_filename_qa = workdir + "/datasets/" + args.dataset + "/debug_utf8.norm.tok.txt"
        args.out_dir = "output/debug" + '/' + args.dataset + '/aranker'
        args.chp_dir = args.out_dir + '/' + args.pt_str + '/' + 'checkpoints'
        args.run_dir = args.out_dir + '/' + args.pt_str + '/' + 'runs'
        args.log_file = args.out_dir + '/' + args.pt_str + '/' + args.dataset + '.log'
        args.conf_file = args.out_dir + '/' + args.pt_str + '/conf.json'
        args.stats_file = args.out_dir + '/' + args.pt_str + '/final_stats.json'
    elif stage == 'train':
        if 'computecanada.ca' in socket.gethostname():
            workdir = '/home/mutux/projects/def-jynie/mutux/data/openqa'
        elif socket.gethostname().lower().startswith('thu-'):
            workdir = '/home/zlx/data'
        else:
            workdir = '/u/pandu/data/openQA/data'
        args.embedding_file = workdir + "/embeddings/glove.840B.300d.txt"
        filename_doc = workdir + "/datasets/" + args.dataset + "/train_utf8.norm_tok_orig.json"
        filename_qa = workdir + "/datasets/" + args.dataset + "/train_utf8.norm.tok.txt"
        if args.dataset.startswith('webquestions') or args.dataset.startswith('trec'):
            dev_filename_doc = workdir + "/datasets/" + args.dataset + "/test_utf8.norm_tok_orig.json"
            dev_filename_qa = workdir + "/datasets/" + args.dataset + "/test_utf8.norm.tok.txt"
            args.parag_len = 350
        else:
            dev_filename_doc = workdir + "/datasets/" + args.dataset + "/dev_utf8.norm_tok_orig.json"
            dev_filename_qa = workdir + "/datasets/" + args.dataset + "/dev_utf8.norm.tok.txt"
            args.parag_len = 200
    else:
        workdir = '/u/pandu/data/openQA/data'
        filename_doc = workdir + "/datasets/" + args.dataset + "/test_utf8.norm_tok_orig.json"
        filename_qa = workdir + "/datasets/" + args.dataset + "/test_utf8.norm.tok.txt"
        dev_filename_doc = None
        dev_filename_qa = None
        args.embedding_file = workdir + "/embeddings/glove.840B.300d.txt"

    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    args.checkpoint = True

    set_out(args)
    set_logger(args)

    logger.info('-' * 100)
    logger.info('Executing ' + os.path.abspath(__file__) + ' on: ' + args.pt_str)

    # pre_best_model = '/u/pandu/data/aqa_output/quasart/aranker/octal18-05-06-14-04-22/checkpoints/best_acc.pth.tar'
    stats = {'timer': data.Timer(), 'epoch': 0, 'best_valid': 0.0, 'best_model_file': None}
    # stats['best_model_file'] = pre_best_model
    pre_best_acc = 0.0

    qa = ranker.Ranker(args, filename_doc, filename_qa, dev_filename_doc, dev_filename_qa, best_model=stats['best_model_file'])

    logger.info('-' * 100)
    # if stats['best_model_file']:
    #     qa.load_checkpoint_for_resume(stats['best_model_file'])

    if not stats['best_model_file']:
        qa.init_optimizer()
    else:
        pre_best_acc = qa.load_checkpoint_for_resume(stats['best_model_file'])
    qa.parallelize()
    for epoch in range(args.num_epochs):
        stats['epoch'] = epoch
        qa.fight(stats, args.dis_epochs, args.gen_epochs)

    logger.info('-' * 100)
    stats['newfound'] = int(pre_best_acc < stats['best_valid'])
    logger.info("Best_EM %.2f, Is_Better: %d" % (stats['best_valid'], stats['newfound']))
    logger.info("Out: %s" % stats['best_model_file'])
    logger.info('-' * 100)
    with open(args.stats_file, 'w') as bf:
        stats['timer'] = stats['timer'].time()
        json.dump(stats, bf)
    logger.info('Done.')
