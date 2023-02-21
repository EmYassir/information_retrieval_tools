
from model.disranker import DIS
from model.disclass import CLS
from model.genranker import GEN
from utils import vector
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import random
from utils import data_tok as dtld
from eval import accranker, ndcgr
from functools import partial
import shutil
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import socket

if 'computecanada.ca' not in socket.gethostname():
    import tensorflow as tf

logger = logging.getLogger(__name__)


class Ranker(object):
    def __init__(self, args, filename_train_docs, filename_train, filename_dev_docs, filename_dev, best_model=None):
        logger.info('-' * 100)
        logger.info('Creating AQA object for dataset ' + args.dataset)
        # self.device = torch.device('cpu')
        self.args = args
        # set random seeds for numpy, torch, and cuda.
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.random_seed)

        # self.device = args.device
        self.num_docs = args.default_num_docs
        self.parag_len = args.parag_len
        self.HasAnswer_Map = {}

        self.train_docs, self.train_questions, _ = dtld.load_data_with_doc(filename_train_docs, self.num_docs)
        self.train_exs_with_doc = dtld.read_data(filename_train, self.train_questions)
        self.dev_docs, self.dev_questions, _ = dtld.load_data_with_doc(filename_dev_docs, self.num_docs)
        self.dev_exs_with_doc = dtld.read_data(filename_dev, self.dev_questions)

        # --------------------------------------------------------------------------
        # Discriminator and Generator
        # share the embedding weights
        self.generator = None
        self.discriminator = None
        self.classifier = None
        self.D_optimizer = None
        self.G_optimizer = None
        self.feature_dict = None
        self.word_dict = None

        if best_model:
            self.load_checkpoint_for_resume(best_model)
        else:
            logger.info('Building feature dict')
            self.feature_dict = dtld.build_feature_dict()
            logger.info('Building word dict from docs')
            self.word_dict = dtld.build_word_dict_docs(self.train_docs, self.args.restrict_vocab,
                                                       self.args.embedding_file)
            self.args.num_features = len(self.feature_dict)
            self.args.vocab_size = len(self.word_dict)
            logger.info('Constructing generator')
            self.generator = GEN(self.args).cuda()
            logger.info('Constructing discriminator')
            self.discriminator = DIS(self.args).cuda()
            logger.info('Constructing classifier')
            self.classifier = CLS(self.args).cuda()
        logger.info('Num features = %d' % len(self.feature_dict))
        logger.info('Num words = %d' % len(self.word_dict))

        # -------------------------------------------------------------------------
        # load embeddings
        self.load_embeddings_for_dis()
        self.generator.embedding.weight.data.copy_(self.discriminator.embedding.weight.data)
        self.classifier.embedding.weight.data.copy_(self.discriminator.embedding.weight.data)

        # --------------------------------------------------------------------------
        # DATA ITERATORS
        # Two datasets: train and dev. If we sort by length it's faster.
        logger.info('-' * 100)
        logger.info('Make data loaders')

        train_dataset_with_doc = dtld.ReaderDataset_with_Doc(self.train_exs_with_doc, self.train_docs, self.word_dict, self.feature_dict, self.num_docs, single_answer=False)
        self.train_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(train_dataset_with_doc)
        self.train_loader_with_doc = torch.utils.data.DataLoader(
            train_dataset_with_doc,
            batch_size=self.args.BATCH_SIZE,
            sampler=self.train_sampler_with_doc,
            num_workers=self.args.data_workers,
            collate_fn=partial(vector.batchify_with_docs, num_docs=self.num_docs, parag_len=self.parag_len),
            pin_memory=self.args.cuda,
        )

        dev_dataset_with_doc = dtld.ReaderDataset_with_Doc(self.dev_exs_with_doc, self.dev_docs, self.word_dict, self.feature_dict, self.num_docs, single_answer=False)
        self.dev_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(dev_dataset_with_doc)
        self.dev_loader_with_doc = torch.utils.data.DataLoader(
            dev_dataset_with_doc,
            batch_size=self.args.BATCH_SIZE,
            sampler=self.dev_sampler_with_doc,
            num_workers=self.args.data_workers,
            collate_fn=partial(vector.batchify_with_docs, num_docs=self.num_docs, parag_len=self.parag_len),
            pin_memory=self.args.cuda,
        )

        # self.writer = None
        # if 'computecanada.ca' not in socket.gethostname():
        #     logger.info('Usring Tensorflow on ' + socket.gethostname() + ' to write summary!')
        #     self.writer = tf.summary.FileWriter(self.args.run_dir)
        self.writer = SummaryWriter(self.args.run_dir)
        logger.info('AQA Created.')

    def load_embeddings_for_dis(self):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in self.word_dict.tokens() if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), self.args.embedding_file))
        embedding = self.discriminator.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(self.args.embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]]).cuda()
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logger.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self, D_opt_state_dict=None, G_opt_state_dict=None, C_opt_state_dict=None):
        if not D_opt_state_dict:
            logger.info("initializing optimizer from scratch")
        else:
            logger.info("initializing optimizer from checkpoints")
        if self.args.fix_embeddings:
            for p in self.discriminator.embedding.parameters():
                p.requires_grad = False
            for p in self.generator.embedding.parameters():
                p.requires_grad = False
            for p in self.classifier.embedding.parameters():
                p.requires_grad = False

        D_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        self.D_optimizer = optim.SGD(D_parameters, self.args.D_LEARNING_RATE, momentum=self.args.D_MOMENTUM,
                                weight_decay=self.args.D_WEIGHT_DECAY)
        C_parameters = [p for p in self.classifier.parameters() if p.requires_grad]
        self.C_optimizer = optim.SGD(C_parameters, self.args.C_LEARNING_RATE, momentum=self.args.C_MOMENTUM,
                                     weight_decay=self.args.C_WEIGHT_DECAY)
        G_parameters = [p for p in self.generator.parameters() if p.requires_grad]
        self.G_optimizer = optim.SGD(G_parameters, self.args.G_LEARNING_RATE, momentum=self.args.G_MOMENTUM,
                                weight_decay=self.args.G_WEIGHT_DECAY)
        if D_opt_state_dict:
            self.D_optimizer.load_state_dict(D_opt_state_dict)
        if C_opt_state_dict:
            self.C_optimizer.load_state_dict(C_opt_state_dict)
        if G_opt_state_dict:
            self.G_optimizer.load_state_dict(G_opt_state_dict)

    # @profile
    def fight_old(self, global_stats, dep, gep):
        """Run through one epoch of model training with the provided data loader."""
        # Initialize meters + timers
        d_train_loss = dtld.AverageMeter()
        g_train_loss = dtld.AverageMeter()
        epoch_time = dtld.Timer()
        exs_with_doc = self.train_exs_with_doc
        docs_by_question = self.train_docs
        data_loader = self.train_loader_with_doc

        if not global_stats['best_model_file']:
            self.init_optimizer()
        # Run one epoch
        logger.info(' ' * 30)
        logger.info('Training DIS...')
        logger.info('+' * 30)
        update_step = 0

        self.discriminator.train()
        self.generator.eval()
        dat, d_labels = None, None
        for d_epoch in range(dep):
            for idx, ex_with_doc in enumerate(data_loader):
                ex = ex_with_doc[0]
                batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

                if idx not in self.HasAnswer_Map:
                    HasAnswer_list = []
                    for idx_doc in range(0, self.num_docs):  # iterate 100 documents
                        # for the idx_doc th position in the 100, get answer for all the questions in the batch.
                        HasAnswer = []
                        for i in range(batch_size):  # iterate 4 questions
                            # HasAnswer.append(has_answer(exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][
                            #     idx_doc % len(docs_by_question[ex_id[i]])]["document"]))
                            HasAnswer.append(
                                docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"])
                        HasAnswer_list.append(HasAnswer)
                    self.HasAnswer_Map[idx] = HasAnswer_list
                else:
                    HasAnswer_list = self.HasAnswer_Map[idx]
                hasAnswer_list_sample = HasAnswer_list
                ex_with_doc_sample = ex_with_doc
                if idx % (self.args.num_display * 2) == 0:
                    dat, d_labels = self.g_4_d(ex_with_doc_sample, hasAnswer_list_sample, docs_by_question, exs_with_doc)
                d_train_loss.update(*self.dis_update_with_doc(dat, d_labels))
                self.add_scalar("DIS/Loss/train", d_train_loss.avg, idx)
                self.add_scalar("DIS/Time/train", global_stats['timer'].time(), idx)
                update_step = (update_step + 1) % 4
                if (idx + 1) % self.args.num_display == 0:
                    logger.info('train: Epoch.d_epoch = %d.%d | iter = %d/%d | ' %
                                (global_stats['epoch'], d_epoch, idx, len(data_loader)) +
                                'loss = %.4f | elapsed time = %.2f (s)' %
                                (d_train_loss.avg, global_stats['timer'].time()))
            logger.info('train: Epoch.d_epoch = %d.%d | iter = %d/%d | ' %
                        (global_stats['epoch'], d_epoch, len(data_loader), len(data_loader)) +
                        'loss = %.4f | elapsed time = %.2f (s), done.' %
                        (d_train_loss.avg, epoch_time.time()))

            d_train_loss.reset()
        logger.info(' ' * 30)
        logger.info('Training GEN...')
        logger.info('+' * 30)
        best_acc = global_stats['best_valid']

        self.generator.train()
        self.discriminator.eval()
        for g_epoch in range(gep):

            for idx, ex_with_doc in enumerate(data_loader):

                ex = ex_with_doc[0]
                batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

                if (idx not in self.HasAnswer_Map):
                    HasAnswer_list = []
                    for idx_doc in range(0, self.num_docs):  # iterate 100 documents
                        # for the idx_doc th position in the 100, get answer for all the questions in the batch.
                        HasAnswer = []
                        for i in range(batch_size):  # iterate 4 questions
                            # HasAnswer.append(has_answer(exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][
                            #     idx_doc % len(docs_by_question[ex_id[i]])]["document"]))
                            HasAnswer.append(docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"])
                        HasAnswer_list.append(HasAnswer)
                    self.HasAnswer_Map[idx] = HasAnswer_list
                else:
                    HasAnswer_list = self.HasAnswer_Map[idx]

                weights = []
                for idx_doc in range(0, self.num_docs):
                    weights.append(1)
                weights = torch.Tensor(weights)
                idx_random = torch.multinomial(weights, int(self.num_docs))

                hasAnswer_list_sample = []
                ex_with_doc_sample = []
                for idx_doc in idx_random:
                    hasAnswer_list_sample.append(HasAnswer_list[idx_doc])
                    ex_with_doc_sample.append(ex_with_doc[idx_doc])

                # g_train_loss.update(*self.gen_update_with_doc(ex_with_doc_sample))
                g_train_loss.update(*self.gen_update_with_doc(ex_with_doc_sample, hasAnswer_list_sample))
                if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
                    logger.info('train: Epoch.g_epoch = %d.%d | iter = %d/%d | ' % (global_stats['epoch'], g_epoch, idx, len(data_loader)) + 'loss = %.4f | elapsed time = %.2f (s)' % (g_train_loss.avg, global_stats['timer'].time()))
                    self.add_scalar("GEN/Loss/train", g_train_loss.avg, idx)
                    self.add_scalar("GEN/Time/train", global_stats['timer'].time(), idx)
                    # acc.eval_unofficial_with_doc(self.args, data_loader, self.generator, global_stats, exs_with_doc,
                    #                                  docs_by_question, 'train')
                if (idx + 1) % self.args.num_display == 0 or (idx + 1) == len(data_loader):
                    with torch.no_grad():
                        acc_rslt = accranker.topk_eval_unofficial_with_doc(self.args, self.dev_loader_with_doc, self.generator, global_stats, self.dev_exs_with_doc, self.dev_docs, 'dev', topk=150)
                        self.add_scalar("GEN/Accuracy/validate", acc_rslt["exact_match"], idx)

                        best_filename = "best_acc.pth.tar"
                        if acc_rslt["exact_match"] > best_acc:
                            best_acc = acc_rslt["exact_match"]
                            is_best_acc = True
                            best_filename = 'best_acc.pth.tar'
                            global_stats['best_model_file'] = self.args.chp_dir + '/' + best_filename
                            global_stats['best_valid'] = best_acc
                        else:
                            is_best_acc = False
                        checkpointname = 'checkpoint.pth.tar'
                        self.add_scalar("GEN/Best_Accuracy/validate", best_acc, idx)
                        state = {"g_epoch": g_epoch, "best_acc": best_acc,
                                 'dis_state': self.discriminator.state_dict(), 'dis_optimizer': self.D_optimizer.state_dict(),
                                 'gen_state': self.generator.state_dict(), 'gen_optimizer': self.G_optimizer.state_dict(), 'word_dict': self.word_dict, 'feature_dict': self.feature_dict}

                        self.save_checkpoint(state, is_best_acc, global_stats['best_model_file'], self.args.chp_dir + '/' + checkpointname)

                        # logger.info('train: Epoch.g_epoch = %d.%d | iter = %d/%d | ' %
                        #             (global_stats['epoch'], g_epoch, len(data_loader), len(data_loader)) +
                        #             'loss = %.4f | elapsed time = %.2f (s), done.' %
                        #             (g_train_loss.avg, epoch_time.time()))
            g_train_loss.reset()
        logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' % (global_stats['epoch'], epoch_time.time()))

    # def fight(self, global_stats, dep, gep):
    #     """Run through one epoch of model training with the provided data loader."""
    #     # Initialize meters + timers
    #     d_train_loss = dtld.AverageMeter()
    #     c_train_loss = dtld.AverageMeter()
    #     g_train_loss = dtld.AverageMeter()
    #     epoch_time = dtld.Timer()
    #     exs_with_doc = self.train_exs_with_doc
    #     docs_by_question = self.train_docs
    #     data_loader = self.train_loader_with_doc
    #
    #     if not global_stats['best_model_file']:
    #         self.init_optimizer()
    #     # Run one epoch
    #     logger.info(' ' * 30)
    #     logger.info('Training DIS...')
    #     logger.info('+' * 30)
    #     update_step = 0
    #
    #     self.discriminator.train()
    #     self.classifier.train()
    #     self.generator.eval()
    #     d_data= None, None
    #     c_data= None, None
    #     for d_epoch in range(dep):
    #         if d_epoch % 3 == 0:
    #             d_data = []
    #             c_data = []
    #             for idx, ex_with_doc in enumerate(data_loader):
    #                 ex = ex_with_doc[0]
    #                 batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
    #
    #                 if idx not in self.HasAnswer_Map:
    #                     HasAnswer_list = []
    #                     for idx_doc in range(0, self.num_docs):  # iterate 100 documents
    #                         # for the idx_doc th position in the 100, get answer for all the questions in the batch.
    #                         HasAnswer = []
    #                         for i in range(batch_size):  # iterate 4 questions
    #                             # HasAnswer.append(has_answer(exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][
    #                             #     idx_doc % len(docs_by_question[ex_id[i]])]["document"]))
    #                             HasAnswer.append(
    #                                 docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"])
    #                         HasAnswer_list.append(HasAnswer)
    #                     self.HasAnswer_Map[idx] = HasAnswer_list
    #                 else:
    #                     HasAnswer_list = self.HasAnswer_Map[idx]
    #                 hasAnswer_list_sample = HasAnswer_list
    #                 ex_with_doc_sample = ex_with_doc
    #
    #                 dat, d_labels = self.g_4_d(ex_with_doc_sample, hasAnswer_list_sample, docs_by_question, exs_with_doc)
    #                 d_data.append((dat, d_labels))
    #
    #                 d_train_loss.update(*self.dis_update_with_doc(dat, d_labels))
    #                 self.add_scalar("DIS/Loss/train", d_train_loss.avg, idx)
    #                 self.add_scalar("DIS/Time/train", global_stats['timer'].time(), idx)
    #                 update_step = (update_step + 1) % 4
    #                 if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
    #                     logger.info('E.d=%d.%d | iter=%d/%d | ' %
    #                                 (global_stats['epoch'], d_epoch, idx+1, len(data_loader)) +
    #                                 'loss=%.4f | time=%.1fs' %
    #                                 (d_train_loss.avg, global_stats['timer'].time()))
    #
    #                 cdat, c_labels = self.g_4_c(ex_with_doc_sample, hasAnswer_list_sample, docs_by_question, exs_with_doc)
    #                 c_data.append((cdat, c_labels))
    #                 c_train_loss.update(*self.cls_update_with_doc(cdat, c_labels))
    #                 self.add_scalar("CLS/Loss/train", c_train_loss.avg, idx)
    #                 self.add_scalar("CLS/Time/train", global_stats['timer'].time(), idx)
    #
    #                 if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
    #                     logger.info('E.c=%d.%d | iter=%d/%d | ' %
    #                                 (global_stats['epoch'], d_epoch, idx + 1, len(data_loader)) +
    #                                 'loss=%.4f | time=%.1fs' %
    #                                 (c_train_loss.avg, global_stats['timer'].time()))
    #         else:
    #             for idx, (dat, d_labels) in enumerate(d_data):
    #                 d_train_loss.update(*self.dis_update_with_doc(dat, d_labels))
    #                 self.add_scalar("DIS/Loss/train", d_train_loss.avg, idx)
    #                 self.add_scalar("DIS/Time/train", global_stats['timer'].time(), idx)
    #                 if (idx + 1) % 100 == 0 or (idx + 1) == len(d_data):
    #                     logger.info('E.d=%d.%d | iter=%d/%d | ' %
    #                                 (global_stats['epoch'], d_epoch, idx+1, len(data_loader)) +
    #                                 'loss=%.4f | time=%.1fs' %
    #                                 (d_train_loss.avg, global_stats['timer'].time()))
    #
    #             for idx, (cdat, c_labels) in enumerate(c_data):
    #                 c_train_loss.update(*self.cls_update_with_doc(cdat, c_labels))
    #                 self.add_scalar("CLS/Loss/train", c_train_loss.avg, idx)
    #                 self.add_scalar("CLS/Time/train", global_stats['timer'].time(), idx)
    #                 if (idx + 1) % 100 == 0 or (idx + 1) == len(c_data):
    #                     logger.info('E.c=%d.%d | iter=%d/%d | ' %
    #                                 (global_stats['epoch'], d_epoch, idx + 1, len(data_loader)) +
    #                                 'loss=%.4f | time=%.1fs' %
    #                                 (c_train_loss.avg, global_stats['timer'].time()))
    #
    #         logger.info('E.d=%d.%d | iter=%d/%d | ' %
    #                     (global_stats['epoch'], d_epoch, len(data_loader), len(data_loader)) +
    #                     'loss=%.4f | time=%.1fs, done.' %
    #                     (d_train_loss.avg, epoch_time.time()))
    #         logger.info('E.c=%d.%d | iter=%d/%d | ' %
    #                     (global_stats['epoch'], d_epoch, len(data_loader), len(data_loader)) +
    #                     'loss=%.4f | time=%.1fs, done.' %
    #                     (c_train_loss.avg, epoch_time.time()))
    #
    #         d_train_loss.reset()
    #         c_train_loss.reset()
    #     logger.info(' ' * 30)
    #     logger.info('Training GEN...')
    #     logger.info('+' * 30)
    #     best_acc = global_stats['best_valid']
    #
    #     self.generator.train()
    #     self.discriminator.eval()
    #     self.classifier.eval()
    #     for g_epoch in range(gep):
    #
    #         for idx, ex_with_doc in enumerate(data_loader):
    #
    #             ex = ex_with_doc[0]
    #             batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]
    #
    #             if (idx not in self.HasAnswer_Map):
    #                 HasAnswer_list = []
    #                 for idx_doc in range(0, self.num_docs):  # iterate 100 documents
    #                     # for the idx_doc th position in the 100, get answer for all the questions in the batch.
    #                     HasAnswer = []
    #                     for i in range(batch_size):  # iterate 4 questions
    #                         # HasAnswer.append(has_answer(exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][
    #                         #     idx_doc % len(docs_by_question[ex_id[i]])]["document"]))
    #                         HasAnswer.append(docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"])
    #                     HasAnswer_list.append(HasAnswer)
    #                 self.HasAnswer_Map[idx] = HasAnswer_list
    #             else:
    #                 HasAnswer_list = self.HasAnswer_Map[idx]
    #
    #             weights = []
    #             for idx_doc in range(0, self.num_docs):
    #                 weights.append(1)
    #             weights = torch.Tensor(weights)
    #             idx_random = torch.multinomial(weights, int(self.num_docs))
    #
    #             hasAnswer_list_sample = []
    #             ex_with_doc_sample = []
    #             for idx_doc in idx_random:
    #                 hasAnswer_list_sample.append(HasAnswer_list[idx_doc])
    #                 ex_with_doc_sample.append(ex_with_doc[idx_doc])
    #
    #             # g_train_loss.update(*self.gen_update_with_doc(ex_with_doc_sample))
    #             g_train_loss.update(*self.gen_update_with_doc(ex_with_doc_sample, hasAnswer_list_sample))
    #             if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
    #                 logger.info('E.g=%d.%d | iter=%d/%d | ' % (global_stats['epoch'], g_epoch, idx+1, len(data_loader)) + 'loss=%.4f | time=%.1fs' % (g_train_loss.avg, global_stats['timer'].time()))
    #                 self.add_scalar("GEN/Loss/train", g_train_loss.avg, idx)
    #                 self.add_scalar("GEN/Time/train", global_stats['timer'].time(), idx)
    #                 # acc.eval_unofficial_with_doc(self.args, data_loader, self.generator, global_stats, exs_with_doc,
    #                 #                                  docs_by_question, 'train')
    #             if (idx + 1) % self.args.num_display == 0 or (idx + 1) == len(data_loader):
    #                 with torch.no_grad():
    #                     acc_rslt = accranker.topk_eval_unofficial_with_doc(self.args, self.dev_loader_with_doc, self.generator, global_stats, self.dev_exs_with_doc, self.dev_docs, 'dev', topk=150)
    #                     self.add_scalar("GEN/Accuracy/validate", acc_rslt["exact_match"], idx)
    #
    #                     best_filename = "best_acc.pth.tar"
    #                     if acc_rslt["exact_match"] > best_acc:
    #                         best_acc = acc_rslt["exact_match"]
    #                         is_best_acc = True
    #                         best_filename = 'best_acc.pth.tar'
    #                         global_stats['best_model_file'] = self.args.chp_dir + '/' + best_filename
    #                         global_stats['best_valid'] = best_acc
    #                     else:
    #                         is_best_acc = False
    #                     checkpointname = 'checkpoint.pth.tar'
    #                     self.add_scalar("GEN/Best_Accuracy/validate", best_acc, idx)
    #                     state = {"g_epoch": g_epoch, "best_acc": best_acc,
    #                              'dis_state': self.discriminator.state_dict(), 'dis_optimizer': self.D_optimizer.state_dict(),
    #                              'gen_state': self.generator.state_dict(), 'gen_optimizer': self.G_optimizer.state_dict(), 'word_dict': self.word_dict, 'feature_dict': self.feature_dict}
    #
    #                     self.save_checkpoint(state, is_best_acc, global_stats['best_model_file'], self.args.chp_dir + '/' + checkpointname)
    #
    #                     # logger.info('train: Epoch.g_epoch = %d.%d | iter = %d/%d | ' %
    #                     #             (global_stats['epoch'], g_epoch, len(data_loader), len(data_loader)) +
    #                     #             'loss = %.4f | elapsed time = %.2f (s), done.' %
    #                     #             (g_train_loss.avg, epoch_time.time()))
    #         g_train_loss.reset()
    #     logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' % (global_stats['epoch'], epoch_time.time()))

    def fight(self, global_stats, dep, gep):
        """Run through one epoch of model training with the provided data loader."""
        # Initialize meters + timers
        d_train_loss = dtld.AverageMeter()
        c_train_loss = dtld.AverageMeter()
        g_train_loss = dtld.AverageMeter()
        epoch_time = dtld.Timer()
        exs_with_doc = self.train_exs_with_doc
        docs_by_question = self.train_docs
        data_loader = self.train_loader_with_doc

        # if not global_stats['best_model_file']:
        #     self.init_optimizer()
        # Run one epoch
        logger.info(' ' * 30)
        logger.info('Training DIS...')
        logger.info('+' * 30)
        update_step = 0

        self.discriminator.train()
        self.classifier.train()
        self.generator.eval()


        for idx, ex_with_doc in enumerate(data_loader):
            ex = ex_with_doc[0]
            batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

            if idx not in self.HasAnswer_Map:
                HasAnswer_list = []
                for idx_doc in range(0, self.num_docs):  # iterate 100 documents
                    # for the idx_doc th position in the 100, get answer for all the questions in the batch.
                    HasAnswer = []
                    for i in range(batch_size):  # iterate 4 questions
                        # HasAnswer.append(has_answer(exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][
                        #     idx_doc % len(docs_by_question[ex_id[i]])]["document"]))
                        HasAnswer.append(
                            docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"])
                    HasAnswer_list.append(HasAnswer)
                self.HasAnswer_Map[idx] = HasAnswer_list
            else:
                HasAnswer_list = self.HasAnswer_Map[idx]
            hasAnswer_list_sample = HasAnswer_list
            ex_with_doc_sample = ex_with_doc

            dat, d_labels = self.g_4_d(ex_with_doc_sample, hasAnswer_list_sample, docs_by_question, exs_with_doc)
            if dat is None: # no positive examples at all in this batch, generate next batch.
                logger.info("Ignoring iter %d/%d" % (idx+1, len(data_loader)))
                continue
            cdat, c_labels = self.g_4_c(ex_with_doc_sample, hasAnswer_list_sample, docs_by_question, exs_with_doc)
            if cdat is None:
                continue

            for d_epoch in range(dep):

                d_train_loss.update(*self.dis_update_with_doc(dat, d_labels))
                self.add_scalar("DIS/Loss/train", d_train_loss.avg, idx)
                self.add_scalar("DIS/Time/train", global_stats['timer'].time(), idx)
                update_step = (update_step + 1) % 4
                if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
                    logger.info('E.d=%d.%d | iter=%d/%d | ' %
                                (global_stats['epoch'], d_epoch, idx+1, len(data_loader)) +
                                'loss=%.4f | time=%.1fs' %
                                (d_train_loss.avg, global_stats['timer'].time()))



                c_train_loss.update(*self.cls_update_with_doc(cdat, c_labels))
                self.add_scalar("CLS/Loss/train", c_train_loss.avg, idx)
                self.add_scalar("CLS/Time/train", global_stats['timer'].time(), idx)

                if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
                    logger.info('E.c=%d.%d | iter=%d/%d | ' %
                                (global_stats['epoch'], d_epoch, idx + 1, len(data_loader)) +
                                'loss=%.4f | time=%.1fs' %
                                (c_train_loss.avg, global_stats['timer'].time()))


            # logger.info('E.d=%d.%d | iter=%d/%d | ' %
            #             (global_stats['epoch'], d_epoch, len(data_loader), len(data_loader)) +
            #             'loss=%.4f | time=%.1fs, done.' %
            #             (d_train_loss.avg, epoch_time.time()))
            # logger.info('E.c=%d.%d | iter=%d/%d | ' %
            #             (global_stats['epoch'], d_epoch, len(data_loader), len(data_loader)) +
            #             'loss=%.4f | time=%.1fs, done.' %
            #             (c_train_loss.avg, epoch_time.time()))

            d_train_loss.reset()
            c_train_loss.reset()
        logger.info(' ' * 30)
        logger.info('Training GEN...')
        logger.info('+' * 30)
        best_acc = global_stats['best_valid']

        self.generator.train()
        self.discriminator.eval()
        self.classifier.eval()
        for g_epoch in range(gep):

            for idx, ex_with_doc in enumerate(data_loader):

                ex = ex_with_doc[0]
                batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

                if (idx not in self.HasAnswer_Map):
                    HasAnswer_list = []
                    for idx_doc in range(0, self.num_docs):  # iterate 100 documents
                        # for the idx_doc th position in the 100, get answer for all the questions in the batch.
                        HasAnswer = []
                        for i in range(batch_size):  # iterate 4 questions
                            # HasAnswer.append(has_answer(exs_with_doc[ex_id[i]]['answer'], docs_by_question[ex_id[i]][
                            #     idx_doc % len(docs_by_question[ex_id[i]])]["document"]))
                            HasAnswer.append(docs_by_question[ex_id[i]][idx_doc % len(docs_by_question[ex_id[i]])]["has_answers"])
                        HasAnswer_list.append(HasAnswer)
                    self.HasAnswer_Map[idx] = HasAnswer_list
                else:
                    HasAnswer_list = self.HasAnswer_Map[idx]

                weights = []
                for idx_doc in range(0, self.num_docs):
                    weights.append(1)
                weights = torch.Tensor(weights)
                idx_random = torch.multinomial(weights, int(self.num_docs))

                hasAnswer_list_sample = []
                ex_with_doc_sample = []
                for idx_doc in idx_random:
                    hasAnswer_list_sample.append(HasAnswer_list[idx_doc])
                    ex_with_doc_sample.append(ex_with_doc[idx_doc])

                # g_train_loss.update(*self.gen_update_with_doc(ex_with_doc_sample))
                g_train_loss.update(*self.gen_update_with_doc(ex_with_doc_sample, hasAnswer_list_sample))
                if (idx + 1) % 100 == 0 or (idx + 1) == len(data_loader):
                    logger.info('E.g=%d.%d | iter=%d/%d | ' % (global_stats['epoch'], g_epoch, idx+1, len(data_loader)) + 'loss=%.4f | time=%.1fs' % (g_train_loss.avg, global_stats['timer'].time()))
                    self.add_scalar("GEN/Loss/train", g_train_loss.avg, idx)
                    self.add_scalar("GEN/Time/train", global_stats['timer'].time(), idx)
                    # acc.eval_unofficial_with_doc(self.args, data_loader, self.generator, global_stats, exs_with_doc,
                    #                                  docs_by_question, 'train')
                if (idx + 1) % self.args.num_display == 0 or (idx + 1) == len(data_loader):
                    with torch.no_grad():
                        acc_rslt = accranker.topk_eval_unofficial_with_doc(self.args, self.dev_loader_with_doc, self.generator, global_stats, self.dev_exs_with_doc, self.dev_docs, 'dev', topk=150)
                        self.add_scalar("GEN/Accuracy/validate", acc_rslt["exact_match"], idx)

                        best_filename = "best_acc.pth.tar"
                        if acc_rslt["exact_match"] > best_acc:
                            best_acc = acc_rslt["exact_match"]
                            is_best_acc = True
                            best_filename = 'best_acc.pth.tar'
                            global_stats['best_model_file'] = self.args.chp_dir + '/' + best_filename
                            global_stats['best_valid'] = best_acc
                        else:
                            is_best_acc = False
                        checkpointname = 'checkpoint.pth.tar'
                        self.add_scalar("GEN/Best_Accuracy/validate", best_acc, idx)
                        state = {"g_epoch": g_epoch, "best_acc": best_acc,
                                 'dis_state': self.discriminator.state_dict(), 'dis_optimizer': self.D_optimizer.state_dict(),
                                 'gen_state': self.generator.state_dict(), 'gen_optimizer': self.G_optimizer.state_dict(), 'word_dict': self.word_dict, 'feature_dict': self.feature_dict}

                        self.save_checkpoint(state, is_best_acc, global_stats['best_model_file'], self.args.chp_dir + '/' + checkpointname)

                        # logger.info('train: Epoch.g_epoch = %d.%d | iter = %d/%d | ' %
                        #             (global_stats['epoch'], g_epoch, len(data_loader), len(data_loader)) +
                        #             'loss = %.4f | elapsed time = %.2f (s), done.' %
                        #             (g_train_loss.avg, epoch_time.time()))
            g_train_loss.reset()
        logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' % (global_stats['epoch'], epoch_time.time()))

    def get_super_answers(self, a, s, e):
        c = torch.LongTensor(a).cuda()
        a1 = c[:, 0].unsqueeze(1).repeat(1, s.shape[0])
        a2 = c[:, 1].unsqueeze(1).repeat(1, s.shape[0])
        f1 = s.unsqueeze(0).repeat(c.shape[0], 1)
        f2 = e.unsqueeze(0).repeat(c.shape[0], 1)
        r = (f1 <= a1) * (f2 >= a2) * (f1 >= 0) * (f2 >= 0)
        inds = r.sum(dim=0).nonzero().squeeze()
        return s.gather(0, inds), e.gather(0, inds)

    # @profile
    def g_4_d(self, ex_with_doc_sample, has_answer_list_sample, docs_by_question, exs_with_doc):
        ex = ex_with_doc_sample[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

        # generate the answers for each question in the batch
        if isinstance(self.generator, torch.nn.DataParallel):
            scores_doc = self.generator.module.predict(ex_with_doc_sample)
        else:
            scores_doc = self.generator.predict(ex_with_doc_sample)
        doc_prob = scores_doc.detach().cpu().numpy()

        # Extract positive positive spans of each document
        pos_aq_dict = {}
        pos_lb_dict = {}
        pos_dq_dict = {}
        num_doc_samples = self.args.num_doc_gen
        for idx_doc in range(self.num_docs):
            answers_batch = has_answer_list_sample[idx_doc]
            # for each question-document pair
            for idx_q, (has, pspans) in enumerate(answers_batch):
                if has:
                    line = docs_by_question[ex_id[idx_q]][idx_doc]
                    # doc_len = len(line['document'])

                    pos_dq_dict.setdefault(idx_q, []).append(line)
                    labels = [1]
                    pos_lb_dict.setdefault(idx_q, []).append(labels)
                    if idx_q not in pos_aq_dict:
                        pos_aq_dict[idx_q] = exs_with_doc[ex_id[idx_q]]
        if len(pos_dq_dict) == 0:
            return None, None
        # Generate negative documents and spans
        neg_dq_dict = {}
        neg_aq_dict = {}
        neg_lb_dict = {}
        for idx_q in pos_aq_dict.keys():
            neg_doc_idx = np.random.choice(range(len(ex_with_doc_sample)), size=len(pos_dq_dict[idx_q]), p=doc_prob[idx_q, 0, :])
            if idx_q not in neg_aq_dict:
                neg_aq_dict[idx_q] = exs_with_doc[ex_id[idx_q]]
                # print(neg_aq_dict[idx_q])
            for idx_d_4q in neg_doc_idx:
                neg_dq_dict.setdefault(idx_q, []).append(docs_by_question[ex_id[idx_q]][idx_d_4q])
                neg_labels = [0]
                neg_lb_dict.setdefault(idx_q, []).append(neg_labels)

        # sampling, padding, and merging the generated data
        # for each question, sample 10 positive and 10 negative documents at maximum, pad if not;
        lbls = []

        dt_lst = []
        at_lst = []
        for q_idx, pdq in pos_dq_dict.items():
            le = min(num_doc_samples, len(neg_dq_dict[q_idx]))
            pdq = pdq[0:le]
            ndq = neg_dq_dict[q_idx][0:le]
            lbls.append(pos_lb_dict[q_idx][0:le] + neg_lb_dict[q_idx][0:le])

            dt = []
            for i, dq in enumerate(pdq):
                dqex = {}
                dqex['document'] = dq['document']
                dqex['question'] = dq['question']
                dqex['id'] = ['s' + str(q_idx), i]
                dqex['has_answers'] = dq['has_answers']
                dqex['answers'] = dq['answers']
                dqex['answers_tok'] = dq['answers_tok']
                dt.append(dqex)

            for i, dq in enumerate(ndq):
                dqex = {}
                dqex['document'] = dq['document']
                dqex['question'] = dq['question']
                dqex['id'] = ['s' + str(q_idx), len(pdq) + i]
                dqex['has_answers'] = dq['has_answers']
                dqex['answers'] = dq['answers']
                dqex['answers_tok'] = dq['answers_tok']
                dt.append(dqex)
            dt_lst.append(dt)

            at = {}
            at['answers'] = pos_aq_dict[q_idx]['answers']  # [' '.join(ans) for ans in pos_aq_dict[q_idx]["answer"]]
            at['question'] = pos_aq_dict[q_idx]['question']
            at['answers_tok'] = pos_aq_dict[q_idx]['answers_tok']  # [' '.join(ans) for ans in pos_aq_dict[q_idx]["answer"]]
            at['question_tok'] = pos_aq_dict[q_idx]['question_tok']
            at_lst.append(at)

        tmp_train_docs, tmp_train_questions, tmp_train_docs_inds = dtld.load_list_with_doc(dt_lst, num_doc_samples * 2)
        tmp_train_exs_with_doc = dtld.read_list(at_lst, self.args)

        tmp_train_dataset_with_doc = dtld.ReaderDataset_with_Doc(tmp_train_exs_with_doc, tmp_train_docs, self.word_dict, self.feature_dict, num_doc_samples * 2, single_answer=False)
        tmp_train_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(tmp_train_dataset_with_doc)
        tmp_train_loader_with_doc = torch.utils.data.DataLoader(
            tmp_train_dataset_with_doc,
            batch_size=self.args.BATCH_SIZE,
            sampler=tmp_train_sampler_with_doc,
            num_workers=self.args.data_workers,
            collate_fn=partial(vector.batchify_with_docs, num_docs=num_doc_samples * 2, parag_len=self.parag_len),
            pin_memory=self.args.cuda,
        )
        dat = []
        for t in tmp_train_loader_with_doc:
            dat.extend(t)

        lbls_ind = []
        for i in range(len(lbls)):
            tmp_ind = tmp_train_docs_inds[i]
            # print(i, tmp_ind)
            lbls_q = lbls[i]
            lbls_ind.append([lbls_q[j] for j in tmp_ind])

        return dat, np.transpose(np.array(lbls_ind), axes=(1, 0, 2))

    def g_4_c(self, ex_with_doc_sample, has_answer_list_sample, docs_by_question, exs_with_doc):
        ex = ex_with_doc_sample[0]
        batch_size, question, ex_id = ex[0].size(0), ex[3], ex[-1]

        # Extract positive positive spans of each document
        pos_aq_dict = {}
        pos_lb_dict = {}
        pos_dq_dict = {}
        num_doc_samples = self.args.num_doc_gen

        neg_aq_dict = {}
        neg_lb_dict = {}
        neg_dq_dict = {}

        for idx_doc in range(self.num_docs):
            answers_batch = has_answer_list_sample[idx_doc]
            # for each question-document pair
            for idx_q, (has, pspans) in enumerate(answers_batch):
                line = docs_by_question[ex_id[idx_q]][idx_doc]
                if has:
                    # doc_len = len(line['document'])

                    pos_dq_dict.setdefault(idx_q, []).append(line)
                    labels = [1]
                    pos_lb_dict.setdefault(idx_q, []).append(labels)
                    if idx_q not in pos_aq_dict:
                        pos_aq_dict[idx_q] = exs_with_doc[ex_id[idx_q]]
                else:
                    neg_dq_dict.setdefault(idx_q, []).append(line)
                    neg_labels = [0]
                    neg_lb_dict.setdefault(idx_q, []).append(neg_labels)
                    if idx_q not in neg_aq_dict:
                        neg_aq_dict[idx_q] = exs_with_doc[ex_id[idx_q]]

        # sampling, padding, and merging the generated data
        # for each question, sample 10 positive and 10 negative documents at maximum, pad if not;
        lbls = []
        if len(pos_dq_dict) == 0:
            return None, None

        dt_lst = []
        at_lst = []
        for q_idx, pdq in pos_dq_dict.items():
            if q_idx not in neg_dq_dict:
                print(q_idx, 'not in neg for CLS')
                continue
            le = min(num_doc_samples, len(neg_dq_dict[q_idx]))
            pdq = pdq[0:le]
            ndq = neg_dq_dict[q_idx][0:le]
            lbls.append(pos_lb_dict[q_idx][0:le] + neg_lb_dict[q_idx][0:le])

            dt = []
            for i, dq in enumerate(pdq):
                dqex = {}
                dqex['document'] = dq['document']
                dqex['question'] = dq['question']
                dqex['id'] = ['s' + str(q_idx), i]
                dqex['has_answers'] = dq['has_answers']
                dqex['answers'] = dq['answers']
                dqex['answers_tok'] = dq['answers_tok']
                dt.append(dqex)

            for i, dq in enumerate(ndq):
                dqex = {}
                dqex['document'] = dq['document']
                dqex['question'] = dq['question']
                dqex['id'] = ['s' + str(q_idx), len(pdq) + i]
                dqex['has_answers'] = dq['has_answers']
                dqex['answers'] = dq['answers']
                dqex['answers_tok'] = dq['answers_tok']
                dt.append(dqex)
            dt_lst.append(dt)

            at = {}
            at['answers'] = pos_aq_dict[q_idx]['answers']  # [' '.join(ans) for ans in pos_aq_dict[q_idx]["answer"]]
            at['question'] = pos_aq_dict[q_idx]['question']
            at['answers_tok'] = pos_aq_dict[q_idx]['answers_tok']  # [' '.join(ans) for ans in pos_aq_dict[q_idx]["answer"]]
            at['question_tok'] = pos_aq_dict[q_idx]['question_tok']
            at_lst.append(at)

        tmp_train_docs, tmp_train_questions, tmp_train_docs_inds = dtld.load_list_with_doc(dt_lst, num_doc_samples * 2)
        tmp_train_exs_with_doc = dtld.read_list(at_lst, self.args)

        tmp_train_dataset_with_doc = dtld.ReaderDataset_with_Doc(tmp_train_exs_with_doc, tmp_train_docs, self.word_dict, self.feature_dict, num_doc_samples * 2, single_answer=False)
        tmp_train_sampler_with_doc = torch.utils.data.sampler.SequentialSampler(tmp_train_dataset_with_doc)
        tmp_train_loader_with_doc = torch.utils.data.DataLoader(
            tmp_train_dataset_with_doc,
            batch_size=self.args.BATCH_SIZE,
            sampler=tmp_train_sampler_with_doc,
            num_workers=self.args.data_workers,
            collate_fn=partial(vector.batchify_with_docs, num_docs=num_doc_samples * 2, parag_len=self.parag_len),
            pin_memory=self.args.cuda,
        )
        dat = []
        for t in tmp_train_loader_with_doc:
            dat.extend(t)

        lbls_ind = []
        for i in range(len(lbls)):
            tmp_ind = tmp_train_docs_inds[i]
            # print(i, tmp_ind)
            lbls_q = lbls[i]
            lbls_ind.append([lbls_q[j] for j in tmp_ind])

        return dat, np.transpose(np.array(lbls_ind), axes=(1, 0, 2))

    # def dis_update_with_doc(self, ex_with_doc, ex_spans_s, ex_spans_e, ex_labels, D_parameters, D_optimizer):
    def dis_update_with_doc(self, ex_with_doc, ex_labels):
        """Forward a batch of examples; step the optimizer to update weights."""

        if not self.D_optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.discriminator.train()

        batch_size = ex_with_doc[0][0].size(0)
        if isinstance(self.discriminator, torch.nn.DataParallel):
            scores_doc = self.discriminator.module(ex_with_doc)
        else:
            scores_doc = self.discriminator(ex_with_doc)

        labels_pos = []

        for idx_doc in range(len(ex_with_doc)):
            # soft labels:
            w_0 = random.uniform(0.0, 0.1)
            w_1 = random.uniform(0.9, 1.0)
            if idx_doc > len(ex_with_doc):
                w_doc = w_0
            else:
                w_doc = w_1
            w_ex_labels_idx_doc = [[w_doc * w_0 if lb == 0 else w_doc * w_1 if lb == 1 else lb for lb in lb_idx_doc] for lb_idx_doc in ex_labels[idx_doc]]
            labels_pos.append(w_ex_labels_idx_doc)
        labels_pos_flat = torch.FloatTensor(labels_pos).permute(1, 2, 0).cuda()
        # pred_pos = self.nmlz(torch.clamp(scores_doc * scores_pos, 1.0e-4, 1.0))  # question wise nomalization [1e-8, 1.0]
        pred_pos = torch.clamp(scores_doc, 1.0e-8, 1.0)
        if torch.sum(pred_pos > 1.0) > 0:
            print(pred_pos)
            raise Exception(' some value > 1.0')
        if torch.sum(pred_pos < 0.0) > 0:
            print(pred_pos)
            raise Exception(' some value < 0.0')

        '''
        process the padding tricks, done.
        '''
        padd_weight_flat = (labels_pos_flat >= 0).type(torch.float32).cuda()
        D_loss = torch.mean(F.binary_cross_entropy(pred_pos, labels_pos_flat, weight=padd_weight_flat))

        self.D_optimizer.zero_grad()
        D_loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(D_parameters, self.args.D_GRAD_CLIP)
        # Update parameters
        self.D_optimizer.step()
        return D_loss.item(), batch_size

    def cls_update_with_doc(self, ex_with_doc, ex_labels):
        """Forward a batch of examples; step the optimizer to update weights."""

        if not self.C_optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.classifier.train()

        batch_size = ex_with_doc[0][0].size(0)
        if isinstance(self.classifier, torch.nn.DataParallel):
            scores_doc = self.classifier.module(ex_with_doc)
        else:
            scores_doc = self.classifier(ex_with_doc)

        labels_pos = []

        for idx_doc in range(len(ex_with_doc)):
            # soft labels:
            w_0 = random.uniform(0.0, 0.1)
            w_1 = random.uniform(0.9, 1.0)
            if idx_doc > len(ex_with_doc):
                w_doc = w_0
            else:
                w_doc = w_1
            w_ex_labels_idx_doc = [[w_doc * w_0 if lb == 0 else w_doc * w_1 if lb == 1 else lb for lb in lb_idx_doc] for lb_idx_doc in ex_labels[idx_doc]]
            labels_pos.append(w_ex_labels_idx_doc)
        labels_pos_flat = torch.FloatTensor(labels_pos).permute(1, 2, 0).cuda()
        # pred_pos = self.nmlz(torch.clamp(scores_doc * scores_pos, 1.0e-4, 1.0))  # question wise nomalization [1e-8, 1.0]
        pred_pos = torch.clamp(scores_doc, 1.0e-8, 1.0)
        if torch.sum(pred_pos > 1.0) > 0:
            print(pred_pos)
            raise Exception(' some value > 1.0')
        if torch.sum(pred_pos < 0.0) > 0:
            print(pred_pos)
            raise Exception(' some value < 0.0')

        '''
        process the padding tricks, done.
        '''
        padd_weight_flat = (labels_pos_flat >= 0).type(torch.float32).cuda()
        C_loss = torch.mean(F.binary_cross_entropy(pred_pos, labels_pos_flat, weight=padd_weight_flat))

        self.C_optimizer.zero_grad()
        C_loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(D_parameters, self.args.D_GRAD_CLIP)
        # Update parameters
        self.C_optimizer.step()
        return C_loss.item(), batch_size

    def nmlz(self, scrs):
        min_scores = torch.min(torch.min(scrs, dim=2)[0], dim=1)[0]
        range_scores = torch.max(torch.max(scrs, dim=2)[0], dim=1)[0] - min_scores
        range_scores = (range_scores + (range_scores <= 1.0e-4).type(torch.float32).cuda()).unsqueeze(dim=1).unsqueeze(dim=2)  # in case divider is 0.0 -> 1.0
        denom_scores = scrs - min_scores.unsqueeze(dim=1).unsqueeze(dim=2)
        # denom_scores = denom_scores + (denom_scores == 0.0).to(self.args.device, dtype=torch.float32) * 1e-8  # minimum prob is 1e-8 > 0.0
        return denom_scores / range_scores

    # @profile
    # def gen_update_with_doc(self, ex_with_doc, G_parameters, G_optimizer):
    def gen_update_with_doc_old(self, ex_with_doc):
        """Forward a batch of examples; step the optimizer to update weights."""

        if not self.G_optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.generator.train()

        batch_size = ex_with_doc[0][0].size(0)
        scores_doc = self.generator(ex_with_doc)

        # # ================== sampling for reward =========================================
        choose_inputs, choose_docs_inds, choose_docs_IS = self.important_sampling(scores_doc, ex_with_doc)
        choose_reward_docs = self.discriminator.get_reward(choose_inputs)
        # ================= end of reward computing =========================================

        gan_scores_doc = torch.gather(scores_doc, 2, choose_docs_inds.permute(1, 2, 0)).clamp(min=1e-8)
        G_loss = -torch.mean(torch.log(gan_scores_doc) * choose_reward_docs * choose_docs_IS.permute(1, 2, 0))

        self.G_optimizer.zero_grad()
        G_loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(G_parameters, self.args.G_GRAD_CLIP)
        # Update parameters
        self.G_optimizer.step()
        return G_loss.item(), batch_size

    def gen_update_with_doc(self, ex_with_doc, hasAnswer_list):
        """Forward a batch of examples; step the optimizer to update weights."""

        if not self.G_optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.generator.train()

        batch_size = ex_with_doc[0][0].size(0)
        num_docs = len(ex_with_doc)
        scores_doc = self.generator(ex_with_doc)

        # # ================== sampling for reward =========================================
        choose_inputs, choose_docs_inds, choose_docs_IS = self.important_sampling(scores_doc, ex_with_doc)
        if isinstance(self.discriminator, torch.nn.DataParallel):
            choose_reward_docs = self.discriminator.module.get_reward(choose_inputs)
        else:
            choose_reward_docs = self.discriminator.get_reward(choose_inputs)
        if isinstance(self.classifier, torch.nn.DataParallel):
            c_choose_reward_docs = self.classifier.module.get_reward(choose_inputs)
        else:
            c_choose_reward_docs = self.classifier.get_reward(choose_inputs)
        # ================= end of reward computing =========================================

        gan_scores_doc = torch.gather(scores_doc, 2, choose_docs_inds.permute(1, 2, 0)).clamp(min=1e-8)
        G_loss = -torch.mean(torch.log(gan_scores_doc) * (choose_reward_docs + 0.25*c_choose_reward_docs) * choose_docs_IS.permute(1, 2, 0))

        num_docs_with_ans_per_question = [0.0] * len(hasAnswer_list[0])
        for ans_batch in hasAnswer_list:
            for i, q_ans in enumerate(ans_batch):
                if q_ans[0]:
                    num_docs_with_ans_per_question[i] += 1

        flag = False
        reg_loss = Variable(torch.FloatTensor([0.0]).cuda())
        for i in range(batch_size):
            num_answer = num_docs_with_ans_per_question[i]
            for idx_doc in range(num_docs):
                if (hasAnswer_list[idx_doc][i][0]):
                    flag = True
                    if (scores_doc[i][0][idx_doc].data.cpu().numpy() > 1e-16):
                        reg_loss += Variable(torch.FloatTensor([1.0 / num_answer]).cuda()) * (
                                    -(scores_doc[i][0][idx_doc] + 1e-16).log() + Variable(
                                torch.FloatTensor([1.0 / num_answer]).cuda().log()))

        if flag:
            G_loss += reg_loss[0]

        self.G_optimizer.zero_grad()
        G_loss.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(G_parameters, self.args.G_GRAD_CLIP)
        # Update parameters
        self.G_optimizer.step()
        return G_loss.item(), batch_size

    # @profile
    def important_sampling(self, scores_docs, ex_with_docs):
        lambda_docs_prob = torch.ones(scores_docs.shape).cuda()
        lambda_docs_prob = F.softmax(lambda_docs_prob, dim=2).view(scores_docs.shape)
        prob_docs_IS = scores_docs * (1.0 - self.args.LAMBDA) + lambda_docs_prob * self.args.LAMBDA

        num_q, _, num_doc = scores_docs.shape
        num_to_doc = num_doc * self.args.num_sample_doc
        choose_docs_inds = torch.zeros(num_to_doc, num_q, 1, dtype=torch.long).cuda()
        choose_docs_IS = torch.zeros(num_to_doc, num_q, 1, dtype=torch.float).cuda()
        for i in range(num_q):
            p_i = prob_docs_IS[i, :, :].squeeze().detach().cpu().numpy()  # tensor to numpy
            choose_docs_inds_q = np.random.choice(np.arange(num_doc), [num_to_doc], p=p_i)
            choose_docs_inds[:, i, 0] = torch.from_numpy(choose_docs_inds_q).long().cuda()

            np_scores_docs = scores_docs[i, :, :].squeeze().detach().cpu().numpy()
            choose_docs_IS_q = np_scores_docs[choose_docs_inds_q] / p_i[choose_docs_inds_q]
            choose_docs_IS[:, i, 0] = torch.from_numpy(choose_docs_IS_q).float().cuda()

        choose_inputs = []
        for d_i in range(num_to_doc):
            ex_d_i_seg0 = None
            ex_d_i_seg1 = None
            ex_d_i_seg2 = None
            ex_d_i_seg3 = None
            ex_d_i_seg4 = None
            ex_d_i_seg5 = None

            # re-padding, get the maximum length for current position of the current batch
            max_len_doc_i = 0
            max_len_que_i = 0
            len_docs_i = []
            len_ques_i = []
            for j in range(num_q):
                i = choose_docs_inds[d_i][j][0]
                # mask_doc_i_j = ex_with_docs[i][2][j, :].detach().cpu().numpy()
                # len_doc_i_j = sum(mask_doc_i_j < 1)
                # len_docs_i.append(len_doc_i_j.tolist())
                len_doc_i_j = (ex_with_docs[i][2][j, :] < 1).sum()
                len_docs_i.append(len_doc_i_j)
                if max_len_doc_i < len_doc_i_j:
                    max_len_doc_i = len_doc_i_j
                # mask_que_i_j = ex_with_docs[i][4][j, :].detach().cpu().numpy()
                # len_que_i_j = sum(mask_que_i_j < 1)
                # len_ques_i.append(len_que_i_j.tolist())
                len_que_i_j = (ex_with_docs[i][4][j, :] < 1).sum()
                len_ques_i.append(len_que_i_j)
                if max_len_que_i < len_que_i_j:
                    max_len_que_i = len_que_i_j
            # re-padding, refilling the vector with new padding.
            for j in range(num_q):
                i = choose_docs_inds[d_i][j][0]

                len_doc_ij = len_docs_i[j]
                len_que_ij = len_ques_i[j]
                pad_len_doc_ij = max_len_doc_i - len_doc_ij
                pad_len_que_ij = max_len_que_i - len_que_ij

                # seg0
                raw_ex_d_i_seg0_j = ex_with_docs[i][0][j, :].narrow(0, 0, len_doc_ij)
                pad_ex_d_i_seg0_j = torch.zeros(pad_len_doc_ij, dtype=torch.int64)
                ex_d_i_seg0_j = torch.cat((raw_ex_d_i_seg0_j, pad_ex_d_i_seg0_j)).unsqueeze(dim=0)
                # ex_d_i_seg0_j = ex_with_docs[i][0][j, :].unsqueeze(dim=0)

                # seg1
                raw_ex_d_i_seg1_j = ex_with_docs[i][1][j, :, :].narrow(0, 0, len_doc_ij)
                pad_ex_d_i_seg1_j = torch.zeros([pad_len_doc_ij, raw_ex_d_i_seg1_j.shape[1]])
                ex_d_i_seg1_j = torch.cat((raw_ex_d_i_seg1_j, pad_ex_d_i_seg1_j)).unsqueeze(dim=0)
                # ex_d_i_seg1_j = ex_with_docs[i][1][j, :, :].unsqueeze(dim=0)

                # seg2
                raw_ex_d_i_seg2_j = ex_with_docs[i][2][j, :].narrow(0, 0, len_doc_ij)
                pad_ex_d_i_seg2_j = torch.ones(pad_len_doc_ij, dtype=torch.uint8)
                ex_d_i_seg2_j = torch.cat((raw_ex_d_i_seg2_j, pad_ex_d_i_seg2_j)).unsqueeze(dim=0)
                # ex_d_i_seg2_j = ex_with_docs[i][2][j, :].unsqueeze(dim=0)

                # seg3
                raw_ex_d_i_seg3_j = ex_with_docs[i][3][j, :].narrow(0, 0, len_que_ij)
                pad_ex_d_i_seg3_j = torch.zeros(pad_len_que_ij, dtype=torch.int64)
                ex_d_i_seg3_j = torch.cat((raw_ex_d_i_seg3_j, pad_ex_d_i_seg3_j)).unsqueeze(dim=0)
                # ex_d_i_seg3_j = ex_with_docs[i][3][j, :].unsqueeze(dim=0)

                # seg 4
                raw_ex_d_i_seg4_j = ex_with_docs[i][4][j, :].narrow(0, 0, len_que_ij)
                pad_ex_d_i_seg4_j = torch.ones(pad_len_que_ij, dtype=torch.uint8)
                ex_d_i_seg4_j = torch.cat((raw_ex_d_i_seg4_j, pad_ex_d_i_seg4_j)).unsqueeze(dim=0)
                # ex_d_i_seg4_j = ex_with_docs[i][4][j, :].unsqueeze(dim=0)

                # seg 5
                ex_d_i_seg5_j = [ex_with_docs[i][5][j]]

                if ex_d_i_seg0 is None:
                    ex_d_i_seg0 = ex_d_i_seg0_j
                    ex_d_i_seg1 = ex_d_i_seg1_j
                    ex_d_i_seg2 = ex_d_i_seg2_j
                    ex_d_i_seg3 = ex_d_i_seg3_j
                    ex_d_i_seg4 = ex_d_i_seg4_j
                    ex_d_i_seg5 = ex_d_i_seg5_j
                else:
                    ex_d_i_seg0 = torch.cat((ex_d_i_seg0, ex_d_i_seg0_j), 0)
                    ex_d_i_seg1 = torch.cat((ex_d_i_seg1, ex_d_i_seg1_j), 0)
                    ex_d_i_seg2 = torch.cat((ex_d_i_seg2, ex_d_i_seg2_j), 0)
                    ex_d_i_seg3 = torch.cat((ex_d_i_seg3, ex_d_i_seg3_j), 0)
                    ex_d_i_seg4 = torch.cat((ex_d_i_seg4, ex_d_i_seg4_j), 0)
                    ex_d_i_seg5.extend(ex_d_i_seg5_j)
            choose_inputs.append((ex_d_i_seg0, ex_d_i_seg1, ex_d_i_seg2, ex_d_i_seg3, ex_d_i_seg4, ex_d_i_seg5))

        return choose_inputs, choose_docs_inds, choose_docs_IS

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------
    def save_checkpoint(self, state, is_best, best_filename, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_filename)

    def load_checkpoint_for_resume(self, filename):
        logger.info('=> loading checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        # start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        # best_f1 = checkpoint['best_f1']
        self.word_dict = checkpoint['word_dict']
        self.feature_dict = checkpoint['feature_dict']
        self.args.num_features = len(self.feature_dict)
        self.args.vocab_size = len(self.word_dict)
        self.generator = GEN(self.args).cuda()
        self.discriminator = DIS(self.args).cuda()
        self.discriminator.load_state_dict(checkpoint['dis_state'])
        self.generator.load_state_dict(checkpoint['gen_state'])
        self.init_optimizer(checkpoint['dis_optimizer'], checkpoint['gen_optimizer'])
        logger.info("=> checkpoint loaded, current best EM %.2f" % best_acc)
        return best_acc

    def add_scalar(self, tag, value, step):
        """Log a scalar variable."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
            # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            # self.writer.add_summary(summary, step)
            # self.writer.flush()

    def parallelize(self):
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()
        self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        self.generator = torch.nn.DataParallel(self.generator).cuda()
