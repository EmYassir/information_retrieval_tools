
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import layers
import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------
class GEN(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(GEN, self).__init__()
        # Store config
        self.args = args
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # self.args.dropout_emb = 0
        my_layer_num = self.args.doc_layers
        # my_layer_num = 1
        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=my_layer_num,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=nn.LSTM,
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=my_layer_num,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=nn.LSTM,
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size

        if args.concat_rnn_layers:
            doc_hidden_size *= my_layer_num
            question_hidden_size *= my_layer_num

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end

        self.ans_attn = layers.BilinearSeqAttn1(
            doc_hidden_size,
            question_hidden_size,
        )
        self.ans_attn1 = layers.BilinearSeqAttn1(
            doc_hidden_size,
            question_hidden_size,
        )
        # self.ans_attn2 = layers.BilinearSeqAttn1(
        #     doc_hidden_size,
        #     question_hidden_size,
        # )

    def forward(self, ex_with_doc):
        scores_doc = self.predict_score_doc(ex_with_doc)
        return scores_doc

    def predict_score_doc(self, ex_with_doc):
        batch_size = ex_with_doc[0][0].shape[0]
        num_docs = len(ex_with_doc)
        scores_doc = torch.zeros(batch_size, 1, num_docs).cuda()
        for idx_doc in range(num_docs):
            ex = ex_with_doc[idx_doc]
            inputs = [e if e is None else Variable(e.cuda()) for e in ex[:5]]
            scores_doc[:, :, idx_doc] = self.predict_score(*inputs)
        return torch.softmax(scores_doc, dim=2)

    # @profile
    def predict_score(self, x1, x1_f, x1_mask, x2, x2_mask):

        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)  # batch * len1 * him

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)  # batch * len2 * him

        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        scores_s_all = self.ans_attn(doc_hiddens, question_hidden, x1_mask)  # .sigmoid()
        scores_e_all = self.ans_attn1(doc_hiddens, question_hidden, x1_mask)  # .sigmoid()
        scores_s = torch.unsqueeze(torch.max(scores_s_all, 1)[0], 1)
        scores_e = torch.unsqueeze(torch.max(scores_e_all, 1)[0], 1)
        scores_doc = (scores_s + scores_e) / 2.0 / self.args.DOC_TEMPERATURE

        return scores_doc

    def predict(self, ex_with_doc):
        self.eval()
        return self.predict_score_doc(ex_with_doc)
