import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric
import math
import numpy as np

INF = 1e20
VERY_SMALL_NUMBER = 1e-12


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1],
                  X.shape[2])  
    X = X.permute(0, 2, 1, 3)  
    return X.reshape(X.shape[0], X.shape[1], -1) 


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers,
                 dropout=0.0, batchnorm=True, activation=True):
        super(MLP, self).__init__()
        modules = OrderedDict()
        previous_size = input_size
        for index, hidden_layer in enumerate(hidden_layers):
            modules[f"dense{index}"] = nn.Linear(previous_size, hidden_layer)
            if batchnorm:
                modules[f"batchnorm{index}"] = nn.BatchNorm1d(hidden_layer)
            if activation:
                modules[f"activation{index}"] = nn.ReLU()
            if dropout:
                modules[f"dropout{index}"] = nn.Dropout(dropout)
            previous_size = hidden_layer
        self.mlp = nn.Sequential(modules)

    def forward(self, x):
        return self.mlp(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x, x_len):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
                num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

from torch.nn.init import xavier_uniform_
def get_key_padding_mask(tokens, padding_token=0):
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == padding_token] = -float('inf')
    return key_padding_mask

class TransformerEncoderwithPE(nn.Module):
    def __init__(self, d_model, num_encoder_layers, dropout):
        super(TransformerEncoderwithPE, self).__init__()
        self.d_model = d_model
        self.positional_embedding = PositionalEncoding(d_model, dropout)
        transfomerlayer = nn.TransformerEncoderLayer(d_model, 8, dim_feedforward=d_model * 4, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(transfomerlayer, num_encoder_layers, encoder_norm)

    def forward(self, x, valid_lens):
        lens = valid_lens.unsqueeze(1)
        key_padding_mask = torch.arange(x.size(1)).expand(x.size(0), x.size(1)).to(x.device) < lens
        key_padding_mask = ~key_padding_mask
        x = self.positional_embedding(x)
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)
        return x

def extract_last_valid_output(output, sequence_lengths):
    max_len = output.size(1)
    batch_size = output.size(0)
    hidden_size = output.size(2)
    device = output.device
    last_valid_index = sequence_lengths - 1
    output = output.reshape(batch_size * max_len, hidden_size)
    indices = last_valid_index + max_len * torch.arange(batch_size).to(device)
    last_valid_output = output[indices]
    last_valid_output = last_valid_output.reshape(batch_size, hidden_size)

    return last_valid_output

class AttentionNetPooling(nn.Module):
    def __init__(self, in_dim):
        super(AttentionNetPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float() 
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1) 
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class Seq(nn.Module):
    def __init__(self, cat_nums, seq_embedding_size, lstm_hidden_size, dropout, reduction=False, use_attention=False,
                 LayerNorm=True, encoder_type='lstm', num_lstm_layers=2):
        super(Seq, self).__init__()
        self.reduction = reduction
        self.lstm_hidden_size = lstm_hidden_size
        self.use_attention = use_attention
        self.LayerNorm = LayerNorm
        self.embeddings = OrderedDict()
        self.num_lstm_layers = num_lstm_layers
        self.cat_nums = cat_nums
        self.embeddings['hist_activity'] = nn.Embedding((cat_nums['hist_activity'] - 3) * 24 + 25, seq_embedding_size,
                                                        padding_idx=0)
        self.add_module(f"embedding:{'hist_activity'}", self.embeddings['hist_activity'])
        self.encoder_type = encoder_type
        if self.encoder_type == "lstm":
            self.encoder = nn.LSTM(input_size=seq_embedding_size, hidden_size=lstm_hidden_size,
                                   num_layers=num_lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.encoder_type == "gru":
            self.encoder = nn.GRU(input_size=seq_embedding_size, hidden_size=lstm_hidden_size,
                                  num_layers=num_lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.encoder_type == "rnn":
            self.encoder = nn.RNN(input_size=seq_embedding_size, hidden_size=lstm_hidden_size,
                                  num_layers=num_lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.encoder_type == "transformer":
            self.encoder = TransformerEncoderwithPE(d_model=lstm_hidden_size, num_encoder_layers=2, dropout=dropout)
        else:
            raise NotImplementedError
        if LayerNorm:
            self.ln = nn.LayerNorm(lstm_hidden_size)
        if use_attention or self.reduction == 'selfattention+avgpooling' or self.reduction == 'lastpositionattention':
            self.SA = MultiHeadAttention(lstm_hidden_size, lstm_hidden_size, lstm_hidden_size, lstm_hidden_size, 8,
                                         dropout)
        if self.reduction == 'clsattention':
            q_t = np.random.normal(loc=0.0, scale=1, size=(1, self.lstm_hidden_size))
            self.q = nn.Parameter(torch.from_numpy(q_t).float())
            self.MHA = MultiHeadAttention(lstm_hidden_size, lstm_hidden_size, lstm_hidden_size, lstm_hidden_size, 8,
                                         dropout)
        if self.reduction == 'attentionnetpooling':
            self.ANP = AttentionNetPooling(self.lstm_hidden_size)

    def forward(self, x):
        seqs_length = x['max_len']
        output = self.embeddings['hist_activity'](x['hist_activity'])
        device = output.device
        if self.encoder_type == "lstm" or self.encoder_type == "rnn" or self.encoder_type == "gru":
            packed_out = pack_padded_sequence(
                output,
                lengths=seqs_length.cpu(),
                batch_first=True,
                enforce_sorted=False)
            output, (h_n, c_n) = self.encoder(packed_out)
            output, _ = pad_packed_sequence(output, batch_first=True)
        elif self.encoder_type == "transformer":
            output = self.encoder(output, valid_lens=seqs_length)
            h_n = [None, None]
        else:
            raise NotImplementedError
        if self.LayerNorm:
            output = self.ln(output)
        if self.reduction:
            if self.reduction == True or self.reduction == 'avgpooling':
                lens = seqs_length.unsqueeze(1)
                padding_mask = torch.arange(output.size(1)).expand(output.size(0), output.size(1)).to(
                    device) < lens
                padding_mask = padding_mask.unsqueeze(-1).float()
                output = torch.sum(output * padding_mask, dim=1) / lens
            elif self.reduction == 'selfattention+avgpooling':
                output = self.SA(output, output, output, seqs_length)
                lens = seqs_length.unsqueeze(1)
                padding_mask = torch.arange(output.size(1)).expand(output.size(0), output.size(1)).to(
                    device) < lens
                padding_mask = padding_mask.unsqueeze(-1).float()
                output = torch.sum(output * padding_mask, dim=1) / lens
            elif self.reduction == 'lastpositionattention':
                query = extract_last_valid_output(output, seqs_length).unsqueeze(1)
                output = self.SA(query, output, output, seqs_length).squeeze()
            elif self.reduction == 'clsattention':
                query = self.q.repeat(output.size(0), 1)
                query = query.unsqueeze(1) 
                lens = seqs_length.unsqueeze(1)
                attention_mask = torch.arange(output.size(1)).expand(output.size(0), output.size(1)).to(
                    device) < lens
                output = self.MHA(query, output, output, seqs_length)
                output = output.squeeze()
            elif self.reduction == 'attentionnetpooling':
                lens = seqs_length.unsqueeze(1)
                attention_mask = torch.arange(output.size(1)).expand(output.size(0), output.size(1)).to(device) < lens
                output = self.ANP(output, attention_mask)
            else:
                raise NotImplementedError
        return output, h_n[-1]

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


def CreateSeq(cat_nums, seq_embedding_size=128, lstm_hidden_size=128, dropout=0.3, reduction=False, use_attention=False,
              LayerNorm=True, encoder_type='lstm', num_lstm_layers=2):
    return Seq(cat_nums, seq_embedding_size, lstm_hidden_size, dropout, reduction, use_attention, LayerNorm,
               encoder_type, num_lstm_layers)


class SeqwithClassifier(nn.Module):
    def __init__(self, cat_nums, seq_embedding_size, lstm_hidden_size, dropout, reduction=False, use_attention=False,
                 LayerNorm=True, encoder_type='lstm', num_lstm_layers=2, num_class=2, embedding_hook=False):
        super(SeqwithClassifier, self).__init__()
        self.embedding_hook = embedding_hook
        self.seq = Seq(cat_nums, seq_embedding_size, lstm_hidden_size, dropout, reduction, use_attention, LayerNorm,
                       encoder_type, num_lstm_layers)
        self.final_layer = nn.Linear(lstm_hidden_size, num_class)
        self.final_layer.apply(init_weights)

    def forward(self, x):
        out = self.seq(x)[0]
        logits = self.final_layer(out.reshape(-1, out.shape[-1]))
        if self.embedding_hook == True:
            return logits, out.reshape(-1, out.shape[-1])
        return logits


def CreateSeqwithClassifier(cat_nums, seq_embedding_size=128, lstm_hidden_size=128, dropout=0.3, reduction=False,
                            use_attention=False, LayerNorm=True, encoder_type='lstm', num_lstm_layers=2, num_class=2, embedding_hook=False):
    return SeqwithClassifier(cat_nums, seq_embedding_size, lstm_hidden_size, dropout, reduction, use_attention,
                             LayerNorm, encoder_type, num_lstm_layers, num_class, embedding_hook=embedding_hook)

class dynamicGCN(torch.nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(dynamicGCN, self).__init__()
        self.conv1 = GCNConv(feat_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size) 
        self.dropout = dropout

    def forward(self, x, adj):
        edge_index, edge_weights = torch_geometric.utils.dense_to_sparse(adj)
        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weights)
        x = F.relu(x) 
        return x

class dynamicGAT(torch.nn.Module):
    def __init__(self, feat_size, hidden_size, dropout, num_heads=8):
        super(dynamicGAT, self).__init__()
        self.conv1 = GATConv(feat_size, hidden_size//num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_size, hidden_size//num_heads, heads=num_heads)
        self.dropout = dropout

    def forward(self, x, adj):
        edge_index, edge_weights = torch_geometric.utils.dense_to_sparse(adj)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)  
        return x



class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=None, num_pers=16, metric_type='cosine'):
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon
        self.metric_type = metric_type
        if metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size) 
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
        elif metric_type == 'cosine':
            pass
        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))
        print('use {}'.format(metric_type))

    def forward(self, context):
        if self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0
        elif self.metric_type == 'cosine':
            context_norm = torch.nn.functional.normalize(context, p=2, dim=-1)
            attention = torch.bmm(context_norm, context_norm.transpose(-1, -2))
            markoff_value = 0
        if self.epsilon is not None: 
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        return attention

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

def batch_diagflat(tensor):
    device = tensor.device
    tensor = tensor.unsqueeze(1)
    identity = torch.eye(tensor.size(-1)).to(device).unsqueeze(0)
    result = tensor * identity
    return result

def batch_trace(tensor):
    return tensor.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def add_batch_graph_loss(out_adj, features, smoothness_ratio=0.2, degree_ratio=0, sparsity_ratio=0.1):
    device = out_adj.device
    topk = out_adj.shape[-1]
    batch = out_adj.shape[0]
    graph_loss = 0
    L = batch_diagflat(torch.sum(out_adj, -1)) - out_adj 
    graph_loss += smoothness_ratio * (batch_trace(torch.bmm(features.transpose(-1, -2), torch.bmm(L, features))) / torch.tensor([topk * topk]).repeat(batch).to(device)).mean()
    ones_vec = torch.ones(out_adj.size(-1)).repeat(batch, 1).to(device)
    graph_loss += -degree_ratio * (torch.bmm(ones_vec.unsqueeze(1), torch.log(torch.bmm(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze() / topk).mean()
    graph_loss += sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / (topk * topk) / batch
    return graph_loss

class InsiderClassifier(nn.Module):
    def __init__(self, num_features, cat_features, seq_features, cat_nums, cat_embedding_size, seq_embedding_size,
                 lstm_hidden_size, dropout, reduction=True, use_attention=False, LayerNorm=True, encoder_type='lstm',
                 num_lstm_layers=2, mlp_hidden_layers=[],
                 pooling_mode='origin', epsilon=0, num_pers=4, graph_metric_type='weighted_cosine',
                 topk=15, num_class=2, add_graph_regularization=False, gnn='GCN', embedding_hook=False):
        super(InsiderClassifier, self).__init__()
        self.topk = topk
        self.num_class = num_class
        self.add_graph_regularization = add_graph_regularization
        self.embedding_hook = embedding_hook
        self.seq = Seq(cat_nums, seq_embedding_size, lstm_hidden_size, dropout, reduction, use_attention, LayerNorm,
                       encoder_type, num_lstm_layers)
        self.drop = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(lstm_hidden_size, lstm_hidden_size,
                                          epsilon=epsilon,
                                          num_pers=num_pers,
                                          metric_type=graph_metric_type)
        if gnn == 'GCN':
            self.gcn = dynamicGCN(feat_size=lstm_hidden_size, hidden_size=lstm_hidden_size, dropout=dropout)
        elif gnn == 'GAT':
            self.gcn = dynamicGAT(feat_size=lstm_hidden_size, hidden_size=lstm_hidden_size, dropout=dropout)
        self.cls = torch.nn.Linear(in_features=lstm_hidden_size, out_features=num_class)

    def forward(self, x):
        features = self.seq(x)[0]
        adj = self.graph_learner(features.reshape(-1, self.topk, features.shape[-1]))
        X_hat = self.gcn(features, adj)
        y_hat = self.cls(X_hat[::self.topk, :])  
        if self.embedding_hook == True:
            return y_hat, X_hat[::self.topk, :], adj
        if self.add_graph_regularization == True:
            grl = add_batch_graph_loss(adj, features.reshape(-1, self.topk, features.shape[-1]))
            return y_hat, grl
        return y_hat

def CreateInsiderClassifier(num_features, cat_features, seq_features, cat_nums, cat_embedding_size=16,
                            seq_embedding_size=128, lstm_hidden_size=128, dropout=0.3, reduction=True,
                            use_attention=False, LayerNorm=True, encoder_type='lstm', num_lstm_layers=2,
                            mlp_hidden_layers=[], pooling_mode='origin', epsilon=0, num_pers=4,
                            graph_metric_type='weighted_cosine', topk=15, num_class=2, add_graph_regularization=False, gnn='GCN', embedding_hook=False):
    return InsiderClassifier(num_features, cat_features, seq_features, cat_nums, cat_embedding_size, seq_embedding_size,
                             lstm_hidden_size, dropout, reduction, use_attention, LayerNorm, encoder_type,
                             num_lstm_layers, mlp_hidden_layers, pooling_mode, epsilon, num_pers, graph_metric_type,
                             topk, num_class=num_class, add_graph_regularization=add_graph_regularization, gnn=gnn, embedding_hook=embedding_hook)
