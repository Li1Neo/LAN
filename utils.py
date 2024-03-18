import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import sklearn
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import faiss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import logging
from torch.utils.data import DataLoader

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def seed_everything(seed=42):
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

class Df2Dataset(Dataset):
    def __init__(self, dfdata, num_features, cat_features,
                 seq_features, encoders, label_col=["label"]):
        self.dfdata = dfdata
        self.num_features = num_features
        self.cat_features = cat_features
        self.seq_features = seq_features
        self.encoders = encoders
        self.label_col = label_col
        self.size = len(self.dfdata)

    def __len__(self):
        return self.size

    @staticmethod
    def pad_sequence(sequence, max_length):
        # zero is special index for padding
        padded_seq = np.zeros(max_length, np.int64)
        padded_seq[0: sequence.shape[0]] = sequence
        return padded_seq

    def __getitem__(self, idx):
        record = OrderedDict()
        for col in self.num_features:
            record[col] = self.dfdata[col].iloc[idx].astype(np.float32)

        for col in self.cat_features:
            record[col] = self.dfdata[col].iloc[idx].astype(np.int64)

        for col in self.seq_features:
            seq = self.dfdata[col].iloc[idx]
            max_length = self.encoders[col].max_length()
            record[col] = Df2Dataset.pad_sequence(seq, max_length)
        record['max_len'] = np.int64(seq.shape[0])
        record['idx'] = np.int64(idx)
        if self.label_col is not None:
            for col in self.label_col:
                if type(self.dfdata[col].iloc[idx]) == np.ndarray:
                    record[col] = Df2Dataset.pad_sequence(self.dfdata[col].iloc[idx], max_length)
                else: # int
                    record[col] = self.dfdata[col].iloc[idx].astype(np.float32)
        return record

    def get_num_batches(self, batch_size):
        return np.ceil(self.size / batch_size)


def bestthreshold_with_ROC(label, predict):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, predict)
    distances = np.sqrt(fpr**2 + (tpr-1)**2)
    idx = np.argsort(distances)[0]
    optimal_threshold = thresholds[idx]
    return fpr[idx], tpr[idx], optimal_threshold

def cal_f1_score(label, predict, threshold):
    y_true = label
    y_pred = (predict>threshold).astype(int)
    return f1_score(y_true, y_pred)


def plot_roc(root, label, predict):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, predict)
    fig = plt.figure(figsize=(8, 6))
    plt.title('ROC Curve', fontsize=15)
    plt.plot(fpr, tpr, 'b', label='ROC', linewidth=3)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR', fontsize=15)
    plt.xlabel('FPR', fontsize=15)
    fig.savefig(os.path.join(root, 'ROC.png'))

def build_vector_pool(args, logger, model, sequential_dl_train, device, rmodel=None): 
    assert args.classification_level == 'action'
    if rmodel:
        rmodel.eval()
    else:
        model.eval()
    logger.info('generating vector...')
    vector_list = []
    t = tqdm(sequential_dl_train, desc='generating vector')
    has_al = False
    for idx, batch in enumerate(t):
        batch = collate_fn(batch, has_al = has_al)
        for k, v in batch.items():
            batch[k] = batch[k].to(device)
        if rmodel:
            out = rmodel(batch)[0]
        else:
            out = model.seq(batch)[0]
        out = out.reshape(-1, out.shape[-1])
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        out = out.detach().cpu().numpy()
        vector_list.append(out)
    xb = np.vstack(vector_list)
    if rmodel:
        dim = rmodel.lstm_hidden_size
    else:
        dim = model.seq.lstm_hidden_size
    measure = faiss.METRIC_INNER_PRODUCT 
    index = faiss.IndexHNSWFlat(dim, 64, measure)
    if index.is_trained == False:
        index.train(xb)
    index.add(xb)  
    faiss.write_index(index, os.path.join(args.root, '{}_{}_{}_{}_{}_level_trained_{}_hard_negative_weight_1e{}_sample_rate_{}topk.index'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight, args.sample_rate, args.topk)))
    logger.info('index is written to {}'.format(os.path.join(args.root, '{}_{}_{}_{}_{}_level_trained_{}_hard_negative_weight_1e{}_sample_rate_{}topk.index'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight, args.sample_rate, args.topk))))
    return index

def pad_sequence(sequence, max_length):
    if isinstance(sequence, list) or isinstance(sequence, np.ndarray):
        padded_seq = torch.zeros([len(sequence), max_length],dtype=torch.int64)
        for i in range(len(sequence)):
            padded_seq[i][0: sequence[i].shape[0]] = torch.tensor(sequence[i])
        return padded_seq
    # zero is special index for padding
    padded_seq = np.zeros(max_length, np.int64)
    padded_seq[0: sequence.shape[0]] = sequence
    return padded_seq

def collate_fn(batch, maxlen=None, has_al=True, flatten=False):
    if flatten:
        for k, v in batch.items():
            if batch[k].dim() == 2:
                batch[k] = batch[k].reshape(-1)
            else:
                batch[k] = batch[k].reshape(-1, batch[k].shape[-1])
    if maxlen == None:
        maxlen = max(batch['max_len']).item()
    batch['hist_activity'] = batch['hist_activity'][:, :maxlen]
    if has_al:
        batch['acts_labels'] = batch['acts_labels'][:, :maxlen]
    return batch

def collate_fn_cp(batch, maxlen=None, has_al=True):
    if maxlen==None:
        maxlen = max(batch['max_len']).item()
    batch_ = {}
    batch_['hist_activity'] = batch['hist_activity'][:, :maxlen]
    if has_al:
        batch_['acts_labels'] = batch['acts_labels'][:, :maxlen]
    return batch_

from torch.utils.data import _utils
def fetch(dataset, index_list):
    data = [dataset[idx] for idx in index_list]
    clfn = _utils.collate.default_collate
    return clfn(data)

class indexed_dataset(Dataset):
    def __init__(self, I, ds_pool, ds_train, ds_val, mode):
        self.mode = mode
        self.ds_pool = ds_pool
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.I = I
        self.size = len(self.I)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.mode == 'train':
            onegraphinput = fetch(self.ds_pool, self.I[idx]) 
            train_self = fetch(self.ds_train, [idx])
            for k, v in train_self.items():
                onegraphinput[k] = torch.cat([train_self[k], onegraphinput[k]])
            return onegraphinput
        elif self.mode == 'val':
            onegraphinput = fetch(self.ds_pool, self.I[idx])
            val_self = fetch(self.ds_val, [idx])
            onegraphinput_ = {}
            for k, v in val_self.items():
                onegraphinput_[k] = torch.cat([val_self[k], onegraphinput[k]])
            return onegraphinput_
        else:
            raise NotImplementedError

def SoftCrossEntropy(inputs, target, reduction='mean'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    N = inputs.shape[0]
    if reduction == 'mean':
        loss = torch.sum(torch.mul(log_likelihood, target)) / N
    elif reduction == 'sum':
        loss = torch.sum(torch.mul(log_likelihood, target))
    else:  # reduction == 'none'
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
    return loss

def recall_dataset(args, logger, model, topk, ds_pool, ds_train, ds_val, random_dl_train, sequential_dl_train, dl_val, device, index_path=None, is_build_vector_pool=False, mode='next_action_prediction', rmodel = None):
    assert args.classification_level == 'action'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_func = SoftCrossEntropy
    maxauc = 0
    start_epoch = 0
    if args.continuetrain == True:
        if os.path.exists('data/{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)):
            print('load data/{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight))
            checkpoint = torch.load('data/{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']+1
            maxauc = checkpoint['auc']
            print('load data/{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight))
            es = 0
    for epoch in range(start_epoch, args.epoch):
        if args.debug == True:
            index_flat = faiss.read_index(os.path.join(args.root, '{}_{}_{}_{}_{}_level_trained_{}_hard_negative_weight.index'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)))
        else:
            if index_path == '' or index_path == None or is_build_vector_pool == True:
                index_flat = build_vector_pool(args, logger, model, sequential_dl_train, device, rmodel=rmodel)
            else:
                logger.info('load index: {}'.format(index_path))
                index_flat = faiss.read_index(index_path)
        y_true = []
        y_pred = []
        t = tqdm(random_dl_train, desc='Trainset Recall, epoch:{}'.format(epoch))
        total_train_df_idx_list = []
        if rmodel:
            rmodel.eval()
        else:
            model.eval()
        for idx, batch in enumerate(t):
            batch = collate_fn(batch, has_al=False)
            for k, v in batch.items():
                batch[k] = batch[k].to(device)
            if rmodel:
                out = rmodel(batch)[0]
            else:
                out = model.seq(batch)[0]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            out = out.detach().cpu().numpy()
            I = index_flat.search(out, topk-1)[1]
            total_train_df_idx_list.extend(list(I))
        with open('data/{}_{}_{}_{}_{}_total_train_df_idx_list_{}_hard_negative_weight.pkl'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight), 'wb') as f:
            pickle.dump(total_train_df_idx_list, f)
        if args.debug == True:
            total_train_df_idx_list = pickle.load(open('data/{}_{}_{}_{}_{}_total_train_df_idx_list_{}_hard_negative_weight.pkl'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight), 'rb'))
        ds_t = indexed_dataset(total_train_df_idx_list, ds_pool, ds_train, ds_val, mode='train')
        dl_t = DataLoader(ds_t, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        t = tqdm(dl_t, desc='Trainset training, epoch:{}'.format(epoch))
        model.train()
        for idx, batch in enumerate(t):
            batch = collate_fn(batch, has_al=False, flatten=True)
            for k, v in batch.items():
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            y_ = batch['target_action'][::topk].type(torch.long)
            y = torch.nn.functional.one_hot(y_, num_classes=model.num_class) 
            if args.mixloss == True:
                aux_y = batch['target_action_label'][::topk].type(torch.long)
                mask = aux_y.unsqueeze(1).repeat(1, model.num_class) 
                onesubmask = 1 - mask  
                y = (y * onesubmask + 1 / (model.num_class - 1) * mask) * (1 - (y * mask)) 
            if args.add_graph_regularization == True:
                y_hat, grloss = model(batch)
            else:
                y_hat = model(batch)
            if args.use_sample_weight == True:
                target_action_label_ = batch['target_action_label'][::topk]
                sample_weight = torch.ones(target_action_label_.shape[0]).to(device)
                sample_weight += args.hard_negative_weight * target_action_label_
                l = loss_func(y_hat, y, reduction='none')
                l = l * sample_weight
                l = l.mean()
            else:
                l = loss_func(y_hat, y)
            if args.add_graph_regularization == True:
                l += grloss
            l.backward()
            optimizer.step()
            t.set_postfix(loss=l.item())
            y_true.append(batch['target_action_label'][::topk].detach().cpu().numpy())
            y_pred.append(-y_hat.gather(1, y_.unsqueeze(1)).squeeze().detach().cpu().numpy())
        if mode == 'next_action_prediction':
            y_true_all = np.hstack(y_true)
            y_pred_all = np.hstack(y_pred)
            auc = roc_auc_score(y_true_all, y_pred_all)
            fpr, tpr, threshold3 = bestthreshold_with_ROC(y_true_all, y_pred_all)
            logger.info("Train Epoch:{}\ntrain_loss:{}\nauc:{}\nfpr:{}\ntpr:{}".format(epoch, l.item(), auc, fpr, tpr))
        else:
            raise NotImplementedError
        print()
        logger.info('Eval Epoch:%d.....' % (epoch))
        y_true = []
        y_pred = []
        t = tqdm(dl_val, desc='Evalset Recall, epoch:{}'.format(epoch))
        total_val_df_idx_list = []
        if rmodel:
            rmodel.eval()
        else:
            model.eval()
        for idx, batch in enumerate(t):
            # batch = collate_fn(batch, has_al=False)
            for k, v in batch.items():
                batch[k] = batch[k].to(device)
            if rmodel:
                out = rmodel(batch)[0]
            else:
                out = model.seq(batch)[0]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            out = out.detach().cpu().numpy()
            I = index_flat.search(out, topk-1)[1]
            total_val_df_idx_list.extend(list(I))
        with open('data/{}_{}_{}_{}_{}_total_val_df_idx_list_{}_hard_negative_weight.pkl'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight), 'wb') as f:
            pickle.dump(total_val_df_idx_list, f)
        if args.debug == True:
            total_val_df_idx_list = pickle.load(open('data/{}_{}_{}_{}_{}_total_val_df_idx_list_{}_hard_negative_weight.pkl'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight), 'rb'))
        ds_t = indexed_dataset(total_val_df_idx_list, ds_pool, ds_train, ds_val, mode='val')
        dl_t = DataLoader(ds_t, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        t = tqdm(dl_t, desc='Evalset training, epoch:{}'.format(epoch))
        for idx, batch in enumerate(t):
            batch = collate_fn(batch, has_al=False, flatten=True)
            for k, v in batch.items():
                batch[k] = batch[k].to(device)
            y = batch['target_action'][::topk].type(torch.long)
            y_ = torch.nn.functional.one_hot(y, num_classes=model.num_class)
            if args.add_graph_regularization == True:
                y_hat, grloss = model(batch)
            else:
                y_hat = model(batch)
            l = loss_func(y_hat, y_)
            if args.add_graph_regularization == True:
                l += grloss
            y_hat = torch.nn.functional.softmax(y_hat, dim=-1)
            t.set_postfix(loss=l.item())
            if mode == 'next_action_prediction':
                y_true.append(batch['target_action_label'][::topk].detach().cpu().numpy())
                y_pred.append(-y_hat.gather(1, y.unsqueeze(1)).squeeze().detach().cpu().numpy())
            else:
                raise NotImplementedError
        y_true_all = np.hstack(y_true)
        y_pred_all = np.hstack(y_pred)
        auc = roc_auc_score(y_true_all, y_pred_all)
        fpr, tpr, threshold3 = bestthreshold_with_ROC(y_true_all, y_pred_all)
        logger.info(
            "Eval Epoch:{}\neval_loss:{}\nAuc:{}\nfpr:{}\ntpr:{}".format(epoch, l.item(), auc, fpr, tpr))
        if auc > maxauc:
            maxauc = auc
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': l.item(),
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
                'threshold3': threshold3,
                'y_true_all': y_true_all,
                'y_pred_all': y_pred_all
            }
            es = 0
            torch.save(state, os.path.join(args.root, '{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)))
        else:
            es += 1
            logger.info("Counter {} of 5".format(es))
            if es > 2:
                logger.info("Early stopping with best_auc: {}".format(maxauc))
                break
    state = torch.load(os.path.join(args.root, '{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)))
    logger.info("----------------------------------------------------------------------------------------------------------------------")
    logger.info("best Epoch:{}\nloss:{}\nAuc:{}\nFPR:{}\nTPR:{}".format(state['epoch'], state['loss'], state['auc'], state['fpr'], state['tpr']))
