from utils import *
import argparse
from torch.utils.data import DataLoader
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from model import *
import sklearn
import copy
import pandas as pd


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcroot', default='data/r4.2', type=str)
    parser.add_argument('--root', default='data', type=str)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0/cuda:1/cpu')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--UnderSampling', type=float, default=-1, help='UnderSampling rate')
    parser.add_argument('--OverSampling', type=float, default=-1, help='OverSampling rate')
    parser.add_argument('--classification_level', default='action', type=str, help='action')
    parser.add_argument('--model', default='GSL', type=str, help='GCN/GAT')
    parser.add_argument('--encoder_type', default='lstm', type=str, help='lstm/gru/transformer')
    parser.add_argument("--mixloss", default=False, action='store_true')
    parser.add_argument("--use_sample_weight", default=False, action='store_true')
    parser.add_argument('--graph_metric_type', default='cosine', type=str, help='weighted_cosine/cosine/...')
    parser.add_argument("--add_graph_regularization", default=False, action='store_true')
    parser.add_argument('--hard_negative_weight', type=int, default=0, help='hard_negative_weight')
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument('--reduction', default='avgpooling', type=str, help='avgpooling/selfattention+avgpooling/lastpositionattention/clsattention/attentionnetpooling')
    parser.add_argument('--num_workers', type=int, default=8, help='index dataloader num_workers')
    parser.add_argument("--continuetrain", default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = ArgumentParser()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if not os.path.isdir(os.path.join(args.root, 'output')):
        os.makedirs(os.path.join(args.root, 'output'))
    print(os.path.join(args.root, 'output/{}_{}_{}_{}_{}_level_checkpoint_{}_hard_negative_weight.log'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)))
    if args.debug == True:
        logger = get_logger(os.path.join(args.root, 'output/debug.log'))
    else:
        logger = get_logger(os.path.join(args.root, 'output/{}_{}_{}_{}_{}_level_checkpoint_{}_hard_negative_weight.log'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)))
    logger.info(args)
    logger.info('write log to output/{}_{}_{}_{}_{}_level_checkpoint_{}_hard_negative_weight.log'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight))
    num_features = ['O', 'C', 'E', 'A', 'N']
    cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
    seq_features = ['hist_activity']
    encoders = pickle.load(open(os.path.join(args.root, str(args.srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
    dftrain = pd.read_pickle(os.path.join(args.root, str(args.srcroot[-4:]) + '_' + 'dftrain_action.pkl'))
    dfval = pd.read_pickle(os.path.join(args.root, str(args.srcroot[-4:]) + '_' + 'dfval_action_do_all.pkl')) 
    if not os.path.isfile(os.path.join(args.root, str(args.srcroot[-4:]) + '_' + 'dfpool_action_drop_duplicates.pkl')):
        dfpool = copy.deepcopy(df[0])
        logger.info("original dfpool shape is {}".format(dfpool.shape))
        tqdm.pandas(desc='apply')
        dfpool['hash'] = dfpool['hist_activity'].progress_apply(lambda row: row.tostring())
        dfpool = dfpool.drop_duplicates(subset='hash')
        logger.info("after drop_duplicates, the dfpool shape is {}".format(dfpool.shape))
        del dfpool['hash']
        dfpool = dfpool[dfpool['target_action_label'] == 0]
        dfpool.to_pickle(os.path.join(args.root, str(args.srcroot[-4:]) + '_' + 'dfpool_action_drop_duplicates.pkl'))
    else:
        dfpool = pd.read_pickle(os.path.join(args.root, str(args.srcroot[-4:]) + '_' + 'dfpool_action_drop_duplicates.pkl'))
    if args.UnderSampling != -1 or args.OverSampling != -1:
        for i in range(1):
            Y = df[i]['target_action_label']
            X = df[i].drop('target_action_label', axis=1)
            logger.info("positive : negtive = %d : %d" % (Y[Y == 1].count(), Y[Y == 0].count()))
            # fit and apply the transform
            if args.UnderSampling != -1:
                under = RandomUnderSampler(sampling_strategy=args.UnderSampling)
                X, Y = under.fit_resample(X, Y)
                logger.info("After UnderSampling, positive : negtive = %d : %d" % (Y[Y == 1].count(), Y[Y == 0].count()))
            if args.OverSampling != -1:
                over = RandomOverSampler(sampling_strategy=args.OverSampling)
                X, Y = over.fit_resample(X, Y)
                logger.info("After OverSampling, positive : negtive = %d : %d" % (Y[Y == 1].count(), Y[Y == 0].count()))
            if args.UnderSampling != -1 or args.OverSampling != -1:
                df[i] = X
                df[i]['target_action_label'] = Y
    dftrain, dfval = df
    cat_nums = {k: v.dimension() for k, v in encoders.items()}
    logger.info(cat_nums)
    num_class = (cat_nums['hist_activity'] - 3) * 24 + 25
    model = CreateInsiderClassifier(num_features, cat_features, seq_features, cat_nums, reduction=args.reduction,
                                    encoder_type=args.encoder_type, epsilon=0.5, graph_metric_type=args.graph_metric_type,
                                    topk=15, num_class=num_class, add_graph_regularization=args.add_graph_regularization, gnn=args.model)
    label_col = ['target_action', 'target_action_label']
    ds_pool = Df2Dataset(dfpool, num_features, cat_features, seq_features, encoders, label_col=label_col)
    ds_train = Df2Dataset(dftrain, num_features, cat_features, seq_features, encoders, label_col=label_col)
    ds_val = Df2Dataset(dfval, num_features, cat_features, seq_features, encoders, label_col=label_col)
    random_dl_train = DataLoader(ds_train, batch_size=args.batch_size, num_workers=8, shuffle=False)
    sequential_dl_train = DataLoader(ds_pool, batch_size=args.batch_size, num_workers=8, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=8, shuffle=False)
    model = model.to(device)
    recall_dataset(args, logger, model, 15, ds_pool, ds_train, ds_val, random_dl_train, sequential_dl_train, dl_val, device, index_path='', mode='next_action_prediction')
