from utils import *
import argparse
from dataprocess import *
from torch.utils.data import DataLoader
from model import *
import sklearn
import copy


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcroot', default='data/r4.2', type=str)
    parser.add_argument('--root', default='data', type=str)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--rtph', type=str, default='realtime',help='realtime/posthoc')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0/cuda:1/cpu')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--classification_level', default='action', type=str, help='action ')
    parser.add_argument('--model', default='GSL', type=str, help='GCN/GAT')
    parser.add_argument('--encoder_type', default='lstm', type=str, help='lstm/gru/transformer')
    parser.add_argument('--graph_metric_type', default='weighted_cosine', type=str, help='weighted_cosine/cosine')
    parser.add_argument("--add_graph_regularization", default=False, action='store_true')
    parser.add_argument('--hard_negative_weight', type=int, default=0, help='hard_negative_weight')
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument('--reduction', default='avgpooling', type=str, help='avgpooling/selfattention+avgpooling/lastpositionattention/clsattention/attentionnetpooling')
    parser.add_argument('--num_workers', type=int, default=8, help='index dataloader num_workers')
    return parser.parse_args()


def inference(args, logger, model, topk, ds_pool, ds_val, sequential_dl_train, dl_val, device, index_path=None, is_build_vector_pool=False, mode='next_action_prediction', rmodel = None, prefix = ''): 
    assert args.classification_level == 'action'
    if args.debug == True:
        index_flat = faiss.read_index(os.path.join(args.root, '{}_{}_{}_{}_{}_level_trained_{}_hard_negative_weight.index'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)))
    else:
        if index_path == '' or index_path == None or is_build_vector_pool == True:
            index_flat = build_vector_pool(args, logger, model, sequential_dl_train, device, rmodel=rmodel)
        else:
            logger.info('load index: {}'.format(index_path))
            index_flat = faiss.read_index(index_path)
    logger.info('Eval .....')
    model.eval()
    y_true = []
    y_pred = []
    embedding_list= []
    adj_list = []
    t = tqdm(dl_val, desc='Evalset Recall')
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
    ds_t = indexed_dataset(total_val_df_idx_list, ds_pool, None, ds_val, mode='val')
    dl_t = DataLoader(ds_t, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    t = tqdm(dl_t, desc='Evalset training')
    for idx, batch in enumerate(t):
        batch = collate_fn(batch, has_al=False, flatten=True)
        for k, v in batch.items():
            batch[k] = batch[k].to(device)
        y = batch['target_action'][::topk].type(torch.long) 
        y_hat, intention_vec, adj = model(batch)
        y_hat = torch.nn.functional.softmax(y_hat, dim=-1)
        if mode == 'next_action_prediction':
            y_true.append(batch['target_action_label'][::topk].detach().cpu().numpy())
            y_pred.append(-y_hat.gather(1, y.unsqueeze(1)).squeeze().detach().cpu().numpy())
            embedding_list.append(intention_vec.detach().cpu().numpy())
            adj_list.append(adj.detach().cpu().numpy())
        else:
            raise NotImplementedError
    y_true_all = np.hstack(y_true)
    y_pred_all = np.hstack(y_pred)
    tot_embedding = np.vstack(embedding_list)
    tot_adj = np.vstack(adj_list)
    auc = roc_auc_score(y_true_all, y_pred_all)
    fpr, tpr, threshold3 = bestthreshold_with_ROC(y_true_all, y_pred_all)
    logger.info(
        "Eval\nAuc:{}\nfpr:{}\ntpr:{}".format(auc, fpr, tpr))
    state = {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'threshold3': threshold3,
        'y_true_all': y_true_all,
        'y_pred_all': y_pred_all
    }
    torch.save(state, os.path.join(args.root, '{}_inference_result.pth'.format(prefix)))
    logger.info("----------------------------------------------------------------------------------------------------------------------")
    logger.info("----------------------------------------------------------------------------------------------------------------------")
    logger.info("Auc:{}\nFPR:{}\nTPR:{}".format(state['auc'], state['fpr'], state['tpr']))


if __name__ == '__main__':
    args = ArgumentParser()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if not os.path.isdir(os.path.join(args.root, 'output')):
        os.makedirs(os.path.join(args.root, 'output'))
    checkpoint_path = "data/{}_{}_{}_{}_{}_level_next_action_prediction-checkpoint_{}_hard_negative_weight.pth".format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight)
    assert os.path.exists(checkpoint_path)
    print('load {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    prefix = 'output/{}_{}_{}_{}_{}_level_checkpoint_{}_hard_negative_weight_{}'.format(args.srcroot[-4:], args.model, args.encoder_type, args.reduction, args.classification_level, args.hard_negative_weight, args.rtph)
    if args.debug == True:
        logger = get_logger(os.path.join(args.root, 'output/debug.log'))
    else:
        logger = get_logger(os.path.join(args.root, '{}_inference.log'.format(prefix)))
    logger.info(args)
    logger.info('write log to {}_inference.log'.format(prefix))
    num_features, cat_features, seq_features, encoders, df = action_sample(args.srcroot, args.root)
    dfval = df[1]
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
    cat_nums = {k: v.dimension() for k, v in encoders.items()} 
    logger.info(cat_nums)
    num_class = (cat_nums['hist_activity'] - 3) * 24 + 25 
    model = CreateInsiderClassifier(num_features, cat_features, seq_features, cat_nums, reduction=args.reduction,
                                      encoder_type=args.encoder_type, epsilon=0.5, graph_metric_type=args.graph_metric_type,
                                      topk=15, num_class=num_class, add_graph_regularization=args.add_graph_regularization, gnn=args.model, embedding_hook=True)
    label_col = ['target_action', 'target_action_label']
    ds_pool = Df2Dataset(dfpool, num_features, cat_features, seq_features, encoders, label_col=label_col)
    ds_val = Df2Dataset(dfval, num_features, cat_features, seq_features, encoders, label_col=label_col)
    sequential_dl_train = DataLoader(ds_pool, batch_size=args.batch_size, num_workers=8, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=8, shuffle=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device)
    start_epoch = checkpoint['epoch'] + 1
    maxauc = checkpoint['auc']
    inference(args, logger, model, 15, ds_pool, ds_val, sequential_dl_train, dl_val, device, index_path='', mode='next_action_prediction', rmodel=None, prefix=prefix)
