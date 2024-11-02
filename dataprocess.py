import pandas as pd
import numpy as np
import os
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from collections import Counter

pd.set_option('display.max_columns', None)

class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_cnt=5, word2idx=None, idx2word=None):
        super().__init__()
        self.min_cnt = min_cnt
        self.word2idx = word2idx if word2idx else dict()
        self.idx2word = idx2word if idx2word else dict()

    def fit(self, x, y=None):
        if not self.word2idx:
            counter = Counter(np.asarray(x).ravel())

            selected_terms = sorted(
                list(filter(lambda x: counter[x] >= self.min_cnt, counter)))

            self.word2idx = dict(
                zip(selected_terms, range(1, len(selected_terms) + 1)))
            self.word2idx['__PAD__'] = 0
            if '__UNKNOWN__' not in self.word2idx:
                self.word2idx['__UNKNOWN__'] = len(self.word2idx)

        if not self.idx2word:
            self.idx2word = {
                index: word for word, index in self.word2idx.items()}

        return self

    def transform(self, x):
        transformed_x = list()
        for term in np.asarray(x).ravel():
            try:
                transformed_x.append(self.word2idx[term])
            except KeyError:
                transformed_x.append(self.word2idx['__UNKNOWN__'])

        return np.asarray(transformed_x, dtype=np.int64)

    def dimension(self):
        return len(self.word2idx)


class SequenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sep=' ', min_cnt=5, max_len=None,
                 word2idx=None, idx2word=None):
        super().__init__()
        self.sep = sep
        self.min_cnt = min_cnt
        self.max_len = max_len

        self.word2idx = word2idx if word2idx else dict()
        self.idx2word = idx2word if idx2word else dict()

    def fit(self, x, y=None):
        if not self.word2idx:
            counter = Counter()
            max_len = 0
            for sequence in np.array(x).ravel():
                words = sequence.split(self.sep)
                counter.update(words)
                max_len = max(max_len, len(words))

            if self.max_len is None:
                self.max_len = max_len

            # drop rare words
            words = sorted(list(filter(lambda x: counter[x] >= self.min_cnt, counter)))

            self.word2idx = dict(zip(words, range(1, len(words) + 1)))
            self.word2idx['__PAD__'] = 0
            if '__UNKNOWN__' not in self.word2idx:
                self.word2idx['__UNKNOWN__'] = len(self.word2idx)

        if not self.idx2word:
            self.idx2word = {
                index: word for word, index in self.word2idx.items()}

        if not self.max_len:
            max_len = 0
            for sequence in np.array(x).ravel():
                words = sequence.split(self.sep)
                max_len = max(max_len, len(words))
            self.max_len = max_len
        return self

    def transform(self, x):
        transformed_x = list()

        for sequence in np.asarray(x).ravel():
            words = list()
            for word in sequence.split(self.sep):
                try:
                    words.append(self.word2idx[word])
                except KeyError:
                    words.append(self.word2idx['__UNKNOWN__'])

            transformed_x.append(
                np.asarray(words[0:self.max_len], dtype=np.int64))
        return transformed_x
        # return np.asarray(transformed_x, dtype=object)

    def dimension(self):
        return len(self.word2idx)

    def max_length(self):
        return self.max_len

def action_sample(srcroot, root):
    num_features = ['O', 'C', 'E', 'A', 'N']
    cat_features = ['role', 'functional_unit', 'department', 'team', 'supervisor', 'host']
    seq_features = ['hist_activity']
    # https://drive.google.com/drive/folders/1Szltzgv1VAzmWg0I193GunwB_4trkZwo?usp=drive_link
    encoders = pickle.load(open(os.path.join(root, str(srcroot[-4:]) + '_' + 'encoders.pkl'), 'rb'))
    dftrain = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dftrain_action.pkl'))
    dfval = pd.read_pickle(os.path.join(root, str(srcroot[-4:]) + '_' + 'dfval_action_do_all.pkl')) 
    return num_features, cat_features, seq_features, encoders, [dftrain, dfval]
