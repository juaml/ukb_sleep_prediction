import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold


class DownsampledRepeatedStratifiedKFold(StratifiedKFold):

    def split(self, X, y=None, groups=None):
        for i in range(self.n_repeats):
            idx_y0 = np.argwhere(~np.in1d(y, self.pos_labels)).ravel()
            idx_y1 = np.argwhere(np.in1d(y, self.pos_labels)).ravel()

            min_size = min(len(idx_y0), len(idx_y1))
            n_samples = self.subsample_size
            if n_samples == 'auto' or n_samples > min_size:
                n_samples = min_size

            idx0 = np.random.choice(idx_y0, size=n_samples)
            idx1 = np.random.choice(idx_y1, size=n_samples)
            idx = np.hstack((idx0, idx1))
            np.random.shuffle(idx)
            t_groups = groups[idx] if groups is not None else None
            if isinstance(X, pd.DataFrame):
                t_X = X.iloc[idx]
            else:
                t_X = X[idx]
            return super().split(t_X, y[idx], groups=t_groups)

    def __init__(self, n_splits=5, n_repeats=10, random_state=None,
                 pos_labels=None, subsample_size='auto'):
        if pos_labels is None:
            pos_labels = [1]
        if not isinstance(pos_labels, list):
            pos_labels = [pos_labels]
        self.pos_labels = pos_labels
        self.subsample_size = subsample_size
        self.n_repeats = n_repeats
        super().__init__(n_splits=n_splits, random_state=random_state)