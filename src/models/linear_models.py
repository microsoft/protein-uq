import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from scipy import stats

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from train_all import split_dict 
from utils import SequenceDataset, load_dataset
from csv import writer
from pathlib import Path

class CSVDataset(Dataset):

    def __init__(self, fpath=None, df=None, split=None, outputs=[]):
        if df is None:
            self.data = pd.read_csv(fpath)
        else:
            self.data = df
        if split is not None:
            self.data = self.data[self.data['split'] == split]
        self.outputs = outputs
        self.data = self.data[['sequence'] + self.outputs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return [row['sequence'], *row[self.outputs]]


class Tokenizer(object):
    """Convert between strings and their one-hot representations."""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x) -> str:
        return ''.join([self.t_to_a[t] for t in x])

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='file path to data directory')
parser.add_argument('task', type=str)
parser.add_argument('--scale', type=bool, default=False)
parser.add_argument('--solver', type=str, default='lsqr')
parser.add_argument('--max_iter', type=float, default=1e6)
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--ensemble', action='store_true')

args = parser.parse_args()

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYVXU'
# grab data
split = split_dict[args.task]
train, test, _ = load_dataset(args.dataset, split+'.csv', val_split=False)

ds_train = SequenceDataset(train)
ds_test = SequenceDataset(test)

# tokenize train data
all_train = list(ds_train)
X_train = [i[0] for i in all_train]
y_train = [i[1] for i in all_train]
if args.dataset == 'meltome':
    AAINDEX_ALPHABET += 'XU'
tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize
X_train = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in X_train]


# tokenize test data
all_test = list(ds_test)
X_test = [i[0] for i in all_test]
y_test = [i[1] for i in all_test]
tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize
X_test = [torch.tensor(tokenizer.tokenize(i)).view(-1,1) for i in X_test]


# padding
maxlen_train = max([len(i) for i in X_train])
maxlen_test = max([len(i) for i in X_test])
maxlen = max([maxlen_train, maxlen_test])
# pad_tok = alphabet.index(PAD)

X_train = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.) for i in X_train]
X_train_enc = [] # ohe
for i in X_train:
    i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
    i_onehot.zero_()
    i_onehot.scatter_(1, i, 1)
    X_train_enc.append(i_onehot)
X_train_enc = np.array([np.array(i.view(-1)) for i in X_train_enc]) # flatten

X_test = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", 0.) for i in X_test]
X_test_enc = [] # ohe
for i in X_test:
    i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
    i_onehot.zero_()
    i_onehot.scatter_(1, i, 1)
    X_test_enc.append(i_onehot)
X_test_enc = np.array([np.array(i.view(-1)) for i in X_test_enc]) # flatten


# scale X
if args.scale:
    scaler = StandardScaler()
    X_train_enc = scaler.fit_transform(X_train_enc)
    X_test_enc = scaler.transform(X_test_enc)

def evaluate_miscalibration_area(abs_error, uncertainty):
        standard_devs = abs_error/uncertainty
        probabilities = [2 * (stats.norm.cdf(
            standard_dev) - 0.5) for standard_dev in standard_devs]
        sorted_probabilities = sorted(probabilities)

        fraction_under_thresholds = []
        threshold = 0

        for i in range(len(sorted_probabilities)):
            while sorted_probabilities[i] > threshold:
                fraction_under_thresholds.append(i/len(sorted_probabilities))
                threshold += 0.001

        # Condition used 1.0001 to catch floating point errors.
        while threshold < 1.0001:
            fraction_under_thresholds.append(1)
            threshold += 0.001

        thresholds = np.linspace(0, 1, num=1001)
        miscalibration = [np.abs(
            fraction_under_thresholds[i] - thresholds[i]) for i in range(
                len(thresholds))]
        miscalibration_area = 0
        for i in range(1, 1001):
            miscalibration_area += np.average([miscalibration[i-1],
                                               miscalibration[i]]) * 0.001

        return {'fraction_under_thresholds': fraction_under_thresholds,
                'thresholds': thresholds,
                'miscalibration_area': miscalibration_area}

def evaluate_log_likelihood(error, uncertainty):
        log_likelihood = 0
        optimal_log_likelihood = 0

        for err, unc in zip(error, uncertainty):
            # Encourage small standard deviations.
            log_likelihood -= np.log(2 * np.pi * max(0.00001, unc**2)) / 2
            optimal_log_likelihood -= np.log(2 * np.pi * err**2) / 2

            # Penalize for large error.
            log_likelihood -= err**2/(2 * max(0.00001, unc**2))
            optimal_log_likelihood -= 1 / 2

        return {'log_likelihood': log_likelihood,
                'optimal_log_likelihood': optimal_log_likelihood,
                'average_log_likelihood': log_likelihood / len(error),
                'average_optimal_log_likelihood': optimal_log_likelihood / len(error)}

def main(args, X_train_enc, y_train, y_test):

    # print('Parameters...')
    # print('Solver: %s, MaxIter: %s, Tol: %s' % (args.solver, args.max_iter, args.tol))

    print('Training...')
    lr = BayesianRidge()
    lr.fit(X_train_enc, y_train)
    preds_mean, preds_std = lr.predict(X_test_enc, return_std=True)

    rho = stats.spearmanr(y_test, preds_mean)
    rmse = mean_squared_error(y_test, preds_mean, squared=False)
    mae = mean_absolute_error(y_test, preds_mean)
    r2 = r2_score(y_test, preds_mean) 

    print('TEST RHO: ', rho)
    print('TEST RMSE: ', rmse)
    print('TEST MAE: ', mae)
    print('TEST R2: ', r2) 

    residual = np.abs(y_test - preds_mean)
    coverage = residual < 2*preds_std
    width_range = 4*preds_std/(max(y_train)-min(y_train))

    df = pd.DataFrame()
    df['y_test'] = y_test
    df['preds_mean'] = preds_mean
    df['preds_std'] = preds_std
    df['residual'] = residual
    df['coverage'] = coverage
    df['width/range'] = width_range 
    df.to_csv(f'{Path.cwd()}/evals_new/{args.dataset}_linear_{split}_test_preds.csv', index=False)

    rho_unc, p_rho_unc = stats.spearmanr(df['residual'], df['preds_std'])
    percent_coverage = sum(df['coverage'])/len(df)
    average_width_range = df['width'].mean()/(max(y_train)-min(y_train))
    miscalibration_area_results = evaluate_miscalibration_area(df['residual'], df['preds_std']) 
    miscalibration_area = miscalibration_area_results['miscalibration_area']
    ll_results = evaluate_log_likelihood(df['residual'], df['preds_std'])
    average_log_likelihood = ll_results['average_log_likelihood']
    average_optimal_log_likelihood = ll_results['average_optimal_log_likelihood']

    print('TEST RHO UNCERTAINTY: ', rho_unc)
    print('TEST RHO UNCERTAINTY P-VALUE: ', p_rho_unc)
    print('PERCENT COVERAGE: ', percent_coverage)
    print('AVERAGE WIDTH / TRAINING SET RANGE: ', average_width_range)
    print('MISCALIBRATION AREA: ', miscalibration_area)
    print('AVERAGE LOG LIKELIHOOD: ', average_log_likelihood)
    print('AVERAGE OPTIMAL LOG LIKELIHOOD: ', average_optimal_log_likelihood)
    print('LL / LL_OPT:', average_log_likelihood/average_optimal_log_likelihood)

    with open(Path.cwd() / 'evals_new'/ (args.dataset+'_results.csv'), 'a', newline='') as f:
        writer(f).writerow([args.dataset, 'linearBayesianRidge', split, 
                            round(rho,2), round(rmse,2), round(mae,2), round(r2,2), 
                            round(rho_unc,2), round(p_rho_unc,2), round(percent_coverage,2), round(average_width_range,2)])


if args.ensemble:
    for i in range(10):
        np.random.seed(i)
        torch.manual_seed(i)
        main(args, X_train_enc, y_train, y_test)
else:
    np.random.seed(0)
    torch.manual_seed(1)
    main(args, X_train_enc, y_train, y_test)
