import gpytorch

import argparse
from sklearn.preprocessing import StandardScaler

import torch
import numpy as np
import torch.nn.functional as F

from train_all import split_dict 
from utils import SequenceDataset, load_dataset, calculate_metrics
from csv import writer
from pathlib import Path

import pandas as pd

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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, device_ids=[0]):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            self.covar_module, device_ids=device_ids,
            output_device=device_ids[0]
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

np.random.seed(0)
torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='file path to data directory')
parser.add_argument('task', type=str)
parser.add_argument('--scale', action='store_true')
parser.add_argument('--gpu', type=int, nargs='+', default=[0])
parser.add_argument('--size', type=int, default=0)
parser.add_argument('--length', type=float, default=1.0)
args = parser.parse_args()

args.dropout = ''

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYVXU'
# grab data
split = split_dict[args.task]
train, test, _ = load_dataset(args.dataset, split+'.csv', val_split=False)

ds_train = SequenceDataset(train, args.dataset)
ds_test = SequenceDataset(test, args.dataset)

print('Encoding...')
# tokenize data
all_train = list(ds_train)
X_train = [i[0] for i in all_train]
y_train = [i[1] for i in all_train]
all_test = list(ds_test)
X_test = [i[0] for i in all_test]
y_test = [i[1] for i in all_test]

tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize
X_train = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in X_train]
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


# scale
if args.scale:
    scaler = StandardScaler()
    X_train_enc = scaler.fit_transform(X_train_enc)
    X_test_enc = scaler.transform(X_test_enc)
    y_train = scaler.fit_transform(np.array(y_train)[:, None])[:, 0]
    y_test = scaler.transform(np.array(y_test)[:, None])[:, 0]

algorithm_type = 'GPcontinuous'

try:
    preds_df = pd.read_csv(
            f"evals_new/evals_new_gp_ky/{args.dataset}_{algorithm_type}_{split}_test_preds.csv",
        )
except FileNotFoundError:
    preds_df = pd.read_csv(
            f"evals_new/evals_new_gp_ky/meltome_GPcontinuous_mixed_split_short999_test_preds.csv",
        )

y_test = preds_df.y_test.values
preds_mean = preds_df.preds_mean.values
preds_std = preds_df.preds_std.values

# unscale
if args.scale:
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    preds_mean = scaler.inverse_transform(preds_mean.reshape(-1, 1))
    preds_std = preds_std.reshape(-1, 1) * np.sqrt(scaler.var_)

print('Calculating metrics...')

metrics = calculate_metrics(np.array(y_test).ravel(), preds_mean.ravel(), preds_std.ravel(), args, split, y_train, algorithm_type)

# Write metric results to file
row = [args.dataset, algorithm_type, split]
for metric in metrics:
    if isinstance(metric, str):
        row.append(metric)
    else:
        try:
            row.append(round(metric, 2))
        except TypeError:
            row.append(round(metric[0], 2))
with open(Path.cwd() / 'evals_new'/ (args.dataset+'_results.csv'), 'a', newline='') as f:
    writer(f).writerow(row)
