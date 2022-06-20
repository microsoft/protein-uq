import argparse
from csv import writer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from train_all import split_dict
from utils import SequenceDataset, Tokenizer, calculate_metrics, load_dataset, load_esm_dataset, vocab

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="file path to data directory")
parser.add_argument("task", type=str)
parser.add_argument("--scale", action="store_true")
parser.add_argument("--max_iter", type=int, default=300)
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--alpha_1", type=float, default=1e-6)
parser.add_argument("--alpha_2", type=float, default=1e-6)
parser.add_argument("--lambda_1", type=float, default=1e-6)
parser.add_argument("--lambda_2", type=float, default=1e-6)
parser.add_argument("--esm", action="store_true")
parser.add_argument("--esm_mean", action="store_true")
parser.add_argument("--esm_mut_mean", action="store_true")
parser.add_argument("--esm_flip", action="store_true")
parser.add_argument("--esm_gb1_shorten", action="store_true")
args = parser.parse_args()

args.dropout = ""

# grab data
split = split_dict[args.task]

if args.esm:
    train, _, test, max_length = load_esm_dataset(
        args.dataset,
        "esm1b",
        split,
        args.esm_mean,
        args.esm_mut_mean,
        args.esm_flip,
        args.esm_gb1_shorten,
    )
    X_train_enc = np.array([i[0].numpy() for i in train]).squeeze()
    y_train = [i[1].item() for i in train]
    X_test_enc = np.array([i[0].numpy() for i in test]).squeeze()
    y_test = [i[1].item() for i in test]
else:
    train, test, _ = load_dataset(args.dataset, split + ".csv", val_split=False)
    ds_train = SequenceDataset(train, args.dataset)
    ds_test = SequenceDataset(test, args.dataset)

    print("Encoding...")
    # tokenize data
    all_train = list(ds_train)
    X_train = [i[0] for i in all_train]
    y_train = [i[1] for i in all_train]
    all_test = list(ds_test)
    X_test = [i[0] for i in all_test]
    y_test = [i[1] for i in all_test]

    tokenizer = Tokenizer(vocab)  # tokenize
    X_train = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in X_train]
    X_test = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in X_test]

    # padding
    maxlen_train = max([len(i) for i in X_train])
    maxlen_test = max([len(i) for i in X_test])
    maxlen = max([maxlen_train, maxlen_test])
    # pad_tok = alphabet.index(PAD)

    X_train = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.0) for i in X_train]
    X_train_enc = []  # ohe
    for i in X_train:
        i_onehot = torch.FloatTensor(maxlen, len(vocab))
        i_onehot.zero_()
        i_onehot.scatter_(1, i, 1)
        X_train_enc.append(i_onehot)
    X_train_enc = np.array([np.array(i.view(-1)) for i in X_train_enc])  # flatten

    X_test = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.0) for i in X_test]
    X_test_enc = []  # ohe
    for i in X_test:
        i_onehot = torch.FloatTensor(maxlen, len(vocab))
        i_onehot.zero_()
        i_onehot.scatter_(1, i, 1)
        X_test_enc.append(i_onehot)
    X_test_enc = np.array([np.array(i.view(-1)) for i in X_test_enc])  # flatten


# scale X
if args.scale:
    scaler = StandardScaler()
    X_train_enc = scaler.fit_transform(X_train_enc)
    X_test_enc = scaler.transform(X_test_enc)
    y_train = scaler.fit_transform(np.array(y_train)[:, None])[:, 0]
    y_test = scaler.transform(np.array(y_test)[:, None])[:, 0]


def main(args, X_train_enc, y_train, y_test):

    # print('Parameters...')
    # print('Solver: %s, MaxIter: %s, Tol: %s' % (args.solver, args.max_iter, args.tol))

    print("Training...")
    lr = BayesianRidge(
        n_iter=args.max_iter,
        tol=args.tol,
        alpha_1=args.alpha_1,
        alpha_2=args.alpha_2,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
    )
    lr.fit(X_train_enc, y_train)
    preds_mean, preds_std = lr.predict(X_test_enc, return_std=True)

    print("Calculating metrics...")
    algorithm_type = "linearBayesianRidge"
    if args.esm:
        algorithm_type += "_esm1b"
        if args.esm_mean:
            algorithm_type += "_mean"
        if args.esm_mut_mean:
            algorithm_type += "_mutmean"
        if args.esm_flip:
            algorithm_type += "_flip"
        if args.esm_gb1_shorten:
            algorithm_type += "_gb1shorten"

    metrics = calculate_metrics(y_test, preds_mean, preds_std, args, split, y_train, algorithm_type)

    # Write metric results to file
    row = [args.dataset, algorithm_type, split]
    for metric in metrics:
        if isinstance(metric, str):
            row.append(metric)
        else:
            row.append(round(metric, 2))
    with open(Path.cwd() / "evals_new" / (args.dataset + "_results.csv"), "a", newline="") as f:
        writer(f).writerow(row)


np.random.seed(0)
torch.manual_seed(1)
main(args, X_train_enc, y_train, y_test)
