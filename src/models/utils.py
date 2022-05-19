import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Any, List
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

vocab = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
pad_index = len(vocab)  # pad index is 20


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
        return "".join([self.t_to_a[t] for t in x])


class ASCollater(object):
    def __init__(
        self, alphabet: str, tokenizer: object, pad=False, pad_tok=0.0, backwards=False
    ):
        self.pad = pad
        self.pad_tok = pad_tok
        self.tokenizer = tokenizer
        self.backwards = backwards
        self.alphabet = alphabet

    def __call__(
        self,
        batch: List[Any],
    ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        sequences = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        sequences = [i.view(-1, 1) for i in sequences]
        maxlen = max([i.shape[0] for i in sequences])
        padded = [
            F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", self.pad_tok)
            for i in sequences
        ]
        padded = torch.stack(padded)
        mask = [torch.ones(i.shape[0]) for i in sequences]
        mask = [F.pad(i, (0, maxlen - i.shape[0])) for i in mask]
        mask = torch.stack(mask)
        y = data[1]
        y = torch.tensor(y).unsqueeze(-1)
        ohe = []
        for i in padded:
            i_onehot = torch.FloatTensor(maxlen, len(self.alphabet))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            ohe.append(i_onehot)
        padded = torch.stack(ohe)

        return padded, y, mask


def encode_pad_seqs(s, length, vocab=vocab):
    """pads all sequences, converts AA string to np.array of indices"""
    aa_dict = {k: v for v, k in enumerate(vocab)}
    result = np.full((length), pad_index)
    for i, l in enumerate(s):
        index = aa_dict[l]
        result[i] = index
    return result


def get_data(df, max_length, encode_pad=True, zip_dataset=True, reverse_seq_target=False, one_hots=False): 
    """returns encoded and padded sequences with targets"""
    target = df.target.values.tolist()
    seq = df.sequence.values.tolist() 
    if encode_pad: 
        seq = [encode_pad_seqs(s, max_length) for s in seq]
        print('encoded and padded all sequences to length', max_length)

    if one_hots:
        seq = [one_hot_pad_seqs(s, max_length) for s in seq]
        print('one-hot encoded and padded all sequences to length', max_length)
        print('flattened one-hot sequences')
        return np.array(seq), np.array(target)

    if zip_dataset:
        if reverse_seq_target:
            return list(zip(target, seq))
        else:
            return list(zip(seq, target))
    else: 
        return torch.FloatTensor(seq), torch.FloatTensor(target)


def load_dataset(dataset, split, val_split=True): # TODO: get updated version of function from FLIP
    """returns dataframe of train, (val), test sets, with max_length param"""

    # datadir = "../../data/" + dataset + "/splits/"
    datadir = "/home/kpg/microsoft/protein-uq/data/" + dataset + "/splits/"  # TODO: fix path (this one for debugging)

    path = datadir + split
    print("reading dataset:", split)

    df = pd.read_csv(path)

    df.sequence.apply(
        lambda s: re.sub(r"[^A-Z]", "", s.upper())
    )  # remove special characters
    max_length = max(df.sequence.str.len())

    if val_split is True:
        test = df[df.set == "test"]
        train = df[(df.set == "train") & (df.validation.isnull())]
        val = df[df.validation == True]

        print("loaded train/val/test:", len(train), len(val), len(test))
        return train, val, test, max_length
    else:
        test = df[df.set == "test"]
        train = df[(df.set == "train")]
        print("loaded train/test:", len(train), len(test))
        return train, test, max_length


def load_esm_dataset(dataset, model, split, mean, mut_mean, flip, gb1_shorten=False):  # TODO: get updated version of function from FLIP

    # embedding_dir = Path("../../../FLIP/baselines/embeddings/")  # TODO: change to path in this repo / not hard-coded
    embedding_dir = Path("/home/kpg/microsoft/FLIP/baselines/embeddings/") # TODO: fix path (this one for debugging)
    PATH = embedding_dir / dataset / model / split
    print('loading ESM embeddings:', split)

    if mean:
        train = torch.load(PATH / 'train_mean.pt') #data_len x seq x 1280
        val = torch.load(PATH / 'val_mean.pt')
        test = torch.load(PATH / 'test_mean.pt') #data_len x seq x 1280
    else:
        train = torch.load(PATH / 'train_aa.pt') #data_len x seq x 1280
        val = torch.load(PATH / 'val_aa.pt')
        test = torch.load(PATH / 'test_aa.pt') #data_len x seq x 1280

        if dataset == 'gb1' and gb1_shorten == True: #fix the sequence to be shorter
            print('shortening gb1 to first 56 AAs')
            train = train[:, :56, :]
            val = val[:, :56, :]
            test = test[:, :56, :]
    
    if dataset == 'aav' and mut_mean == True:
        train = torch.mean(train[:, 560:590, :], 1)
        val = torch.mean(val[:, 560:590, :], 1)
        test = torch.mean(test[:, 560:590, :], 1)

    if dataset == 'gb1' and mut_mean == True: #positions 39, 40, 41, 54 in sequence
        train = torch.mean(train[:, [38, 39, 40, 53], :], 1)
        val = torch.mean(val[:, [38, 39, 40, 53], :], 1)
        test = torch.mean(test[:, [38, 39, 40, 53], :], 1)
    

    train_l = torch.load(PATH / 'train_labels.pt')
    val_l = torch.load(PATH / 'val_labels.pt')
    test_l = torch.load(PATH / 'test_labels.pt')

    if flip:
        train_l, test_l = test_l, train_l 
        train, test = test, train
   
    train_esm_data = TensorDataset(train, train_l)
    val_esm_data = TensorDataset(val, val_l)
    test_esm_data = TensorDataset(test, test_l)

    max_length = test.shape[1]

    print('loaded train/val/test:', len(train_esm_data), len(val_esm_data), len(test_esm_data), file = sys.stderr) 
    
    return train_esm_data, val_esm_data, test_esm_data, max_length


class SequenceDataset(Dataset):
    def __init__(self, data, dataset_name):
        self.data = data
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if self.dataset_name == "aav":
            return (
                row["sequence"][560:604],
                row["target"],
            )  # only look at part of sequence that changes
        elif self.dataset_name == "meltome":
            max_len = 1024  # truncate to first 1024 characters
            return row["sequence"][:max_len], row["target"]
        else:
            return row["sequence"], row["target"]


class ESMSequenceDataset(Dataset): #TODO: remove?
    "special dataset class just to deal with ESM tensors"

    def __init__(self, emb, mask, labels):
        self.emb = emb
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.emb[index], self.mask[index], self.labels[index]


class HugeDataset(Dataset):
    "load in the data directly from saved .pt files output from batch ESM. Include test/train in path"

    def __init__(self, embeddings_path, label_path, mean=False):
        self.path = embeddings_path
        self.label = torch.load(label_path)
        self.mean = mean

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        if self.mean:
            e = torch.load(self.path + str(index) + ".pt")["mean_representations"][33]
        else:
            e = torch.load(self.path + str(index) + ".pt")["representations"][33]

        return e, self.label[index]


def negative_log_likelihood(pred_targets, pred_var, targets):
    clamped_var = torch.clamp(pred_var, min=0.00001)
    loss = torch.log(clamped_var) / 2 + (pred_targets - targets) ** 2 / (
        2 * clamped_var
    )
    return torch.mean(loss)


def evidential_loss(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    L_NLL = nll  # torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return torch.mean(loss)


def evaluate_miscalibration_area(abs_error, uncertainty):
    standard_devs = abs_error / uncertainty
    probabilities = [
        2 * (stats.norm.cdf(standard_dev) - 0.5) for standard_dev in standard_devs
    ]
    sorted_probabilities = sorted(probabilities)

    fraction_under_thresholds = []
    threshold = 0

    for i in range(len(sorted_probabilities)):
        while sorted_probabilities[i] > threshold:
            fraction_under_thresholds.append(i / len(sorted_probabilities))
            threshold += 0.001

    # Condition used 1.0001 to catch floating point errors.
    while threshold < 1.0001:
        fraction_under_thresholds.append(1)
        threshold += 0.001

    thresholds = np.linspace(0, 1, num=1001)
    miscalibration = [
        np.abs(fraction_under_thresholds[i] - thresholds[i])
        for i in range(len(thresholds))
    ]
    miscalibration_area = 0
    for i in range(1, 1001):
        miscalibration_area += (
            np.average([miscalibration[i - 1], miscalibration[i]]) * 0.001
        )

    return {
        "fraction_under_thresholds": fraction_under_thresholds,
        "thresholds": thresholds,
        "miscalibration_area": miscalibration_area,
    }


def evaluate_log_likelihood(error, uncertainty):
    log_likelihood = 0
    optimal_log_likelihood = 0

    for err, unc in zip(error, uncertainty):
        # Encourage small standard deviations.
        log_likelihood -= np.log(2 * np.pi * max(0.00001, unc ** 2)) / 2
        optimal_log_likelihood -= np.log(2 * np.pi * err ** 2) / 2

        # Penalize for large error.
        log_likelihood -= err ** 2 / (2 * max(0.00001, unc ** 2))
        optimal_log_likelihood -= 1 / 2

    return {
        "log_likelihood": log_likelihood,
        "optimal_log_likelihood": optimal_log_likelihood,
        "average_log_likelihood": log_likelihood / len(error),
        "average_optimal_log_likelihood": optimal_log_likelihood / len(error),
    }


def calculate_metrics(
    y_test,
    preds_mean,
    preds_std,
    args,
    split,
    y_train,
    algorithm_type,
    evidential=False,
    out_fpath=None,
):
    if out_fpath is None:
        out_fpath = Path.cwd()

    rho = stats.spearmanr(y_test, preds_mean).correlation
    rmse = mean_squared_error(y_test, preds_mean, squared=False)
    mae = mean_absolute_error(y_test, preds_mean)
    r2 = r2_score(y_test, preds_mean)

    print("TEST RHO: ", rho)
    print("TEST RMSE: ", rmse)
    print("TEST MAE: ", mae)
    print("TEST R2: ", r2)

    residual = np.abs(y_test - preds_mean)

    df = pd.DataFrame()
    df["y_test"] = y_test
    df["preds_mean"] = preds_mean
    df["residual"] = residual

    if evidential:
        aleatoric_unc = preds_std[:, 0]
        epistemic_unc = preds_std[:, 1]
        total_unc = aleatoric_unc + epistemic_unc
        metrics = [rho, rmse, mae, r2]
        for name, preds_std in zip(
            ["aleatoric", "epistemic", "total"],
            [aleatoric_unc, epistemic_unc, total_unc],
        ):
            coverage = residual < 2 * preds_std
            width_range = 4 * preds_std / (max(y_train) - min(y_train))
            df[f"preds_std ({name})"] = preds_std
            df[f"coverage ({name})"] = coverage
            df[f"width/range ({name})"] = width_range
            rho_unc, p_rho_unc = stats.spearmanr(
                df["residual"], df[f"preds_std ({name})"]
            )
            percent_coverage = sum(df[f"coverage ({name})"]) / len(df)
            average_width_range = df[f"width/range ({name})"].mean() / (
                max(y_train) - min(y_train)
            )
            miscalibration_area_results = evaluate_miscalibration_area(
                df["residual"], df[f"preds_std ({name})"]
            )
            miscalibration_area = miscalibration_area_results["miscalibration_area"]
            ll_results = evaluate_log_likelihood(
                df["residual"], df[f"preds_std ({name})"]
            )
            average_log_likelihood = ll_results["average_log_likelihood"]
            average_optimal_log_likelihood = ll_results[
                "average_optimal_log_likelihood"
            ]
            print(f"TEST RHO UNCERTAINTY ({name}): ", rho_unc)
            print(f"TEST RHO UNCERTAINTY P-VALUE ({name}): ", p_rho_unc)
            print(f"PERCENT COVERAGE ({name}): ", percent_coverage)
            print(f"AVERAGE WIDTH / TRAINING SET RANGE ({name}): ", average_width_range)
            print(f"MISCALIBRATION AREA ({name}): ", miscalibration_area)
            print(f"AVERAGE NLL ({name}): ", average_log_likelihood)
            print(f"AVERAGE OPTIMAL NLL ({name}): ", average_optimal_log_likelihood)
            print(
                f"NLL / NLL_OPT ({name}):",
                average_log_likelihood / average_optimal_log_likelihood,
            )
            metrics.extend(
                [
                    rho_unc,
                    p_rho_unc,
                    percent_coverage,
                    average_width_range,
                    miscalibration_area,
                    average_log_likelihood,
                    average_optimal_log_likelihood,
                    average_log_likelihood / average_optimal_log_likelihood,
                    args.dropout,
                ]
            )
    else:
        coverage = residual < 2 * preds_std
        width_range = 4 * preds_std / (max(y_train) - min(y_train))
        df["preds_std"] = preds_std
        df["coverage"] = coverage
        df["width/range"] = width_range
        rho_unc, p_rho_unc = stats.spearmanr(df["residual"], df["preds_std"])
        percent_coverage = sum(df["coverage"]) / len(df)
        average_width_range = df["width/range"].mean() / (max(y_train) - min(y_train))
        miscalibration_area_results = evaluate_miscalibration_area(
            df["residual"], df["preds_std"]
        )
        miscalibration_area = miscalibration_area_results["miscalibration_area"]
        ll_results = evaluate_log_likelihood(df["residual"], df["preds_std"])
        average_log_likelihood = ll_results["average_log_likelihood"]
        average_optimal_log_likelihood = ll_results["average_optimal_log_likelihood"]
        print("TEST RHO UNCERTAINTY: ", rho_unc)
        print("TEST RHO UNCERTAINTY P-VALUE: ", p_rho_unc)
        print("PERCENT COVERAGE: ", percent_coverage)
        print("AVERAGE WIDTH / TRAINING SET RANGE: ", average_width_range)
        print("MISCALIBRATION AREA: ", miscalibration_area)
        print("AVERAGE NLL: ", average_log_likelihood)
        print("AVERAGE OPTIMAL NLL: ", average_optimal_log_likelihood)
        print("NLL / NLL_OPT:", average_log_likelihood / average_optimal_log_likelihood)
        metrics = [
            rho,
            rmse,
            mae,
            r2,
            rho_unc,
            p_rho_unc,
            percent_coverage,
            average_width_range,
            miscalibration_area,
            average_log_likelihood,
            average_optimal_log_likelihood,
            average_log_likelihood / average_optimal_log_likelihood,
        ]

    df.to_csv(
        f"{out_fpath}/evals_new/{args.dataset}_{algorithm_type}_{split}_test_preds.csv",
        index=False,
    )

    return metrics
