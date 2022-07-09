import argparse
import functools
import random
import re
from csv import writer
from pathlib import Path

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from evals import evaluate_cnn, evaluate_gp, evaluate_ridge
from models import BayesianRidgeRegression, ExactGPModel, FluorescenceModel
from train import train_cnn, train_gp, train_ridge
from utils import (ASCollater, ESMSequenceMeanDataset, SequenceDataset,
                   Tokenizer, det_loss, evidential_loss, get_data, load_dataset, load_esm_dataset,
                   negative_log_likelihood, vocab)

split_dict = {
    "aav_1": "des_mut",
    "aav_2": "mut_des",
    "aav_3": "one_vs_many",
    "aav_4": "two_vs_many",
    "aav_5": "seven_vs_many",
    "aav_6": "low_vs_high",
    "aav_7": "sampled",
    "meltome_1": "mixed_split",
    "meltome_2": "human",
    "meltome_3": "human_cell",
    "gb1_1": "one_vs_rest",
    "gb1_2": "two_vs_rest",
    "gb1_3": "three_vs_rest",
    "gb1_4": "sampled",
    "gb1_5": "low_vs_high",
}


def create_parser():
    parser = argparse.ArgumentParser(description="train esm")
    # General
    parser.add_argument("--split", type=str)
    parser.add_argument("--model", choices=["ridge", "gp", "cnn"], type=str)
    parser.add_argument("--representation", choices=["ohe", "esm"], type=str)  # TODO: separate into esm_mean and esm_full
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--uncertainty",
        choices=["ridge", "gp", "dropout", "ensemble", "mve", "evidential", "svi"],
        type=str,
    )
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--results_dir", type=str, default="test_results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--flip", action="store_true")  # for flipping mut-des and des-mut
    parser.add_argument("--gb1_shorten", action="store_true")
    # ESM
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--mut_mean", action="store_true")
    # CNN hyperparameters
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Ridge hyperparameters
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--alpha_1", type=float, default=1e-6)
    parser.add_argument("--alpha_2", type=float, default=1e-6)
    parser.add_argument("--lambda_1", type=float, default=1e-6)
    parser.add_argument("--lambda_2", type=float, default=1e-6)
    # GP hyperparameters
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--regularizer_coeff", type=float, default=1.0)

    return parser


def train_eval(
    dataset,
    model,
    representation,
    uncertainty,
    split,
    device,
    scale,
    results_dir,
    mean,
    mut_mean,
    batch_size,
    flip,
    kernel_size,
    input_size,
    dropout,
    gb1_shorten,
    max_iter,
    tol,
    alpha_1,
    alpha_2,
    lambda_1,
    lambda_2,
    size,
    length,
    gpu,
    regularizer_coeff,
):

    results_dir = Path(results_dir)
    EVAL_PATH = results_dir / dataset / split / model / representation / uncertainty
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    # load data
    if representation == "esm":
        if model == "cnn":
            train, val, test, _ = load_esm_dataset(
                dataset, model, split, mean, mut_mean, flip, gb1_shorten=gb1_shorten
            )
        else:
            train, _, test, max_length = load_esm_dataset(dataset, model, split, mean, mut_mean, flip, gb1_shorten=gb1_shorten)
            train_seq = np.array([i[0].numpy() for i in train]).squeeze()
            train_target = [i[1].item() for i in train]
            test_seq = np.array([i[0].numpy() for i in test]).squeeze()
            test_target = [i[1].item() for i in test]
    elif representation == "ohe":
        if model == "cnn":
            train, val, test, _ = load_dataset(dataset, split + ".csv", gb1_shorten=gb1_shorten)
        else:
            train, test, max_length = load_dataset(dataset, split + ".csv", val_split=False, gb1_shorten=gb1_shorten)
            train_seq, train_target = get_data(train, max_length, encode_pad=False, one_hots=True)
            test_seq, test_target = get_data(test, max_length, encode_pad=False, one_hots=True)

    # scale data
    if scale and (model in ["ridge", "gp"]):
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        train_seq = x_scaler.fit_transform(train_seq)
        test_seq = x_scaler.transform(test_seq)
        train_target = y_scaler.fit_transform(np.array(train_target)[:, None])[:, 0]
        test_target = y_scaler.transform(np.array(test_target)[:, None])[:, 0]
    elif scale and (model == "cnn") and (representation == "esm"):
        X_train_mean = train.tensors[0].mean(axis=0)
        X_train_std = train.tensors[0].std(axis=0)
        y_train_mean = train.tensors[1].mean(axis=0)
        y_train_std = train.tensors[1].std(axis=0)

        train_seq_new = (train.tensors[0] - X_train_mean) / X_train_std
        train_target_new = (train.tensors[1] - y_train_mean) / y_train_std
        train = TensorDataset(train_seq_new, train_target_new)
        val_seq_new = (val.tensors[0] - X_train_mean) / X_train_std
        val_target_new = (val.tensors[1] - y_train_mean) / y_train_std
        val = TensorDataset(val_seq_new, val_target_new)
        test_seq_new = (test.tensors[0] - X_train_mean) / X_train_std
        test_target_new = (test.tensors[1] - y_train_mean) / y_train_std
        test = TensorDataset(test_seq_new, test_target_new)

        x_scaler = (X_train_mean, X_train_std)
        y_scaler = (y_train_mean, y_train_std)
    elif scale and (model == "cnn") and (representation == "ohe"):
        x_scaler = None
        y_scaler = StandardScaler()
        train.target = y_scaler.fit_transform(train.target.values.reshape(-1, 1))
        val.target = y_scaler.fit_transform(val.target.values.reshape(-1, 1))
        test.target = y_scaler.transform(test.target.values.reshape(-1, 1))
    else:
        x_scaler = None
        y_scaler = None

    if model in ["ridge", "gp"]:
        kernel_size = input_size = dropout = ""  # get rid of unused variables

    # train and evaluate models
    if model == "ridge":
        lr_model = BayesianRidgeRegression(
            max_iter,
            tol,
            alpha_1,
            alpha_2,
            lambda_1,
            lambda_2,
        )  # initialize model
        lr_trained, _ = train_ridge(train_seq, train_target, lr_model)  # train and pass back trained model
        train_rho, train_rmse, train_mae, train_r2 = evaluate_ridge(train_seq, train_target, lr_trained, EVAL_PATH / "train", y_scaler)  # evaluate on train
        test_rho, test_rmse, test_mae, test_r2 = evaluate_ridge(test_seq, test_target, lr_trained, EVAL_PATH / "test", y_scaler)  # evaluate on test

    if model == "gp":
        train_seq, train_target = torch.tensor(train_seq).float(), torch.tensor(train_target).float()
        test_seq, test_target = torch.tensor(test_seq).float(), torch.tensor(test_target).float()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = ExactGPModel(train_seq, train_target, likelihood, device_ids=gpu)
        gp_model.covar_module.module.base_kernel.lengthscale *= length
        gp_trained, _ = train_gp(train_seq, train_target, gp_model, likelihood, device, length, size)

        train_rho, train_rmse, train_mae, train_r2 = evaluate_gp(train_seq, train_target, gp_trained, likelihood, device, size, EVAL_PATH / "train", y_scaler)  # evaluate on train
        test_rho, test_rmse, test_mae, test_r2 = evaluate_gp(test_seq, test_target, gp_trained, likelihood, device, size, EVAL_PATH / "test", y_scaler)  # evaluate on test

    if model == "cnn":
        if representation == "ohe":
            cnn_input_type = "ohe"
        if representation == "esm":
            cnn_input_type = "esm_mean"  # TODO: separate into esm_mean and esm_full
            input_size = 1280  # size of ESM mean embeddings is fixed and different from 1024 default for OHE

        if dataset == "meltome":
            batch_size = 30  # smaller batch sizes for meltome since seqs are long
        if representation == "ohe":
            collate = ASCollater(vocab, Tokenizer(vocab), pad=True)
            train_iterator = DataLoader(
                SequenceDataset(train, dataset),
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            val_iterator = DataLoader(
                SequenceDataset(val, dataset),
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            test_iterator = DataLoader(
                SequenceDataset(test, dataset),
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
        elif representation == "esm":  # TODO: separate into esm_mean and esm_full
            train_iterator = DataLoader(
                ESMSequenceMeanDataset(train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            val_iterator = DataLoader(
                ESMSequenceMeanDataset(val),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            test_iterator = DataLoader(
                ESMSequenceMeanDataset(test),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
        # initialize model (always use dropout = 0.0 for training)
        cnn_model = FluorescenceModel(len(vocab), kernel_size, input_size, 0.0, input_type=cnn_input_type,
                                      mve=uncertainty == "mve", evidential=uncertainty == "evidential",
                                      svi=uncertainty == "svi", n_batches=1)  # TODO: fix n_batches
        if uncertainty == "mve":
            criterion = negative_log_likelihood
        elif uncertainty == "evidential":
            criterion = functools.partial(evidential_loss, lam=regularizer_coeff)
        elif uncertainty == "svi":
            criterion = det_loss
        else:
            criterion = nn.MSELoss()
        # create optimizer and loss function
        optimizer = optim.Adam(
            [
                {
                    "params": cnn_model.encoder.parameters(),
                    "lr": 1e-3,
                    "weight_decay": 0,
                },
                {
                    "params": cnn_model.embedding.parameters(),
                    "lr": 5e-5,
                    "weight_decay": 0.05,
                },
                {
                    "params": cnn_model.decoder.parameters(),
                    "lr": 5e-6,
                    "weight_decay": 0.05,
                },
            ]
        )
        # train and pass back epochs trained - for CNN, save model
        epochs_trained = train_cnn(
            train_iterator,
            val_iterator,
            cnn_model,
            device,
            criterion,
            optimizer,
            100,
            EVAL_PATH,
            mve=uncertainty == "mve",
            evidential=uncertainty == "evidential",
            svi=uncertainty == "svi",
        )

        # evaluate
        train_rho, train_rmse, train_mae, train_r2 = evaluate_cnn(train_iterator, cnn_model, device, EVAL_PATH, EVAL_PATH / "train", y_scaler, dropout=dropout,
                                                                  mve=uncertainty == "mve", evidential=uncertainty == "evidential", svi=uncertainty == "svi")
        test_rho, test_rmse, test_mae, test_r2 = evaluate_cnn(test_iterator, cnn_model, device, EVAL_PATH, EVAL_PATH / "test", y_scaler, dropout=dropout,
                                                              mve=uncertainty == "mve", evidential=uncertainty == "evidential", svi=uncertainty == "svi")

    print("done training and testing: dataset: {0} model: {1} split: {2} \n".format(dataset, model, split))
    print("full results saved at: ", EVAL_PATH)
    print(f"train stats: Spearman: {train_rho:.2f} RMSE: {train_rmse:.2f} MAE: {train_mae:.2f} R2: {train_r2:.2f}")
    print(f"test stats: Spearman: {test_rho:.2f} RMSE: {test_rmse:.2f} MAE: {test_mae:.2f} R2: {test_r2:.2f}")

    # TODO: make sure all outputs in good format to read in for auto-generating plots

    with open(results_dir / (dataset + "_results.csv"), "a", newline="") as f:
        writer(f).writerow(
            [
                dataset,
                split,
                model,
                uncertainty,
                dropout,
                train_rho,
                train_rmse,
                train_mae,
                train_r2,
                test_rho,
                test_rmse,
                test_mae,
                test_r2,
            ]
        )


def main(args):
    device = torch.device("cpu")
    device = torch.device("cuda:%d" % args.gpu[0])
    split = split_dict[args.split]
    dataset = re.findall(r"(\w*)\_", args.split)[0]

    print("dataset: {0} model: {1} split: {2} \n".format(dataset, args.model, split))

    if args.uncertainty == "ensemble":
        for i in range(5):
            random.seed(i)
            torch.manual_seed(i)
            train_eval(
                dataset,
                args.model,
                args.representation,
                args.uncertainty,
                split,
                device,
                args.scale,
                args.results_dir,
                args.mean,
                args.mut_mean,
                args.batch_size,
                args.flip,
                args.kernel_size,
                args.input_size,
                args.dropout,
                args.gb1_shorten,
                args.max_iter,
                args.tol,
                args.alpha_1,
                args.alpha_2,
                args.lambda_1,
                args.lambda_2,
                args.size,
                args.length,
                args.gpu,
            )  # TODO: call special CNN eval function for ensemble
    else:
        random.seed(0)
        torch.manual_seed(0)
        train_eval(
            dataset,
            args.model,
            args.representation,
            args.uncertainty,
            split,
            device,
            args.scale,
            args.results_dir,
            args.mean,
            args.mut_mean,
            args.batch_size,
            args.flip,
            args.kernel_size,
            args.input_size,
            args.dropout,
            args.gb1_shorten,
            args.max_iter,
            args.tol,
            args.alpha_1,
            args.alpha_2,
            args.lambda_1,
            args.lambda_2,
            args.size,
            args.length,
            args.gpu,
            args.regularizer_coeff,
        )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if (args.uncertainty in ["dropout", "ensemble", "mve", "evidential", "svi"]) and (args.model != "cnn"):
        raise ValueError("The uncertainty method you selected only works with CNN.")
    if (args.uncertainty in ["ridge", "gp"]) and (args.model == "cnn"):
        raise ValueError("The uncertainty method you selected doesn't work with CNN.")
    if (args.uncertainty == "dropout") and (args.dropout == 0.0):
        raise ValueError("Dropout uncertainty requires dropout to be non-zero.")
    if (args.uncertainty != "dropout") and (args.dropout != 0.0):
        raise ValueError("Cannot use nonzero dropout with non-dropout uncertainty.")

    main(args)  # TODO: remove X_ files, lint/format all files, make config files or "make" files
