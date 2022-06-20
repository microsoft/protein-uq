import argparse
import random
import re
import sys
from csv import writer
from pathlib import Path

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from evals import evaluate_cnn, evaluate_gp, evaluate_ridge
from filepaths import BASELINE_DIR, RESULTS_DIR
from models import BayesianRidgeRegression, ExactGPModel, FluorescenceModel
from train import train_cnn, train_gp, train_ridge
from utils import ASCollater, SequenceDataset, Tokenizer, get_data, load_dataset, load_esm_dataset, vocab

sys.path.append(BASELINE_DIR)

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
    parser.add_argument("--representation", choices=["ohe", "esm"], type=str)
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--uncertainty",
        choices=["ridge", "gp", "dropout", "ensemble", "mve", "evidential", "svi"],
        type=str,
    )
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--flip", action="store_true")  # for flipping mut-des and des-mut
    parser.add_argument("--gb1_shorten", action="store_true")
    # ESM
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--mut_mean", action="store_true")
    # CNN hyperparameters
    parser.add_argument("--lr", type=float, default=0.001)
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

    return parser


def train_eval(
    dataset,
    model,
    representation,
    uncertainty,
    split,
    device,
    scale,
    mean,
    mut_mean,
    batch_size,
    flip,
    lr,
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
):

    results_dir = Path(RESULTS_DIR)
    EVAL_PATH = results_dir / dataset / model / split
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    # if mean:
    #     model += "_mean"  # TODO: if including this, need to modify below blocks to have .startswith()
    # if mut_mean:
    #     model += "_mut_mean"
    # if flip:
    #     split += "_flipped"

    # load data
    if representation == "esm":
        if model == "cnn":
            train_data, val_data, test_data, _ = load_esm_dataset(
                dataset, model, split, mean, mut_mean, flip, gb1_shorten=gb1_shorten
            )
            # TODO: add transformations for CNN ESM
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
    if scale:
        scaler = StandardScaler()
        train_seq = scaler.fit_transform(train_seq)
        test_seq = scaler.transform(test_seq)
        train_target = scaler.fit_transform(np.array(train_target)[:, None])[:, 0]
        test_target = scaler.transform(np.array(test_target)[:, None])[:, 0]
    # TODO: unscale before evaluating

    if model in ["ridge", "gp"]:
        lr = kernel_size = input_size = dropout = ""  # get rid of unused variables
        np.random.seed(0)
        torch.manual_seed(1)

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
        # TODO: add more to evaluation function based on my calculate_metrics function
        train_rho, train_mse = evaluate_ridge(train_seq, train_target, lr_trained, EVAL_PATH / "train")  # evaluate on train
        test_rho, test_mse = evaluate_ridge(test_seq, test_target, lr_trained, EVAL_PATH / "test")  # evaluate on test

    if model == "gp":
        train_seq, train_target = torch.tensor(train_seq).float(), torch.tensor(train_target).float()
        test_seq, test_target = torch.tensor(test_seq).float(), torch.tensor(test_target).float()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_seq, train_target, likelihood, device_ids=gpu)
        model.covar_module.module.base_kernel.lengthscale *= length
        gp_trained, _ = train_gp(train_seq, train_target, model, device, length, size)

        train_rho, train_mse = evaluate_gp(train_seq, train_target, lr_trained, EVAL_PATH / "train")  # evaluate on train
        test_rho, test_mse = evaluate_gp(test_seq, test_target, lr_trained, EVAL_PATH / "test")  # evaluate on test

    if model == "cnn":
        lr = alpha = ""  # get rid of unused variables
        collate = ASCollater(vocab, Tokenizer(vocab), pad=True)
        if dataset == "meltome":
            batch_size = 30  # smaller batch sizes for meltome since seqs are long
        train_iterator = DataLoader(
            SequenceDataset(train),
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_iterator = DataLoader(
            SequenceDataset(val),
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_iterator = DataLoader(
            SequenceDataset(test),
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        # initialize model
        cnn_model = FluorescenceModel(len(vocab), kernel_size, input_size, dropout)
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
        criterion = nn.MSELoss()
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
        )
        # evaluate
        train_rho, train_mse = evaluate_cnn(train_iterator, cnn_model, device, EVAL_PATH, EVAL_PATH / "train")
        test_rho, test_mse = evaluate_cnn(test_iterator, cnn_model, device, EVAL_PATH, EVAL_PATH / "test")

    print("done training and testing: dataset: {0} model: {1} split: {2} \n".format(dataset, model, split))
    print("full results saved at: ", EVAL_PATH)
    print("train stats: Spearman: %.2f MSE: %.2f " % (train_rho, train_mse))
    print("test stats: Spearman: %.2f MSE: %.2f " % (test_rho, test_mse))

    with open(results_dir / (dataset + "_results.csv"), "a", newline="") as f:
        writer(f).writerow(
            [
                dataset,
                model,
                uncertainty,
                split,
                train_rho,
                train_mse,
                test_rho,
                test_mse,
                dropout,
            ]
        )


def main(args):
    device = torch.device("cpu")
    device = torch.device("cuda:%d" % args.gpu[0])
    split = split_dict[args.split]
    dataset = re.findall(r"(\w*)\_", args.split)[0]

    print("dataset: {0} model: {1} split: {2} \n".format(dataset, args.model, split))

    random.seed(10)
    torch.manual_seed(10)
    train_eval(
        dataset,
        args.model,
        args.representation,
        args.uncertainty,
        split,
        device,
        args.scale,
        args.mean,
        args.mut_mean,
        256,
        args.flip,
        args.lr,
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
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if (args.uncertainty in ["dropout", "ensemble", "mve", "evidential", "svi"]) and (args.model != "cnn"):
        raise ValueError("The uncertainty method you selected only works with CNN.")

    main(args)
