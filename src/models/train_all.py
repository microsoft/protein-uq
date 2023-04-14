import argparse
import random
import re
from csv import writer
from pathlib import Path

import numpy as np
import torch

from evals import eval_model
from train import train_model
from utils import load_and_scale_data

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
    parser.add_argument(
        "--representation", choices=["ohe", "esm"], type=str
    )  # TODO: separate into esm_mean and esm_full
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--uncertainty",
        choices=["ridge", "gp", "dropout", "ensemble", "mve", "evidential", "svi"],
        type=str,
    )
    parser.add_argument("--seed", type=int, default=0)  # cross-validation with different random initializations of weights
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--flip", action="store_true"
    )  # for flipping mut-des and des-mut
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
    seed,
):

    results_dir = Path(results_dir)
    EVAL_PATH = results_dir / dataset / split / model / representation / uncertainty / f"cv_fold_{int(seed)}"
    EVAL_PATH.mkdir(parents=True, exist_ok=True)

    # load data
    (
        train,
        val,
        test,
        max_length,
        x_scaler,
        y_scaler,
        train_seq,
        train_target,
        test_seq,
        test_target,
    ) = load_and_scale_data(
        representation, model, dataset, split, mean, mut_mean, flip, gb1_shorten, scale
    )

    if model in ["ridge", "gp"]:
        kernel_size = input_size = dropout = ""  # get rid of unused variables

    # train and evaluate models
    (
        lr_trained,
        gp_trained,
        EVAL_PATH_BASE,
        likelihood,
        train_seq,
        train_target,
        test_seq,
        test_target,
        train_labels,
        train_out_mean,
        train_out_std,
        test_labels,
        test_out_mean,
        test_out_std,
    ) = train_model(
        model,
        representation,
        dataset,
        uncertainty,
        EVAL_PATH,
        max_iter,
        tol,
        alpha_1,
        alpha_2,
        lambda_1,
        lambda_2,
        size,
        length,
        regularizer_coeff,
        kernel_size,
        dropout,
        gpu,
        device,
        y_scaler,
        batch_size,
        input_size,
        train_seq,
        train_target,
        test_seq,
        test_target,
        train,
        val,
        test,
        cv_seed=seed,
    )
    (
        train_rho,
        train_rmse,
        train_mae,
        train_r2,
        train_unc_metrics,
        test_rho,
        test_rmse,
        test_mae,
        test_r2,
        test_unc_metrics,
    ) = eval_model(
        model,
        y_scaler,
        train_seq=train_seq,
        train_target=train_target,
        test_seq=test_seq,
        test_target=test_target,
        lr_trained=lr_trained,
        gp_trained=gp_trained,
        EVAL_PATH=EVAL_PATH,
        EVAL_PATH_BASE=EVAL_PATH_BASE,
        likelihood=likelihood,
        device=device,
        size=size,
        train_labels=train_labels,
        train_out_mean=train_out_mean,
        train_out_std=train_out_std,
        test_labels=test_labels,
        test_out_mean=test_out_mean,
        test_out_std=test_out_std,
    )

    print(
        "done training and testing: dataset: {0} model: {1} split: {2} \n".format(
            dataset, model, split
        )
    )
    print("full results saved at: ", EVAL_PATH)
    print(
        f"train stats: Spearman: {train_rho:.2f} RMSE: {train_rmse:.2f} MAE: {train_mae:.2f} R2: {train_r2:.2f}"
    )
    print(
        f"test stats: Spearman: {test_rho:.2f} RMSE: {test_rmse:.2f} MAE: {test_mae:.2f} R2: {test_r2:.2f}"
    )

    (
        train_rho_unc,
        train_p_rho_unc,
        train_percent_coverage,
        train_average_width_range,
        train_miscalibration_area,
        train_average_nll,
        train_average_optimal_nll,
        train_average_nll_ratio,
    ) = train_unc_metrics
    (
        test_rho_unc,
        test_p_rho_unc,
        test_percent_coverage,
        test_average_width_range,
        test_miscalibration_area,
        test_average_nll,
        test_average_optimal_nll,
        test_average_nll_ratio,
    ) = test_unc_metrics

    with open(results_dir / (dataset + "_results.csv"), "a", newline="") as f:
        writer(f).writerow(
            [
                dataset,
                split,
                representation,
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
                train_rho_unc,
                train_p_rho_unc,
                train_percent_coverage,
                train_average_width_range,
                train_miscalibration_area,
                train_average_nll,
                train_average_optimal_nll,
                train_average_nll_ratio,
                test_rho_unc,
                test_p_rho_unc,
                test_percent_coverage,
                test_average_width_range,
                test_miscalibration_area,
                test_average_nll,
                test_average_optimal_nll,
                test_average_nll_ratio,
                seed,
            ]
        )


def main(args):
    device = torch.device("cpu")
    device = torch.device("cuda:%d" % args.gpu[0])
    split = split_dict[args.split]
    dataset = re.findall(r"(\w*)\_", args.split)[0]

    print("dataset: {0} model: {1} split: {2} \n".format(dataset, args.model, split))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        args.seed,
    )

    print("Max memory allocated: ", torch.cuda.max_memory_allocated(device=device), "bytes")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if (args.uncertainty in ["dropout", "ensemble", "mve", "evidential", "svi"]) and (
        args.model != "cnn"
    ):
        raise ValueError("The uncertainty method you selected only works with CNN.")
    if (args.uncertainty in ["ridge", "gp"]) and (args.model == "cnn"):
        raise ValueError("The uncertainty method you selected doesn't work with CNN.")
    if (args.uncertainty == "dropout") and (args.dropout == 0.0):
        raise ValueError("Dropout uncertainty requires dropout to be non-zero.")
    if (args.uncertainty != "dropout") and (args.dropout != 0.0):
        raise ValueError("Cannot use nonzero dropout with non-dropout uncertainty.")

    main(args)
