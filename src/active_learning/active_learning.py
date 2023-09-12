"""Train a model using active learning on a dataset."""
import argparse
import datetime
import os
import random
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

sys.path.append("../models/")
from evals import eval_model
from train import train_model
from train_all import split_dict
from utils import load_and_scale_data


def ordered_list_diff(a, b):
    """Returns the elements of a without any elements of b, while preserving
        order.

    :param a: list or array to remove elements from
    :param b: list or array to find and remove
    """
    return a[~np.in1d(a, b)]


def create_parser():
    parser = argparse.ArgumentParser(description="active learning")
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
    parser.add_argument("--results_dir", type=str, default="al_results")
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
    # Active Learning Arguments
    parser.add_argument(
        "--num_folds",
        type=int,
        default=1,
        help="Number of cross-validation folds to do",  # number of different initial training datasets
    )
    parser.add_argument(
        "--al_init_ratio",
        type=float,
        default=0.1,
        help="Percent of training data to use on first active learning iteration",
    )
    parser.add_argument(
        "--al_end_ratio",
        type=float,
        default=None,
        help="Fraction of total data To stop active learning early. By default, explore full train data",
    )
    parser.add_argument(
        "--num_al_loops",
        type=int,
        default=20,
        help="Number of active learning loops to add new data",
    )
    parser.add_argument(
        "--al_topk",
        type=int,
        default=1000,
        help="Top-K acquired molecules to consider during active learning",
    )
    parser.add_argument(
        "--al_std_mult",
        type=float,
        default=1,
        help="Multiplier for std in lcb acquisition",
    )
    parser.add_argument(
        "--al_step_scale",
        type=str,
        default="log",
        choices=["log", "linear", "single"],
        help="scale of spacing for active learning steps (log or linear). `Single` samples one additional point per step",
    )
    parser.add_argument(
        "--acquire_min",
        action="store_true",
        help="if we should acquire min or max score molecules",
    )
    parser.add_argument(
        "--al_strategy",
        type=str,
        nargs="+",
        choices=[
            "random",  # *
            "explorative_greedy",  # *
            "explorative_sample",  # *
            "score_greedy",
            "score_sample",
            "exploit",  # true greedy
            "exploit_ucb",
            "exploit_lcb",
            "exploit_ts",
        ],
        default=["explorative_greedy"],
        help="Strategy for active learning regime",
    )
    return parser


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

    device = torch.device("cpu")
    device = torch.device("cuda:%d" % args.gpu[0])
    split = split_dict[args.split]
    dataset = re.findall(r"(\w*)\_", args.split)[0]

    print("dataset: {0} model: {1} split: {2} \n".format(dataset, args.model, split))

    results_root = Path(
        f"{args.results_dir}/{dataset}/{split}/{args.model}/{args.representation}/{args.uncertainty}"
    )
    if args.uncertainty == "dropout":
        results_root = results_root / f"dropout{args.dropout}"
    results_root.mkdir(parents=True, exist_ok=True)

    for i_trial in range(args.num_folds):
        df = pd.DataFrame(
            columns=[
                "Strategy",
                "Trial",
                "Train Data Ratio",
                "TopKPercentOverlap",
                "TestRho",
                "TestRMSE",
                "TestMAE",
                "TestR2",
                "MeanUncertainty",
                "train_rho_unc",
                "train_p_rho_unc",
                "train_percent_coverage",
                "train_average_width_range",
                "train_miscalibration_area",
                "train_average_nll",
                "train_average_optimal_nll",
                "train_average_nll_ratio",
                "test_rho_unc",
                "test_p_rho_unc",
                "test_percent_coverage",
                "test_average_width_range",
                "test_miscalibration_area",
                "test_average_nll",
                "test_average_optimal_nll",
                "test_average_nll_ratio",
                "best_sample_in_train",
                "best_sample_aquired",
            ]
        )

        # Load the data
        (
            train,
            val,
            test,
            max_length,
            x_scaler,
            scaler,
            train_seq,
            train_target,
            test_seq,
            test_target,
        ) = load_and_scale_data(
            args.representation,
            args.model,
            dataset,
            split,
            args.mean,
            args.mut_mean,
            args.flip,
            args.gb1_shorten,
            args.scale,
        )

        if args.model == "cnn":
            all_train_data = train
            val_data = val
            test_data = test
        else:
            all_train_data = train_seq
            all_train_seq = train_seq
            all_train_target = train_target
            val_data = test_seq  # val data not used for GP or ridge
            test_data = test_seq

        # Define active learning step variables and subsample the tasks
        n_total = len(all_train_data)
        n_sample = n_total
        n_loops = args.num_al_loops

        # Change active learning n_sample for early stopping
        if args.al_end_ratio is not None:
            if args.al_end_ratio > 1:
                raise ValueError("Arg al_end_ratio must be less than train size")
            total_data = len(all_train_data) + len(val_data) + len(test_data)
            early_stop_num = int(n_total * args.al_end_ratio)
            n_sample = early_stop_num

        n_start = int(n_total * args.al_init_ratio)

        # Compute the number of samples to use at each step of active learning
        if args.al_step_scale == "linear":
            n_samples_per_run = np.linspace(n_start, n_sample, n_loops)
        elif args.al_step_scale == "log":
            n_samples_per_run = np.logspace(
                np.log10(n_start), np.log10(n_sample), n_loops
            )
        elif args.al_step_scale == "single":
            n_samples_per_run = range(n_start, n_sample)
            print("Ignoring input for n_loops, using n_loops = n_sample - n_start")
            n_loops = n_sample - n_start
            thirty_percent_of_train = int(n_total * 0.3)
        n_samples_per_run = np.round(n_samples_per_run).astype(int)

        np.random.seed(i_trial)
        random.seed(i_trial)
        torch.manual_seed(i_trial)
        train_subset_inds_start = np.random.choice(n_total, n_start, replace=False)
        for strategy in args.al_strategy:
            train_subset_inds = np.copy(train_subset_inds_start)

            tic_time = time.time()  # grab the current time for logging

            # Main active learning loop
            for i in range(n_loops):
                print(
                    f"===> [{strategy}] Running trial {i_trial} with {n_samples_per_run[i]} samples"
                )

                # need to sort indices to ensure training process same for same data in different order
                if args.model == "cnn" and args.representation == "ohe":
                    train_data = all_train_data.iloc[np.sort(train_subset_inds)]
                elif args.model == "cnn" and args.representation == "esm":
                    train_data = TensorDataset(
                        all_train_data.tensors[0][np.sort(train_subset_inds)],
                        all_train_data.tensors[1][np.sort(train_subset_inds)],
                    )
                else:
                    train_data = all_train_data[np.sort(train_subset_inds)]
                    train_seq = all_train_seq[np.sort(train_subset_inds)]
                    train_target = all_train_target[np.sort(train_subset_inds)]

                EVAL_PATH = results_root / strategy / str(i_trial) / str(i)
                EVAL_PATH.mkdir(parents=True, exist_ok=True)

                # Train with the data subset, return the best models
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
                    al_full_train_out_mean,
                    al_full_train_out_std,
                ) = train_model(
                    args.model,
                    args.representation,
                    dataset,
                    args.uncertainty,
                    EVAL_PATH,
                    args.max_iter,
                    args.tol,
                    args.alpha_1,
                    args.alpha_2,
                    args.lambda_1,
                    args.lambda_2,
                    args.size,
                    args.length,
                    args.regularizer_coeff,
                    args.kernel_size,
                    args.dropout,
                    args.gpu,
                    device,
                    scaler,
                    args.batch_size,
                    args.input_size,
                    train_seq,
                    train_target,
                    test_seq,
                    test_target,
                    train_data,
                    val_data,
                    test_data,
                    all_train_data,
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
                    args.model,
                    scaler,
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
                    size=args.size,
                    train_labels=train_labels,
                    train_out_mean=train_out_mean,
                    train_out_std=train_out_std,
                    test_labels=test_labels,
                    test_out_mean=test_out_mean,
                    test_out_std=test_out_std,
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

                if args.model == "cnn" and args.representation == "ohe":
                    all_train_data_unscaled_targets = scaler.inverse_transform(
                        all_train_data.target.to_numpy().reshape(-1, 1)
                    )
                elif args.model == "cnn" and args.representation == "esm":
                    all_train_data_unscaled_targets = (
                        all_train_data.tensors[1].unsqueeze(1).numpy()
                        * scaler[1].numpy()
                        + scaler[0].numpy()
                    )
                else:
                    all_train_data_unscaled_targets = scaler.inverse_transform(
                        all_train_target.reshape(-1, 1)
                    )

                all_train_preds = al_full_train_out_mean
                mean_uncertainty = al_full_train_out_std.flatten()

                # Sample according to a strategy
                if (
                    "explorative" in strategy
                    or "score" in strategy
                    or "exploit" in strategy
                ):

                    # Find the lowest confidence (highest unc) samples, add
                    # them to the training inds consider average entropy across
                    # tasks

                    sq_error = np.square(
                        np.array(all_train_data_unscaled_targets) - all_train_preds
                    )
                    rmse = np.sqrt(np.mean(sq_error.astype(np.float32), axis=1))

                    if "explorative" in strategy:
                        per_sample_weight = mean_uncertainty
                    elif "score" in strategy:
                        per_sample_weight = rmse
                    elif "exploit" in strategy:
                        if args.model == "cnn" and args.representation == "esm":
                            scaled_preds = (
                                all_train_preds - scaler[0].numpy()
                            ) / scaler[1].numpy()
                            scaled_preds = scaled_preds.reshape(-1, 1)
                        else:
                            scaled_preds = scaler.transform(all_train_preds.reshape(-1, 1))
                        per_sample_weight = np.mean(scaled_preds, 1).astype(np.float32)

                        # Reverse and make sure weights (preds) are positive
                        if args.acquire_min:
                            per_sample_weight *= -1

                        std_mult = args.al_std_mult
                        if args.model == "gp":
                            mean_uncertainty = mean_uncertainty.numpy()
                        if "_lcb" in strategy:  # lower confidence bound
                            per_sample_weight += -std_mult * mean_uncertainty
                        elif "_ucb" in strategy:  # upper confidence bound
                            per_sample_weight += +std_mult * mean_uncertainty
                        elif "_ts" in strategy:  # thompson sampling
                            per_sample_weight = np.random.normal(
                                per_sample_weight, mean_uncertainty
                            )

                        per_sample_weight -= per_sample_weight.min()

                    # Save all the smiles along with their uncertainties/errors
                    train_subset_mask = np.zeros((n_total,))
                    train_subset_mask[train_subset_inds] = 1
                    df_scores = pd.DataFrame(
                        data={
                            # "Smiles": all_train_data.smiles(),
                            "Uncertainty": mean_uncertainty,
                            "Error": rmse,
                            "TrainInds": train_subset_mask,
                        }
                    )
                    Path(os.path.join(results_root, strategy, "tracks")).mkdir(
                        parents=True, exist_ok=True
                    )
                    df_scores.to_csv(
                        os.path.join(
                            results_root,
                            strategy,
                            "tracks",
                            f"{strategy}_step_{i}_{tic_time}.csv",
                        )
                    )
                elif strategy == "random":
                    per_sample_weight = np.ones((n_total,))  # uniform
                else:
                    raise ValueError(f"Unknown active learning strategy {strategy}")

                # Compute the top-k percent acquired
                # Grab the indicies that are in the top-k of only the training data
                if args.model == "cnn" and args.representation == "ohe":
                    top_k_scores_in_pool = np.sort(all_train_data.target.to_numpy())
                elif args.model == "cnn" and args.representation == "esm":
                    top_k_scores_in_pool = np.sort(all_train_data.tensors[1].numpy())
                else:
                    top_k_scores_in_pool = np.sort(
                        all_train_data_unscaled_targets.flatten()
                    )

                top_k_scores_in_pool = (
                    top_k_scores_in_pool[: args.al_topk]
                    if args.acquire_min
                    else top_k_scores_in_pool[-args.al_topk:]
                )

                if args.model == "cnn" and args.representation == "ohe":
                    top_k_scores_in_selection = np.sort(train_data.target.to_numpy())
                elif args.model == "cnn" and args.representation == "esm":
                    top_k_scores_in_selection = np.sort(train_data.tensors[1].numpy())
                else:
                    top_k_scores_in_selection = np.sort(
                        all_train_data_unscaled_targets[train_subset_inds].flatten()
                    )

                top_k_scores_in_selection = (
                    top_k_scores_in_selection[: args.al_topk]
                    if args.acquire_min
                    else top_k_scores_in_selection[-args.al_topk:]
                )

                # Find the overlap in indicies with our already acquired data points
                selection_overlap = np.in1d(
                    top_k_scores_in_selection, top_k_scores_in_pool
                )

                # Compute the percent overlap
                percent_top_k_overlap = np.mean(selection_overlap) * 100

                # find best sample in all training data
                if args.acquire_min:
                    best_sample = top_k_scores_in_pool.min()
                else:
                    best_sample = top_k_scores_in_pool.max()

                if args.acquire_min:
                    best_sample_aquired = top_k_scores_in_selection.min()
                else:
                    best_sample_aquired = top_k_scores_in_selection.max()

                if args.model in ["cnn", "gp"]:
                    mean_unc = test_out_std.mean().item()
                elif args.model == "ridge":
                    mean_unc = np.mean(test_out_std)

                print(
                    f"\nTest rho: {test_rho:.3f}, rmse: {test_rmse:.3f}, mae: {test_mae:.3f}, r2: {test_r2:.3f}, mean unc: {mean_unc:.3f}"
                )

                df = df.append(
                    {
                        "Strategy": strategy,
                        "Trial": i_trial,
                        "Train Data Ratio": round(
                            n_samples_per_run[i] / float(n_total), 3
                        ),
                        "TopKPercentOverlap": percent_top_k_overlap,
                        "TestRho": test_rho,
                        "TestRMSE": test_rmse,
                        "TestMAE": test_mae,
                        "TestR2": test_r2,
                        "MeanUncertainty": round(mean_unc, 3),
                        "train_rho_unc": round(train_rho_unc, 3),
                        "train_p_rho_unc": round(train_p_rho_unc, 3),
                        "train_percent_coverage": round(train_percent_coverage, 3),
                        "train_average_width_range": round(
                            train_average_width_range, 3
                        ),
                        "train_miscalibration_area": round(
                            train_miscalibration_area, 3
                        ),
                        "train_average_nll": round(train_average_nll, 3),
                        "train_average_optimal_nll": round(
                            train_average_optimal_nll, 3
                        ),
                        "train_average_nll_ratio": round(train_average_nll_ratio, 3),
                        "test_rho_unc": round(test_rho_unc, 3),
                        "test_p_rho_unc": round(test_p_rho_unc, 3),
                        "test_percent_coverage": round(test_percent_coverage, 3),
                        "test_average_width_range": round(test_average_width_range, 3),
                        "test_miscalibration_area": round(test_miscalibration_area, 3),
                        "test_average_nll": round(test_average_nll, 3),
                        "test_average_optimal_nll": round(test_average_optimal_nll, 3),
                        "test_average_nll_ratio": round(test_average_nll_ratio, 3),
                        "best_sample_in_train": round(best_sample, 3),
                        "best_sample_acquired": round(best_sample_aquired, 3),
                    },
                    ignore_index=True,
                )

                print(f"Percent top-k = {round(percent_top_k_overlap, 2)}")

                # Add new samples to training set
                n_add = (
                    n_samples_per_run[min(i + 1, n_loops - 1)] - n_samples_per_run[i]
                )
                if n_add > 0:  # n_add = 0 on the last iteration, when we are done

                    # Probability of sampling a new point, depends on the weight
                    per_sample_prob = deepcopy(per_sample_weight)

                    # Exclude data we've already trained with, and normalize to probability
                    per_sample_prob[train_subset_inds] = 0.0
                    per_sample_prob = per_sample_prob / per_sample_prob.sum()

                    # Sample accordingly and add to our training inds
                    if "sample" in strategy or strategy == "random":
                        train_inds_to_add = np.random.choice(
                            n_total, size=n_add, p=per_sample_prob, replace=False
                        )
                    else:
                        # greedy, just pick the highest probability indicies
                        inds_sorted = np.argsort(per_sample_prob)  # smallest to largest
                        train_inds_to_add = inds_sorted[-n_add:]  # grab the last k inds

                    # Add the indices to the training set
                    train_subset_inds = np.append(train_subset_inds, train_inds_to_add)

                torch.cuda.empty_cache()

                # End AL loop if we've acquired the best sample or if we've reached 30% of the training data
                if args.al_step_scale == "single" and (best_sample_aquired == best_sample):
                    print("Acquired best sample, ending fold early")
                    break
                elif args.al_step_scale == "single" and (len(train_subset_inds) >= thirty_percent_of_train):
                    print("Acquired 30% of training data, ending fold early")
                    break

        # END SINGLE FOLD
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        df.to_csv(os.path.join(results_root, f"{timestamp}.csv"))

    print(f"\nDone with all folds and saved into {results_root}")
    os._exit(1)
