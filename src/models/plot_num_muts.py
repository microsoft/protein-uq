import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from train_all import split_dict
from utils import load_and_scale_data


def create_parser():
    parser = argparse.ArgumentParser(description='Train and test dimensionality reduction models')
    parser.add_argument("--split", type=str)
    parser.add_argument('--method', type=str, choices=['pca', 'umap', 'tsne', 'pca_tsne', 'umap_tsne'], default='pca', help='Dimensionality reduction model')
    parser.add_argument("--model", choices=["ridge", "gp", "cnn"], type=str)
    parser.add_argument("--representation", choices=["ohe", "esm"], type=str)
    parser.add_argument(
        "--uncertainty",
        choices=["ridge", "gp", "dropout", "ensemble", "mve", "evidential", "svi"],
        type=str,
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors')
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity')
    parser.add_argument('--min_dist', type=float, default=0.1, help='Minimum distance')
    parser.add_argument('--metric', type=str, choices=['euclidean', 'manhattan', 'cosine', 'correlation', 'jaccard'], default='euclidean', help='Distance metric')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    return parser


def count_mutations(wild_type, sequences):
    mutation_counts = []

    # Assuming the sequences are of the same length as the wild type
    if len(wild_type) != len(sequences[0]):
        raise ValueError("The length of the wild type sequence should be the same as the input sequences.")

    for seq in sequences:
        if len(seq) != len(wild_type):
            raise ValueError("All sequences should have the same length as the wild type sequence.")

        # Calculate the number of mutations by comparing each amino acid
        mutations = sum(a != b for a, b in zip(wild_type, seq))
        mutation_counts.append(mutations)

    return mutation_counts


def plot_num_muts(split, representation, method, model, uncertainty, dropout=0.0, n_neighbors=15, perplexity=30.0, min_dist=0.1, metric='euclidean', n_jobs=1, random_state=42):
    split_orig = split
    split = split_dict[split]
    dataset = re.findall(r"(\w*)\_", split_orig)[0]

    # Load data
    (
        _,
        _,
        _,
        _,
        x_scaler,
        y_scaler,
        train_seq,
        train_target,
        test_seq,
        test_target,
    ) = load_and_scale_data(
        representation, "gp", dataset, split, True, False, False, False, False
    )

    # load original amino acid sequences from file
    orig_df = pd.read_csv(f"../../data/{dataset}/splits/{split}.csv")

    # count number of mutations in each sequence
    # wild type is sequence in train with target 1.0
    wild_type_sequence = orig_df[orig_df["target"] == 1.0]["sequence"].values[0]
    mut_sequences = orig_df[orig_df["target"] != 1.0]["sequence"].values

    try:
        mutation_counts = count_mutations(wild_type_sequence, mut_sequences)
        orig_df["num_muts"] = [0] + mutation_counts
        # print(orig_df.value_counts("num_muts"))
    except ValueError as e:
        print(f"Error: {e}")

    # exclude validation data (43 points for GB1)
    orig_df = orig_df[orig_df["validation"].isna()]

    # orig_train_df = orig_df[orig_df["set"] == "train"]
    orig_test_df = orig_df[orig_df["set"] == "test"]

    # Load results
    results_dir = f"results/{dataset}/{split}/{model}/{representation}/{uncertainty}/cv_fold_0"
    test_results_df = pd.read_csv(f"{results_dir}/test/preds.csv")

    # combine train and test
    all_seq = np.concatenate((train_seq, test_seq))
    all_target = np.concatenate((train_target, test_target))
    print(split, all_seq.shape, all_target.shape)

    test_uncertainty = test_results_df["preds_std"].values
    test_residual = test_results_df["residual"].values
    test_num_muts = orig_test_df["num_muts"].values

    # violin plot of uncertainty vs number of mutations
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.title(f"{dataset} {split} {representation} {model} {uncertainty}")
    plt.xlabel("Number of Mutations")
    plt.ylabel("Uncertainty")
    plt.grid(False)
    sns.violinplot(x=test_num_muts, y=test_uncertainty)
    plt.savefig(f"violin/{dataset}_{split}_{representation}_{model}_{uncertainty}.png")

    # violin plot of uncertainty vs number of mutations
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.title(f"{dataset} {split} {representation} {model} {uncertainty}")
    plt.xlabel("Number of Mutations")
    plt.ylabel("Uncertainty")
    plt.grid(False)
    sns.violinplot(x=test_num_muts, y=test_residual)
    plt.savefig(f"violin_residual/{dataset}_{split}_{representation}_{model}_{uncertainty}.png")

    # load dimensionality reduction embeddings from file
    X = np.load(f"dimred_all/{method}_{dataset}_{split}_{representation}_{perplexity}.npy")
    X_test = X[len(train_target):, :]

    # plot reduced dimensions colored by number of mutations
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.title(f"{method} {dataset} {split} {representation}")
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    plt.grid(False)
    for num_muts in [4, 3, 2, 1, 0]:
        # plt.scatter(X_test[test_num_muts == num_muts, 0], X_test[test_num_muts == num_muts, 1], s=10, label=num_muts, alpha=0.1)
        plt.scatter(X[orig_df["num_muts"] == num_muts, 0], X[orig_df["num_muts"] == num_muts, 1], s=10, label=num_muts, alpha=0.5)
    plt.legend()
    # remove plot axes and labels
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    if 'tsne' in method:
        plt.savefig(f"dimred_num_muts/{method}_{dataset}_{split}_{representation}_{perplexity}.png")
    else:
        plt.savefig(f"dimred_num_muts/{method}_{dataset}_{split}_{representation}.png")

    # plot reduced dimensions colored by uncertainty
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.title(f"{method} {dataset} {split} {representation} {model} {uncertainty} test")
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    plt.grid(False)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=test_uncertainty, s=10, cmap="Blues")
    plt.colorbar(label="Uncertainty")
    # remove plot axes and labels
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    if 'tsne' in method:
        plt.savefig(f"dimred_num_muts_unc/{method}_{dataset}_{split}_{representation}_{model}_{uncertainty}_{perplexity}.png")
    else:
        plt.savefig(f"dimred_num_muts_unc/{method}_{dataset}_{split}_{representation}_{model}_{uncertainty}.png")

    return


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    plot_num_muts(args.split, args.representation, args.method, args.model, args.uncertainty, args.dropout, args.n_neighbors, args.perplexity, args.min_dist, args.metric, args.n_jobs, args.random_state)
