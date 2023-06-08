import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd

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
    parser.add_argument('--metric', type=str, choices=['euclidean', 'manhattan', 'cosine', 'correlation'], default='euclidean', help='Distance metric')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    split = split_dict[args.split]
    dataset = re.findall(r"(\w*)\_", args.split)[0]

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
        args.representation, "gp", dataset, split, True, False, False, False, True
    )

    # Load results
    results_dir = f"results/{dataset}/{split}/{args.model}/{args.representation}/{args.uncertainty}/cv_fold_0"
    test_results_df = pd.read_csv(f"{results_dir}/test/preds.csv")

    test_uncertainty = test_results_df["preds_std"].values

    # Do dimensionality reduction
    if args.method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=args.random_state)
        X_test = pca.fit_transform(test_seq)

    elif args.method == "umap":
        import umap

        umap_ = umap.UMAP(
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )
        X_test = umap_.fit_transform(test_seq)

    elif args.method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            metric=args.metric,
            random_state=args.random_state,
        )
        X_test = tsne.fit_transform(test_seq)

    elif args.method == "pca_tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        pca = PCA(n_components=50, random_state=args.random_state)
        X_test = pca.fit_transform(test_seq)

        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            metric=args.metric,
            random_state=args.random_state,
        )
        X_test = tsne.fit_transform(test_seq)

    elif args.method == "umap_tsne":
        import umap
        from sklearn.manifold import TSNE

        umap_ = umap.UMAP(
            n_components=50,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )
        X_test = umap_.fit_transform(test_seq)

        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            metric=args.metric,
            random_state=args.random_state,
        )
        X_test = tsne.fit_transform(test_seq)

    # Make plot
    plt.scatter(X_test[:, 0], X_test[:, 1], c=test_uncertainty, cmap="viridis")
    plt.xlabel("Reduced Dimension 1")
    plt.ylabel("Reduced Dimension 2")
    plt.colorbar(label="Uncertainty")
    plt.title(f"{args.method} {dataset} {split} {args.representation} {args.model} {args.uncertainty}")
    plt.savefig(f"dimred_unc/{args.method}_train_and_test_color_by_unc_{dataset}_{split}_{args.representation}_{args.model}_{args.uncertainty}.png")
    plt.show()
