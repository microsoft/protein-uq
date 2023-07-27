from plot_num_muts import plot_num_muts

for method in ['pca', 'pca_tsne']:
    for representation in ['ohe', 'esm']:
        for split in ['gb1_2']:
            for model in ['gp', 'cnn']:  # ['ridge', 'gp', 'cnn']:
                if model == "ridge":
                    allowed_uncertainties = ["ridge"]
                elif model == "gp":
                    allowed_uncertainties = ["gp"]
                elif model == "cnn":
                    allowed_uncertainties = [
                        # "dropout",
                        "ensemble",
                        # "mve",
                        "evidential",
                        # "svi",
                    ]
                for uncertainty in allowed_uncertainties:
                    # for perplexity in [5, 10, 20, 50, 100]:
                    #plot_num_muts(split, representation, method, model, uncertainty, perplexity=30)
                    plot_num_muts(split, representation, method, model, uncertainty, perplexity=5)
                    print(f"Finished {split} {representation} {method} {model} {method}")
