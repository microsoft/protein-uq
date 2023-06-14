from dimred_train_test_model_unc import dimred_train_test_model_unc

for method in ['pca', 'umap']:
    for representation in ['ohe', 'esm']:
        for split in ['gb1_2', 'gb1_3', 'gb1_4', 'gb1_1']:
            for model in ['ridge', 'gp', 'cnn']:
                if model == "ridge":
                    allowed_uncertainties = ["ridge"]
                elif model == "gp":
                    allowed_uncertainties = ["gp"]
                elif model == "cnn":
                    allowed_uncertainties = [
                        # "dropout",
                        "ensemble",
                        "mve",
                        "evidential",
                        "svi",
                    ]
                for uncertainty in allowed_uncertainties:
                    dimred_train_test_model_unc(split, representation, method, model, uncertainty)
                    print(f"Finished {split} {representation} {method} {model} {method}")

for method in ['pca', 'umap']:
    for representation in ['ohe', 'esm']:
        for split in ['aav_2', 'aav_5', 'aav_7', 'meltome_1']:
            for model in ['ridge', 'gp', 'cnn']:
                if model == "ridge":
                    allowed_uncertainties = ["ridge"]
                elif model == "gp":
                    allowed_uncertainties = ["gp"]
                elif model == "cnn":
                    allowed_uncertainties = [
                        # "dropout",
                        "ensemble",
                        "mve",
                        "evidential",
                        "svi",
                    ]
                for uncertainty in allowed_uncertainties:
                    dimred_train_test_model_unc(split, representation, method, model, uncertainty)
                    print(f"Finished {split} {representation} {method} {model} {method}")
