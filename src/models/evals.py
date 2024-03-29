import pickle

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import activate_dropout


def concat_tensor(tensor_list, keep_tensor=False):
    """converts a list of tensors to a numpy array for stats analysis"""
    for i, item in enumerate(tensor_list):
        item.to("cpu")
        if i == 0:
            output_tensor = item
        if i > 0:
            output_tensor = torch.cat((output_tensor, item), 0)

    if keep_tensor:
        return output_tensor
    else:
        return np.array(output_tensor)


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


def evaluate_log_likelihood(error, uncertainty, min_val=1e-5):
    log_likelihood = 0
    optimal_log_likelihood = 0

    for err, unc in zip(error, uncertainty):
        # Encourage small standard deviations.
        log_likelihood -= np.log(2 * np.pi * max(min_val, unc**2)) / 2
        optimal_log_likelihood -= np.log(2 * np.pi * err**2) / 2  # optimal means (error = std dev)

        # Penalize for large error.
        log_likelihood -= err**2 / (2 * max(min_val, unc**2))
        optimal_log_likelihood -= 1 / 2  # optimal means (error = std dev)

    return {
        "log_likelihood": log_likelihood,
        "optimal_log_likelihood": optimal_log_likelihood,
        "average_log_likelihood": log_likelihood / len(error),
        "average_optimal_log_likelihood": optimal_log_likelihood / len(error),
    }


def regression_eval(predicted, labels, SAVE_PATH=None):
    """
    input: 1D tensor or array of predicted values and labels
    output: saves spearman, MSE, and graph of predicted vs actual
    """
    predicted = np.array(predicted)
    labels = np.array(labels)

    rho = stats.spearmanr(labels, predicted).correlation
    rmse = mean_squared_error(labels, predicted, squared=False)
    mae = mean_absolute_error(labels, predicted)
    r2 = r2_score(labels, predicted)

    # remove graphing - causes segmentation fault  # TODO: reenable regression plotting (include parities for train, validation, test + learning curves)
    # plt.figure()
    # plt.title('predicted (y) vs. labels (x)')
    # sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
    # plt.savefig(SAVE_PATH / 'preds_vs_labels.png', dpi = 300)

    return rho, rmse, mae, r2


def uncertainty_eval(
    preds_mean, labels, preds_std, SAVE_PATH, train_label_range
):  # TODO: add uncertainty plots (parity with error bars, miscalbration area, etc.)
    """evaluate uncertainty predictions"""

    residual = np.abs(labels - preds_mean)

    df = pd.DataFrame()
    df["labels"] = labels
    df["preds_mean"] = preds_mean
    df["residual"] = residual

    coverage = residual < 2 * preds_std
    width_range = 4 * preds_std / train_label_range
    df["preds_std"] = preds_std
    df["coverage"] = coverage
    df["width/range"] = width_range
    rho_unc, p_rho_unc = stats.spearmanr(df["residual"], df["preds_std"])
    percent_coverage = sum(df["coverage"]) / len(df)
    average_width_range = df["width/range"].mean() / train_label_range
    miscalibration_area_results = evaluate_miscalibration_area(
        df["residual"], df["preds_std"]
    )
    miscalibration_area = miscalibration_area_results["miscalibration_area"]
    ll_results = evaluate_log_likelihood(df["residual"], df["preds_std"])
    average_log_likelihood = ll_results["average_log_likelihood"]
    average_optimal_log_likelihood = ll_results["average_optimal_log_likelihood"]
    print("RHO UNCERTAINTY: ", rho_unc)
    print("RHO UNCERTAINTY P-VALUE: ", p_rho_unc)
    print("PERCENT COVERAGE: ", percent_coverage)
    print("AVERAGE WIDTH / TRAINING SET RANGE: ", average_width_range)
    print("MISCALIBRATION AREA: ", miscalibration_area)
    print("AVERAGE NLL: ", -1 * average_log_likelihood)
    print("AVERAGE OPTIMAL NLL: ", -1 * average_optimal_log_likelihood)
    print("NLL / NLL_OPT:", average_log_likelihood / average_optimal_log_likelihood)
    metrics = [
        rho_unc,
        p_rho_unc,
        percent_coverage,
        average_width_range,
        miscalibration_area,
        -1 * average_log_likelihood,
        -1 * average_optimal_log_likelihood,
        average_log_likelihood / average_optimal_log_likelihood,
    ]

    df.to_csv(
        SAVE_PATH / "preds.csv",
        index=False,
    )

    return metrics


def pred_gp(X, model, likelihood, device, size, y_scaler=None, y=None):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(
        size
    ):
        observed_pred = likelihood(model(X.to(device)))
        preds_mean = observed_pred.mean.cpu()
        (
            lower,
            upper,
        ) = observed_pred.confidence_region()  # 2 st dev above and below mean

    lower = lower.cpu()
    upper = upper.cpu()
    preds_std = (upper - preds_mean) / 2

    if y_scaler:
        if y is not None:
            y = y_scaler.inverse_transform(y.reshape(-1, 1))
        preds_mean = y_scaler.inverse_transform(preds_mean.reshape(-1, 1))
        preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    return preds_mean, preds_std, y


def pred_cnn(
    data_iterator,
    model,
    device,
    MODEL_PATH,
    y_scaler=None,
    dropout=0.0,
    mve=False,
    evidential=False,
    svi=False,
):
    """run data through model"""

    calculate_std = False

    model = model.to(device)
    bestmodel_save = MODEL_PATH / "bestmodel.tar"
    sd = torch.load(bestmodel_save)
    model.load_state_dict(sd["model_state_dict"])
    print("loaded the saved model")

    def test_step(model, batch):
        try:
            src, tgt, mask = batch
        except ValueError:  # No masks for ESM mean embeddings
            src, tgt = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        try:
            mask = mask.to(device).float()
        except UnboundLocalError:  # No masks for ESM mean embeddings
            mask = None
        output = model(src, mask, evidential)
        return output.detach().cpu(), tgt.detach().cpu()

    model = model.eval()

    if dropout > 0 or svi or mve or evidential:
        calculate_std = True

    # Turn on dropout for inference to estimate uncertainty
    if dropout > 0:

        def activate_dropout_(model):
            return activate_dropout(model, dropout)

        model.apply(activate_dropout_)
    if dropout > 0 or svi:
        num_evals = 10
        out_list = []
    else:
        num_evals = 1

    for i in range(num_evals):
        if isinstance(data_iterator.batch_sampler.sampler, torch.utils.data.sampler.RandomSampler):
            np.random.seed(0)
            torch.manual_seed(0)  # set same random seeds to get same order of samples in training set
        outputs = []
        tgts = []
        n_seen = 0
        for _, batch in enumerate(data_iterator):
            np.random.seed(i)
            torch.manual_seed(i)  # set different random seeds to get different predictions from dropout/SVI
            output, tgt = test_step(model, batch)
            outputs.append(output)
            tgts.append(tgt)
            n_seen += len(batch[0])

        out = torch.cat(outputs).numpy()
        labels = torch.cat(tgts).cpu().numpy()

        if dropout > 0 or svi:
            out_list.append(out)

    if dropout > 0 or svi:
        out = np.mean(out_list, axis=0)
        preds_std = np.std(out_list, axis=0)
    elif mve:
        preds_std = out[:, 1]
        out = out[:, 0]
    elif evidential:
        lambdas = out[:, 1]  # also called nu or v
        alphas = out[:, 2]
        betas = out[:, 3]
        out = out[:, 0]

        aleatoric_unc_var = betas / (alphas - 1)
        epistemic_unc_var = aleatoric_unc_var / lambdas

        preds_std = np.sqrt(epistemic_unc_var + aleatoric_unc_var)

    if y_scaler:
        if isinstance(y_scaler, tuple):
            labels = labels * y_scaler[1].numpy() + y_scaler[0].numpy()
            out = out * y_scaler[1].numpy() + y_scaler[0].numpy()
            if calculate_std:
                preds_std = preds_std.reshape(-1, 1) * y_scaler[1].numpy()
        else:
            labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
            out = y_scaler.inverse_transform(out.reshape(-1, 1))
            if calculate_std:
                preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    if not calculate_std:
        preds_std = None

    return labels, out, preds_std


def evaluate_cnn(labels, out, preds_std, SAVE_PATH, train_label_range, return_preds=False):
    """print eval stats"""

    SAVE_PATH.mkdir(
        parents=True, exist_ok=True
    )  # make directory if it doesn't exist already
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((labels, out, preds_std), f)
    rho, rmse, mae, r2 = regression_eval(
        predicted=out, labels=labels, SAVE_PATH=SAVE_PATH
    )
    metrics = uncertainty_eval(
        out.squeeze(),
        labels.squeeze(),
        preds_std.squeeze(),
        SAVE_PATH,
        train_label_range,
    )
    if return_preds:
        return rho, rmse, mae, r2, metrics, out, preds_std
    else:
        return rho, rmse, mae, r2, metrics


def evaluate_ridge(X, y, model, SAVE_PATH, train_label_range, y_scaler=None, return_preds=False):
    preds_mean, preds_std = model.predict(X, return_std=True)

    if y_scaler:
        y = y_scaler.inverse_transform(y.reshape(-1, 1))
        preds_mean = y_scaler.inverse_transform(preds_mean.reshape(-1, 1))
        preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    SAVE_PATH.mkdir(
        parents=True, exist_ok=True
    )  # make directory if it doesn't exist already
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((y, preds_mean, preds_std), f)
    rho, rmse, mae, r2 = regression_eval(
        predicted=preds_mean, labels=y, SAVE_PATH=SAVE_PATH
    )
    metrics = uncertainty_eval(
        preds_mean.squeeze(),
        y.squeeze(),
        preds_std.squeeze(),
        SAVE_PATH,
        train_label_range,
    )

    if return_preds:
        return rho, rmse, mae, r2, metrics, preds_mean, preds_std
    else:
        return rho, rmse, mae, r2, metrics


def evaluate_gp(
    X, y, model, likelihood, device, size, SAVE_PATH, train_label_range, y_scaler=None, return_preds=False,
):
    preds_mean, preds_std, y = pred_gp(X, model, likelihood, device, size, y_scaler, y)

    SAVE_PATH.mkdir(
        parents=True, exist_ok=True
    )  # make directory if it doesn't exist already
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((y, preds_mean, preds_std), f)
    rho, rmse, mae, r2 = regression_eval(
        predicted=preds_mean, labels=y, SAVE_PATH=SAVE_PATH
    )
    metrics = uncertainty_eval(
        preds_mean.squeeze(),
        y.squeeze(),
        preds_std.squeeze().numpy(),
        SAVE_PATH,
        train_label_range,
    )
    if return_preds:
        return rho, rmse, mae, r2, metrics, preds_mean, preds_std
    else:
        return rho, rmse, mae, r2, metrics


def eval_model(
    model,
    y_scaler,
    train_seq=None,
    train_target=None,
    test_seq=None,
    test_target=None,
    lr_trained=None,
    gp_trained=None,
    EVAL_PATH=None,
    EVAL_PATH_BASE=None,
    likelihood=None,
    device=None,
    size=None,
    train_labels=None,
    train_out_mean=None,
    train_out_std=None,
    test_labels=None,
    test_out_mean=None,
    test_out_std=None,
):
    if model == "ridge":
        train_label_range = np.max(train_target) - np.min(train_target)
        print("\nEvaluating model on train set...")
        train_rho, train_rmse, train_mae, train_r2, train_unc_metrics = evaluate_ridge(
            train_seq,
            train_target,
            lr_trained,
            EVAL_PATH / "train",
            train_label_range,
            y_scaler,
        )
        print("\nEvaluating model on test set...")
        test_rho, test_rmse, test_mae, test_r2, test_unc_metrics = evaluate_ridge(
            test_seq,
            test_target,
            lr_trained,
            EVAL_PATH / "test",
            train_label_range,
            y_scaler,
        )

    if model == "gp":
        train_label_range = np.max(train_target.numpy()) - np.min(train_target.numpy())
        print("\nEvaluating model on train set...")
        train_rho, train_rmse, train_mae, train_r2, train_unc_metrics = evaluate_gp(
            train_seq,
            train_target,
            gp_trained,
            likelihood,
            device,
            size,
            EVAL_PATH / "train",
            train_label_range,
            y_scaler,
        )
        print("\nEvaluating model on test set...")
        test_rho, test_rmse, test_mae, test_r2, test_unc_metrics = evaluate_gp(
            test_seq,
            test_target,
            gp_trained,
            likelihood,
            device,
            size,
            EVAL_PATH / "test",
            train_label_range,
            y_scaler,
        )

    if model == "cnn":

        train_label_range = np.max(train_labels) - np.min(train_labels)
        print("\nEvaluating model on train set...")
        train_rho, train_rmse, train_mae, train_r2, train_unc_metrics = evaluate_cnn(
            train_labels,
            train_out_mean,
            train_out_std,
            EVAL_PATH_BASE / "train",
            train_label_range,
        )
        print("\nEvaluating model on test set...")
        test_rho, test_rmse, test_mae, test_r2, test_unc_metrics = evaluate_cnn(
            test_labels,
            test_out_mean,
            test_out_std,
            EVAL_PATH_BASE / "test",
            train_label_range,
        )

    return (
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
    )
