import pickle

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from scipy.stats import spearmanr
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
    probabilities = [2 * (stats.norm.cdf(standard_dev) - 0.5) for standard_dev in standard_devs]
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
    miscalibration = [np.abs(fraction_under_thresholds[i] - thresholds[i]) for i in range(len(thresholds))]
    miscalibration_area = 0
    for i in range(1, 1001):
        miscalibration_area += np.average([miscalibration[i - 1], miscalibration[i]]) * 0.001

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
        log_likelihood -= np.log(2 * np.pi * max(0.00001, unc**2)) / 2
        optimal_log_likelihood -= np.log(2 * np.pi * err**2) / 2

        # Penalize for large error.
        log_likelihood -= err**2 / (2 * max(0.00001, unc**2))
        optimal_log_likelihood -= 1 / 2

    return {
        "log_likelihood": log_likelihood,
        "optimal_log_likelihood": optimal_log_likelihood,
        "average_log_likelihood": log_likelihood / len(error),
        "average_optimal_log_likelihood": optimal_log_likelihood / len(error),
    }


def regression_eval(predicted, labels, SAVE_PATH):  # TODO: add uncertainty_eval fxn with stuff from main branch calculate_metrics fxn, add uncertainty plots, call uncetainty_eval below (3x)
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

    # remove graphing - causes segmentation fault  # TODO: try to reenable plotting (include parities for train, validation, test + learning curves)
    # plt.figure()
    # plt.title('predicted (y) vs. labels (x)')
    # sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
    # plt.savefig(SAVE_PATH / 'preds_vs_labels.png', dpi = 300)

    return round(rho, 2), round(rmse, 2), round(mae, 2), round(r2, 2)


def evaluate_cnn(data_iterator, model, device, MODEL_PATH, SAVE_PATH, y_scaler=None, dropout=0.0):  # TOOD: write separate function to evaluate cnn ensemble (based on prediction files that were written out)
    """run data through model and print eval stats"""

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
        output = model(src, mask)
        return output.detach().cpu(), tgt.detach().cpu()

    model = model.eval()

    # Turn on dropout for inference to estimate uncertainty
    if dropout > 0:
        def activate_dropout_(model):
            return activate_dropout(model, dropout)
        model.apply(activate_dropout_)
        num_evals = 5
        out_list = []
        calculate_std = True
    else:
        num_evals = 1

    for i in range(num_evals):
        np.random.seed(i)
        torch.manual_seed(i)
        outputs = []
        tgts = []
        n_seen = 0
        for i, batch in enumerate(data_iterator):
            output, tgt = test_step(model, batch)
            outputs.append(output)
            tgts.append(tgt)
            n_seen += len(batch[0])

        out = torch.cat(outputs).numpy()
        labels = torch.cat(tgts).cpu().numpy()

        if dropout > 0:
            out_list.append(out)

    if dropout > 0:
        out = np.mean(out_list, axis=0)
        preds_std = np.std(out_list, axis=0)

    if y_scaler:
        if isinstance(y_scaler, tuple):
            labels = labels * y_scaler[1].numpy() + y_scaler[0].numpy()
            out = out * y_scaler[1].numpy() + y_scaler[0].numpy()
            if calculate_std:
                preds_std = preds_std.reshape(-1, 1) * y_scaler[1]
        else:
            labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
            out = y_scaler.inverse_transform(out.reshape(-1, 1))
            if calculate_std:
                preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    SAVE_PATH.mkdir(parents=True, exist_ok=True)  # make directory if it doesn't exist already  # TODO: make sure preds_std and out are correct shape with dropout
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((out, labels), f)  # TODO: write std to file if applicable
    rho, rmse, mae, r2 = regression_eval(predicted=out, labels=labels, SAVE_PATH=SAVE_PATH)

    return rho, rmse, mae, r2


def evaluate_ridge(X, y, model, SAVE_PATH, y_scaler=None):
    preds_mean, preds_std = model.predict(X, return_std=True)

    if y_scaler:
        y = y_scaler.inverse_transform(y.reshape(-1, 1))
        preds_mean = y_scaler.inverse_transform(preds_mean.reshape(-1, 1))
        preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    SAVE_PATH.mkdir(parents=True, exist_ok=True)  # make directory if it doesn't exist already
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((preds_mean, y), f)  # TODO: write std to file if applicable
    rho, rmse, mae, r2 = regression_eval(predicted=preds_mean, labels=y, SAVE_PATH=SAVE_PATH)

    return rho, rmse, mae, r2


def evaluate_gp(X, y, model, likelihood, device, size, SAVE_PATH, y_scaler=None):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(size):
        observed_pred = likelihood(model(X.to(device)))
        preds_mean = observed_pred.mean.cpu()
        lower, upper = observed_pred.confidence_region()  # 2 st dev above and below mean

    lower = lower.cpu()
    upper = upper.cpu()
    preds_std = (upper - preds_mean) / 2

    if y_scaler:
        y = y_scaler.inverse_transform(y.reshape(-1, 1))
        preds_mean = y_scaler.inverse_transform(preds_mean.reshape(-1, 1))
        preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    SAVE_PATH.mkdir(parents=True, exist_ok=True)  # make directory if it doesn't exist already
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((preds_mean, y), f)  # TODO: write std to file if applicable
    rho, rmse, mae, r2 = regression_eval(predicted=preds_mean, labels=y, SAVE_PATH=SAVE_PATH)

    return rho, rmse, mae, r2
