import pickle
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.metrics import mean_squared_error


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


def regression_eval(predicted, labels, SAVE_PATH):  # TODO: add uncertainty_eval fxn with stuff from main branch calculate_metrics fxn, add uncertainty plots, call uncetainty_eval below (3x)
    """
    input: 1D tensor or array of predicted values and labels
    output: saves spearman, MSE, and graph of predicted vs actual
    """
    predicted = np.array(predicted)
    labels = np.array(labels)

    rho, _ = stats.spearmanr(predicted, labels)  # spearman
    mse = mean_squared_error(predicted, labels)  # MSE

    # remove graphing - causes segmentation fault  # TODO: try to reenable plotting (include parities for train, validation, test + learning curves)
    # plt.figure()
    # plt.title('predicted (y) vs. labels (x)')
    # sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
    # plt.savefig(SAVE_PATH / 'preds_vs_labels.png', dpi = 300)

    return round(rho, 2), round(mse, 2)  # TODO: add more regression metrics to this


def evaluate_cnn(data_iterator, model, device, MODEL_PATH, SAVE_PATH, y_scaler=None):
    """run data through model and print eval stats"""

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

    if y_scaler:
        labels = y_scaler.inverse_transform(labels)
        out = y_scaler.inverse_transform(out)
        # preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_ # TODO: unscale st dev (make sure st dev and not var)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)  # make directory if it doesn't exist already # TODO: add saving to ridge and GP
    with open(SAVE_PATH / "preds_labels_raw.pickle", "wb") as f:
        pickle.dump((out, labels), f)
    rho, mse = regression_eval(predicted=out, labels=labels, SAVE_PATH=SAVE_PATH)

    return rho, mse


def evaluate_ridge(X, y, model, SAVE_PATH, y_scaler=None):
    preds_mean, preds_std = model.predict(X, return_std=True)

    if y_scaler:
        y = y_scaler.inverse_transform(y.reshape(-1, 1))
        preds_mean = y_scaler.inverse_transform(preds_mean.reshape(-1, 1))
        preds_std = preds_std.reshape(-1, 1) * y_scaler.scale_

    rho, mse = regression_eval(predicted=preds_mean, labels=y, SAVE_PATH=SAVE_PATH)
    return rho, mse


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

    rho, mse = regression_eval(predicted=preds_mean, labels=y, SAVE_PATH=SAVE_PATH)
    return rho, mse
