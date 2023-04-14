import functools
import gpytorch
import numpy as np
import random
import torch
from scipy.stats import spearmanr

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from evals import pred_cnn, pred_gp
from models import BayesianRidgeRegression, ExactGPModel, FluorescenceModel
from utils import (
    ASCollater,
    ESMSequenceMeanDataset,
    SequenceDataset,
    Tokenizer,
    det_loss,
    evidential_loss,
    negative_log_likelihood,
    # save_parity_plot,
    vocab,
)


def train_cnn(
    train_iterator,
    val_iterator,
    model,
    device,
    criterion,
    optimizer,
    epoch_num,
    MODEL_PATH,
    mve=False,
    evidential=False,
    svi=False,
):

    patience = 20
    p = 0
    best_rho = -1.1
    model = model.to(device)

    def step(model, batch, train=True):
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
        if tgt.ndim == 1 and output.ndim == 2:
            tgt = tgt.unsqueeze(
                -1
            )  # unsqueeze targets for ESM mean ([batch_size] -> [batch_size, 1])
        if mve:
            loss = criterion(output[:, 0], output[:, 1], np.squeeze(tgt))
        elif evidential:
            loss = criterion(
                output[:, 0], output[:, 1], output[:, 2], output[:, 3], np.squeeze(tgt)
            )
        elif svi:
            loss = criterion(output.squeeze(), tgt.squeeze(), model)
        else:
            loss = criterion(output, tgt)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), output.detach().cpu(), tgt.detach().cpu()

    def epoch(model, train, current_step=0):
        if train:
            model = model.train()
            loader = train_iterator
            # t = "Training"
            # n_total = len(train_iterator)
        else:
            model = model.eval()
            loader = val_iterator
            # t = "Validating"
            # n_total = len(val_iterator)

        losses = []
        outputs = []
        tgts = []
        n_seen = 0
        for i, batch in enumerate(loader):
            loss, output, tgt = step(model, batch, train)
            losses.append(loss)
            outputs.append(output)
            tgts.append(tgt)

            n_seen += len(batch[0])
            if train:
                nsteps = current_step + i + 1
            else:
                nsteps = i

        outputs = torch.cat(outputs).numpy()
        tgts = torch.cat(tgts).cpu().numpy()

        if train:
            with torch.no_grad():
                _, val_rho = epoch(model, False, current_step=nsteps)
            print("epoch: %d loss: %.3f val rho: %.3f" % (e + 1, loss, val_rho))

            # if mve or evidential:
            #     save_parity_plot(tgts, outputs[:, 0], epoch_num=current_step, set_name="train", axlim=(-2, 5))
            # else:
            #     save_parity_plot(tgts, outputs, epoch_num=current_step, set_name="train", axlim=(-2, 5))

        if not train:
            if mve or evidential:
                val_rho = spearmanr(tgts, outputs[:, 0]).correlation
                # mse = mean_squared_error(tgts, outputs[:, 0])
            else:
                val_rho = spearmanr(tgts, outputs).correlation
                # mse = mean_squared_error(tgts, outputs)

            # if mve or evidential:
            #     save_parity_plot(tgts, outputs[:, 0], epoch_num=current_step, set_name="val", axlim=(-2, 5))
            # else:
            #     save_parity_plot(tgts, outputs, epoch_num=current_step, set_name="val", axlim=(-2, 5))

        return i, val_rho

    nsteps = 0
    e = 0
    bestmodel_save = MODEL_PATH / "bestmodel.tar"  # path to save best model
    for e in range(epoch_num):
        s, val_rho = epoch(model, train=True, current_step=nsteps)
        # print(val_rho)
        nsteps += s

        if val_rho > best_rho:
            p = 0
            best_rho = val_rho
            torch.save(
                {
                    "step": nsteps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                bestmodel_save,
            )

        else:
            p += 1
        if p == patience:
            print("MET PATIENCE")
            print("Finished training at epoch {0}".format(e))
            return

    print("Finished training CNN at epoch {0}".format(epoch_num))
    return


def train_ridge(X_train, y_train, model):
    model.fit(X_train, y_train)
    iterations = model.n_iter_
    print("Finished training BayesianRidge at iteration {0}".format(iterations))
    # save_parity_plot(y_train, model.predict(X_train), set_name="train", axlim=(-2, 5))
    return model, iterations


def train_gp(X_train, y_train, model, likelihood, device, length, size):
    # model.covar_module.module.base_kernel.lengthscale *= length
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 5000
    prev_loss = 1e10
    with gpytorch.beta_features.checkpoint_kernel(size):
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            print(
                "Iter %d/%d - Loss: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.likelihood.noise.item(),
                )
            )
            optimizer.step()
            if prev_loss - loss < 1e-3 and i > 25:
                break
            else:
                prev_loss = loss

    print(f"Finished training GP at iteration {i}")
    return model, i


def train_model(
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
    train_seq=None,
    train_target=None,
    test_seq=None,
    test_target=None,
    train=None,
    val=None,
    test=None,
    al_full_train=None,
    cv_seed=0,
):
    if model == "ridge":
        lr_model = BayesianRidgeRegression(
            max_iter,
            tol,
            alpha_1,
            alpha_2,
            lambda_1,
            lambda_2,
        )  # initialize model
        lr_trained, _ = train_ridge(
            train_seq, train_target, lr_model
        )  # train and pass back trained model
        if al_full_train is not None:
            test_out_mean, test_out_std = lr_trained.predict(test_seq, return_std=True)
            al_full_train_out_mean, al_full_train_out_std = lr_trained.predict(al_full_train, return_std=True)

    if model == "gp":
        train_seq, train_target = (
            torch.tensor(train_seq).float(),
            torch.tensor(train_target).float(),
        )
        test_seq, test_target = (
            torch.tensor(test_seq).float(),
            torch.tensor(test_target).float(),
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = ExactGPModel(train_seq, train_target, likelihood, device_ids=gpu)
        gp_trained, _ = train_gp(
            train_seq, train_target, gp_model, likelihood, device, length, size
        )

        if al_full_train is not None:
            test_out_mean, test_out_std, _ = pred_gp(test_seq, gp_trained, likelihood, device, size, y_scaler, y=None)
            al_full_train_out_mean, al_full_train_out_std, _ = pred_gp(torch.FloatTensor(al_full_train), gp_trained, likelihood, device, size, y_scaler, y=None)

    if model == "cnn":
        if representation == "ohe":
            cnn_input_type = "ohe"
        if representation == "esm":
            cnn_input_type = "esm_mean"  # TODO: separate into esm_mean and esm_full
            input_size = 1280  # size of ESM mean embeddings is fixed and different from 1024 default for OHE

        if dataset == "meltome":
            batch_size = 30  # smaller batch sizes for meltome since seqs are long
        if representation == "ohe":
            collate = ASCollater(vocab, Tokenizer(vocab), pad=True)
            train_iterator = DataLoader(
                SequenceDataset(train, dataset),
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            val_iterator = DataLoader(
                SequenceDataset(val, dataset),
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            test_iterator = DataLoader(
                SequenceDataset(test, dataset),
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            if al_full_train is not None:
                al_full_train_iterator = DataLoader(
                    SequenceDataset(al_full_train, dataset),
                    collate_fn=collate,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                )
        elif representation == "esm":  # TODO: separate into esm_mean and esm_full
            train_iterator = DataLoader(
                ESMSequenceMeanDataset(train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            val_iterator = DataLoader(
                ESMSequenceMeanDataset(val),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            test_iterator = DataLoader(
                ESMSequenceMeanDataset(test),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            if al_full_train is not None:
                al_full_train_iterator = DataLoader(
                    ESMSequenceMeanDataset(al_full_train),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                )
        if uncertainty == "mve":
            criterion = negative_log_likelihood
        elif uncertainty == "evidential":
            criterion = functools.partial(evidential_loss, lam=regularizer_coeff)
        elif uncertainty == "svi":
            criterion = det_loss
        else:
            criterion = nn.MSELoss()

        EVAL_PATH_BASE = EVAL_PATH
        ensemble_count = 1
        if uncertainty == "ensemble":
            ensemble_count = 5
            train_out_list = []
            test_out_list = []
            if al_full_train is not None:
                al_full_train_out_list = []

        for i in range(ensemble_count):
            # set seeds to ensure different initializations for each member of an ensemble *and* for cross-validation
            np.random.seed(cv_seed * ensemble_count + i)
            random.seed(cv_seed * ensemble_count + i)
            torch.manual_seed(cv_seed * ensemble_count + i)
            if uncertainty == "ensemble":
                EVAL_PATH = EVAL_PATH_BASE / str(i)
                EVAL_PATH.mkdir(parents=True, exist_ok=True)
            # initialize model
            cnn_model = FluorescenceModel(
                len(vocab),
                kernel_size,
                input_size,
                0.0,  # dropout always 0.0 for training
                input_type=cnn_input_type,
                mve=uncertainty == "mve",
                evidential=uncertainty == "evidential",
                svi=uncertainty == "svi",
                n_batches=len(train_iterator),
            )
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
            # train - for CNN, save model
            train_cnn(
                train_iterator,
                val_iterator,
                cnn_model,
                device,
                criterion,
                optimizer,
                100,
                EVAL_PATH,
                mve=uncertainty == "mve",
                evidential=uncertainty == "evidential",
                svi=uncertainty == "svi",
            )

            # evaluate
            train_labels, train_out, train_preds_std = pred_cnn(
                train_iterator,
                cnn_model,
                device,
                EVAL_PATH,
                y_scaler,
                dropout=dropout,
                mve=uncertainty == "mve",
                evidential=uncertainty == "evidential",
                svi=uncertainty == "svi",
            )
            test_labels, test_out, test_preds_std = pred_cnn(
                test_iterator,
                cnn_model,
                device,
                EVAL_PATH,
                y_scaler,
                dropout=dropout,
                mve=uncertainty == "mve",
                evidential=uncertainty == "evidential",
                svi=uncertainty == "svi",
            )
            if al_full_train is not None:
                al_full_train_labels, al_full_train_out, al_full_train_preds_std = pred_cnn(
                    al_full_train_iterator,
                    cnn_model,
                    device,
                    EVAL_PATH,
                    y_scaler,
                    dropout=dropout,
                    mve=uncertainty == "mve",
                    evidential=uncertainty == "evidential",
                    svi=uncertainty == "svi",
                )

            if uncertainty == "ensemble":
                train_out_list.append(train_out)
                test_out_list.append(test_out)
                if al_full_train is not None:
                    al_full_train_out_list.append(al_full_train_out)

        if uncertainty == "ensemble":
            train_out_mean = np.mean(train_out_list, axis=0)
            train_out_std = np.std(train_out_list, axis=0)
            test_out_mean = np.mean(test_out_list, axis=0)
            test_out_std = np.std(test_out_list, axis=0)
            if al_full_train is not None:
                al_full_train_out_mean = np.mean(al_full_train_out_list, axis=0)
                al_full_train_out_std = np.std(al_full_train_out_list, axis=0)
        else:
            train_out_mean = train_out
            train_out_std = train_preds_std
            test_out_mean = test_out
            test_out_std = test_preds_std
            if al_full_train is not None:
                al_full_train_out_mean = al_full_train_out
                al_full_train_out_std = al_full_train_preds_std

    if model != "ridge":
        lr_trained = None
    if model != "gp":
        gp_trained = None
        likelihood = None
    if model != "cnn":
        EVAL_PATH_BASE = None
        train_labels = None
        train_out_mean = None
        train_out_std = None
        test_labels = None
    if model == "cnn":
        train_seq, train_target, test_seq, test_target = None, None, None, None

    if al_full_train is not None:
        return (
            lr_trained,
            gp_trained,
            EVAL_PATH_BASE,
            likelihood,
            train_seq,
            train_target,  # input for ridge and gp
            test_seq,
            test_target,  # input for ridge and gp
            train_labels,  # output from CNN
            train_out_mean,
            train_out_std,
            test_labels,  # output from CNN
            test_out_mean,
            test_out_std,
            al_full_train_out_mean,
            al_full_train_out_std,
        )
    else:
        if model != "cnn":
            test_out_mean = None
            test_out_std = None
        return (
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
        )
