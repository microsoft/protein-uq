import argparse
import functools
from csv import writer
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models import LengthMaxPool1D, MaskedConv1d
from train_all import split_dict
from utils import (
    ASCollater,
    SequenceDataset,
    Tokenizer,
    calculate_metrics,
    evidential_loss,
    load_dataset,
    negative_log_likelihood,
    vocab,
)


class FluorescenceModel(nn.Module):
    def __init__(self, n_tokens, kernel_size, input_size, dropout, mve=False, evidential=False):
        super(FluorescenceModel, self).__init__()
        self.encoder = MaskedConv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(linear=True, in_dim=input_size, out_dim=input_size * 2)
        if mve:
            output_size = 2
        elif evidential:
            output_size = 4
        else:
            output_size = 1
        self.decoder = nn.Linear(input_size * 2, output_size)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size

    def forward(self, x, mask, evidential=False):
        # encoder
        x = F.relu(self.encoder(x, input_mask=mask.repeat(self.n_tokens, 1, 1).permute(1, 2, 0)))
        x = x * mask.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        # embed
        x = self.embedding(x)
        x = self.dropout(x)
        # decoder
        output = self.decoder(x)

        if evidential:
            min_val = 1e-6
            # Split the outputs into the four distribution parameters
            means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1] // 4, dim=1)
            lambdas = torch.nn.Softplus()(loglambdas) + min_val  # also called nu or v
            alphas = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
            betas = torch.nn.Softplus()(logbetas) + min_val

            # Return these parameters as the output of the model
            output = torch.stack((means, lambdas, alphas, betas), dim=2).view(output.size())
        return output


def train(args):
    # set up training environment

    batch_size = 256
    epochs = 1000
    device = torch.device("cuda:%d" % args.gpu)
    alphabet = vocab
    if args.dataset == "meltome":
        alphabet += "XU"
        batch_size = 32
    tokenizer = Tokenizer(alphabet)
    print("USING OHE HOT ENCODING")
    if args.mve:
        criterion = negative_log_likelihood
    elif args.evidential:
        criterion = functools.partial(evidential_loss, lam=args.regularizer_coeff)
    else:
        criterion = nn.MSELoss()
    model = FluorescenceModel(
        len(alphabet),
        args.kernel_size,
        args.input_size,
        args.dropout,
        args.mve,
        args.evidential,
    )
    model = model.to(device)
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 1e-3, "weight_decay": 0},
            {"params": model.embedding.parameters(), "lr": 5e-5, "weight_decay": 0.05},
            {"params": model.decoder.parameters(), "lr": 5e-6, "weight_decay": 0.05},
        ]
    )
    patience = 20
    p = 0
    best_rho = -1

    # grab data
    split = split_dict[args.task]
    train, val, test, _ = load_dataset(args.dataset, split + ".csv")

    if args.scale:
        scaler = StandardScaler()
        train[["target"]] = scaler.fit_transform(train[["target"]])
        val[["target"]] = scaler.transform(val[["target"]])
        test[["target"]] = scaler.transform(test[["target"]])

    ds_train = SequenceDataset(train, args.dataset)
    ds_valid = SequenceDataset(val, args.dataset)
    ds_test = SequenceDataset(test, args.dataset)

    # setup dataloaders
    dl_train_AA = DataLoader(
        ds_train,
        collate_fn=ASCollater(alphabet, tokenizer, pad=True, pad_tok=0.0),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    dl_valid_AA = DataLoader(
        ds_valid,
        collate_fn=ASCollater(alphabet, tokenizer, pad=True, pad_tok=0.0),
        batch_size=batch_size,
        num_workers=4,
    )
    dl_test_AA = DataLoader(
        ds_test,
        collate_fn=ASCollater(alphabet, tokenizer, pad=True, pad_tok=0.0),
        batch_size=batch_size,
        num_workers=4,
    )

    def step(model, batch, train=True, dropout_inference=False):
        src, tgt, mask = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        mask = mask.to(device).float()
        output = model(src, mask, args.evidential)
        if args.mve:
            loss = criterion(output[:, 0], output[:, 1], np.squeeze(tgt))
        elif args.evidential:
            loss = criterion(output[:, 0], output[:, 1], output[:, 2], output[:, 3], np.squeeze(tgt))
        else:
            loss = criterion(output, tgt)
        if train and not dropout_inference:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), output.detach().cpu(), tgt.detach().cpu()

    def epoch(model, train, current_step=0, return_values=False, dropout_inference=False):
        start_time = datetime.now()
        if train:
            model = model.train()
            loader = dl_train_AA
            t = "Training"
            n_total = len(ds_train)
        else:
            model = model.eval()
            loader = dl_valid_AA
            t = "Validating"
            n_total = len(ds_valid)
        losses = []
        outputs = []
        tgts = []
        chunk_time = datetime.now()
        n_seen = 0
        for i, batch in enumerate(loader):
            loss, output, tgt = step(model, batch, train, dropout_inference=dropout_inference)
            losses.append(loss)
            outputs.append(output)
            tgts.append(tgt)
            n_seen += len(batch[0])
            if train:
                nsteps = current_step + i + 1
            else:
                nsteps = i
            print(
                "\r%s Epoch %d of %d Step %d Example %d of %d loss = %.4f"
                % (
                    t,
                    e + 1,
                    epochs,
                    nsteps,
                    n_seen,
                    n_total,
                    np.mean(np.array(losses)),
                ),
                end="",
            )
        outputs = torch.cat(outputs).numpy()
        tgts = torch.cat(tgts).cpu().numpy()
        if train and not dropout_inference:
            print("\nTraining complete in " + str(datetime.now() - chunk_time))
            with torch.no_grad():
                _, val_rho = epoch(model, False, current_step=nsteps)
            chunk_time = datetime.now()
        if dropout_inference or not train:
            print("\nValidation complete in " + str(datetime.now() - start_time))
            if args.mve or args.evidential:
                val_rho = spearmanr(tgts, outputs[:, 0]).correlation
                mse = mean_squared_error(tgts, outputs[:, 0])
            else:
                val_rho = spearmanr(tgts, outputs).correlation
                mse = mean_squared_error(tgts, outputs)

            print("Epoch complete in " + str(datetime.now() - chunk_time) + "\n")
        if return_values:
            return i, mse, val_rho, tgts, outputs
        else:
            return i, val_rho

    nsteps = 0
    e = 0
    bestmodel_name = (
        "bestmodel_"
        + str(args.algorithm_type)
        + "_"
        + str(args.task)
        + "_"
        + str(args.kernel_size)
        + "_"
        + str(args.input_size)
        + "_"
        + str(args.dropout)
        + ".tar"
    )
    for e in range(epochs):
        s, val_rho = epoch(model, True, current_step=nsteps)
        # print(val_rho)
        nsteps += s

        """
        if (e%10 == 0) or (e == epochs-1):
            torch.save({
                'step': nsteps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.out_fpath + 'checkpoint%d.tar' % nsteps)
        """
        if val_rho > best_rho:
            p = 0
            best_rho = val_rho
            torch.save(
                {
                    "step": nsteps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.out_fpath + "/bestmodels" + "/" + bestmodel_name,
            )
        else:
            p += 1
        if p == patience:
            print("MET PATIENCE")
            break
    print("Testing...")
    sd = torch.load(args.out_fpath + "/bestmodels" + "/" + bestmodel_name)
    model.load_state_dict(sd["model_state_dict"])
    dl_valid_AA = dl_test_AA

    if args.ensemble and args.dropout > 0.0:
        pre = []
        for i in range(5):
            np.random.seed(i)
            torch.manual_seed(i)
            _, mse, val_rho, tgt, pre_ = epoch(
                model,
                True,
                current_step=nsteps,
                return_values=True,
                dropout_inference=True,
            )
            pre.append(list(pre_))
    else:
        _, mse, val_rho, tgt, pre = epoch(model, False, current_step=nsteps, return_values=True)

    if args.evidential:
        lambdas = pre[:, 1]  # also called nu or v
        alphas = pre[:, 2]
        betas = pre[:, 3]
        pre = pre[:, 0]

        aleatoric_unc = betas / (alphas - 1)
        epistemic_unc = aleatoric_unc / lambdas

    y_train = [ds_train[i][1] for i in range(len(ds_train))]
    y_test = tgt
    y_test_pred = pre

    print("rho = %.2f" % val_rho)
    print("mse = %.2f" % mse)
    np.savez_compressed("preds_cnn/%s.npz" % args.task, prediction=pre, tgt=tgt)
    # with open(Path.cwd() / 'evals_new'/ (args.dataset+'_results.csv'), 'a', newline='') as f:
    #     writer(f).writerow([args.dataset, 'CNN', split, val_rho, mse, e, args.kernel_size, args.input_size, args.dropout])

    if args.scale:
        y_test = scaler.inverse_transform(y_test)
        y_test_pred = scaler.inverse_transform(y_test_pred)

    if args.evidential:
        return y_train, y_test, y_test_pred, aleatoric_unc, epistemic_unc
    else:
        return y_train, y_test, y_test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="file path to data directory")
    parser.add_argument("task", type=str)
    parser.add_argument("out_fpath", type=str, help="save directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ensemble", action="store_true", default=False)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mve", action="store_true", default=False)
    parser.add_argument("--evidential", action="store_true", default=False)
    parser.add_argument("--regularizer_coeff", type=float, default=1.0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()

    if args.cpu > 1:
        torch.set_num_threads(args.cpu)

    if args.ensemble:
        y_test_preds = []
        if args.dropout == 0.0:
            args.algorithm_type = "CNN_ensemble"
            for i in range(5):
                np.random.seed(i)
                torch.manual_seed(i)
                y_train, y_test, y_test_pred = train(args)
                y_test_preds.append(list(y_test_pred))
        else:
            args.algorithm_type = "CNN_dropout"
            np.random.seed(1)
            torch.manual_seed(1)
            y_train, y_test, y_test_preds = train(args)

        y_test_preds = np.squeeze(np.array(y_test_preds))
        y_test = np.squeeze(np.array(y_test))
        y_train = np.squeeze(np.array(y_train))

        preds_mean = np.mean(y_test_preds, axis=0)
        preds_std = np.std(y_test_preds, axis=0)

    elif args.evidential:
        args.algorithm_type = "CNN_evidential"
        np.random.seed(1)
        torch.manual_seed(1)
        y_train, y_test, y_test_preds, aleatoric_unc, epistemic_unc = train(args)

        y_test_preds = np.squeeze(np.array(y_test_preds))
        y_test = np.squeeze(np.array(y_test))
        y_train = np.squeeze(np.array(y_train))
        aleatoric_unc = np.squeeze(np.array(aleatoric_unc))
        epistemic_unc = np.squeeze(np.array(epistemic_unc))

        preds_mean = y_test_preds
        preds_std = np.hstack((np.sqrt(aleatoric_unc), np.sqrt(epistemic_unc))).reshape((-1, 2))

    else:
        if args.mve:
            args.algorithm_type = "CNN_mve"

        np.random.seed(1)
        torch.manual_seed(1)
        y_train, y_test, y_test_preds = train(args)

        if args.mve:
            y_test_preds = np.squeeze(np.array(y_test_preds))
            y_test = np.squeeze(np.array(y_test))
            y_train = np.squeeze(np.array(y_train))

            preds_mean = y_test_preds[:, 0]
            preds_std = np.sqrt(y_test_preds[:, 1])

    if args.ensemble or args.mve or args.evidential:

        metrics = calculate_metrics(
            y_test,
            preds_mean,
            preds_std,
            args,
            args.task,
            y_train,
            args.algorithm_type,
            evidential=args.evidential,
        )

        # Write metric results to file
        row = [args.dataset, args.algorithm_type, split_dict[args.task]]
        for metric in metrics:
            row.append(round(metric, 2))
        with open(Path.cwd() / "evals_new" / (args.dataset + "_results.csv"), "a", newline="") as f:
            writer(f).writerow(row)


if __name__ == "__main__":
    main()
