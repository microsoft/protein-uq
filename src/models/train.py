import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import gpytorch


def train_esm(train_iterator, val_iterator, model, device, criterion, optimizer, epoch_num, mean):

    val_loss = []

    model.to(device)

    for epoch in range(epoch_num):
        model.train()
        for i, (inp, l) in enumerate(train_iterator):
            optimizer.zero_grad()

            inp = inp.to(device)
            l = l.to(device)

            if mean:
                o = model(inp)
            else:
                m = (inp[:, :, 0] != 0).long().to(device)
                o = model(inp, m)

            loss = criterion(o, l.unsqueeze(-1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():  # evaluate validation loss here

            model.eval()
            val_loss_epochs = []

            for (inp, l) in val_iterator:

                inp = inp.to(device)
                l = l.to(device)

                if mean:
                    o = model(inp)
                else:
                    m = (inp[:, :, 0] != 0).long().to(device)
                    o = model(inp, m)  # Forward prop without storing gradients

                loss = criterion(o, l.unsqueeze(-1))  # Calculate validation loss
                val_loss_epochs.append(loss.item())

            val_loss_epoch = np.mean(val_loss_epochs)
            val_loss.append(round(val_loss_epoch, 3))

        print("epoch: %d loss: %.3f val loss: %.3f" % (epoch + 1, loss.item(), val_loss_epoch))

        # evalutate whether validation loss is dropping; if not, stop
        if epoch > 21:
            if val_loss[-1] >= np.min(val_loss[:-20]):
                print("Finished training at epoch {0}".format(epoch))
                return epoch


def train_cnn(train_iterator, val_iterator, model, device, criterion, optimizer, epoch_num, MODEL_PATH):

    patience = 20
    p = 0
    best_rho = -1
    model = model.to(device)

    def step(model, batch, train=True):
        src, tgt, mask = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        mask = mask.to(device).float()
        output = model(src, mask)
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
            t = "Training"
            n_total = len(train_iterator)
        else:
            model = model.eval()
            loader = val_iterator
            t = "Validating"
            n_total = len(val_iterator)

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
            print("epoch: %d loss: %.3f val loss: %.3f" % (e + 1, loss, val_rho))

        if not train:
            val_rho = spearmanr(tgts, outputs).correlation
            mse = mean_squared_error(tgts, outputs)

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
                {"step": nsteps, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                bestmodel_save,
            )

        else:
            p += 1
        if p == patience:
            print("MET PATIENCE")
            print("Finished training at epoch {0}".format(e))
            return e

    print("Finished training CNN at epoch {0}".format(epoch_num))
    return e


def train_ridge(X_train, y_train, model):
    model.fit(X_train, y_train)
    iterations = model.n_iter_[0]
    print("Finished training BayesianRidge at iteration {0}".format(iterations))
    return model, iterations


def train_gp(X_train, y_train, model, device, length, size):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.covar_module.module.base_kernel.lengthscale *= length
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

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
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.covar_module.module.base_kernel.lengthscale.item(),
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
