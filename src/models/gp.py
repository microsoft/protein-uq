import gpytorch

import argparse
from sklearn.preprocessing import StandardScaler

import torch
import numpy as np
import torch.nn.functional as F

from train_all import split_dict 
from utils import SequenceDataset, load_dataset, calculate_metrics
from csv import writer
from pathlib import Path

class Tokenizer(object):
    """Convert between strings and their one-hot representations."""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x) -> str:
        return ''.join([self.t_to_a[t] for t in x])

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, device_ids=[0]):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            self.covar_module, device_ids=device_ids,
            output_device=device_ids[0]
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

np.random.seed(0)
torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='file path to data directory')
parser.add_argument('task', type=str)
parser.add_argument('--scale', action='store_true')
parser.add_argument('--gpu', type=int, nargs='+', default=0)
parser.add_argument('--size', type=int, default=0)
parser.add_argument('--length', type=float, default=1.0)
args = parser.parse_args()

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYVXU'
# grab data
split = split_dict[args.task]
train, test, _ = load_dataset(args.dataset, split+'.csv', val_split=False)

ds_train = SequenceDataset(train, args.dataset)
ds_test = SequenceDataset(test, args.dataset)

print('Encoding...')
# tokenize data
all_train = list(ds_train)
X_train = [i[0] for i in all_train]
y_train = [i[1] for i in all_train]
all_test = list(ds_test)
X_test = [i[0] for i in all_test]
y_test = [i[1] for i in all_test]

tokenizer = Tokenizer(AAINDEX_ALPHABET) # tokenize
X_train = [torch.tensor(tokenizer.tokenize(i)).view(-1, 1) for i in X_train]
X_test = [torch.tensor(tokenizer.tokenize(i)).view(-1,1) for i in X_test]

# padding
maxlen_train = max([len(i) for i in X_train])
maxlen_test = max([len(i) for i in X_test])
maxlen = max([maxlen_train, maxlen_test])
# pad_tok = alphabet.index(PAD)

X_train = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]), "constant", 0.) for i in X_train]
X_train_enc = [] # ohe
for i in X_train:
    i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
    i_onehot.zero_()
    i_onehot.scatter_(1, i, 1)
    X_train_enc.append(i_onehot)
X_train_enc = np.array([np.array(i.view(-1)) for i in X_train_enc]) # flatten

X_test = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", 0.) for i in X_test]
X_test_enc = [] # ohe
for i in X_test:
    i_onehot = torch.FloatTensor(maxlen, len(AAINDEX_ALPHABET))
    i_onehot.zero_()
    i_onehot.scatter_(1, i, 1)
    X_test_enc.append(i_onehot)
X_test_enc = np.array([np.array(i.view(-1)) for i in X_test_enc]) # flatten


# scale X
if args.scale:
    scaler = StandardScaler()
    X_train_enc = scaler.fit_transform(X_train_enc)
    X_test_enc = scaler.transform(X_test_enc)
    y_train = scaler.fit_transform(np.array(y_train)[:, None])[:, 0]
    y_test = scaler.transform(np.array(y_test)[:, None])[:, 0]

train_x, train_y = torch.tensor(X_train_enc).float(), torch.tensor(y_train).float()
test_x, test_y = torch.tensor(X_test_enc).float(), torch.tensor(y_test).float()

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood, device_ids=args.gpu)
model.covar_module.module.base_kernel.lengthscale *= args.length
device = torch.device('cuda:%d' %args.gpu[0])
train_x = train_x.to(device)
train_y = train_y.to(device)
model = model.to(device)
likelihood = likelihood.to(device)

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
print('Training...')
training_iter = 5000
prev_loss = 1e10
with gpytorch.beta_features.checkpoint_kernel(args.size):
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
        if prev_loss - loss < 1e-3 and i > 25:
            break
        else:
            prev_loss = loss



# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(args.size):
    observed_pred = likelihood(model(test_x.to(device)))
    preds_mean = observed_pred.mean.cpu()
    lower, upper = observed_pred.confidence_region() # 2 standard deviations above and below mean

lower = lower.cpu()
upper = upper.cpu()
preds_std = (upper-preds_mean)/2

train_x = train_x.cpu()
train_y = train_y.cpu()
test_x = test_x.cpu()


print('Calculating metrics...')
algorithm_type = 'GPcontinuous'
metrics = calculate_metrics(np.array(y_test), preds_mean.numpy(), preds_std.numpy(), args, split, y_train, algorithm_type)

# Write metric results to file
row = [args.dataset, algorithm_type, split]
for metric in metrics:
    row.append(round(metric, 2))
with open(Path.cwd() / 'evals_new'/ (args.dataset+'_results.csv'), 'a', newline='') as f:
    writer(f).writerow(row)
