from dataclasses import dataclass

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sequence_models.structure import Attention1d
from sklearn.linear_model import BayesianRidge
from torch import distributions


class ESMAttention1d(nn.Module):
    """Outputs of the ESM model with the attention1d"""

    def __init__(
        self, max_length, d_embedding
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(ESMAttention1d, self).__init__()
        self.attention1d = Attention1d(in_dim=d_embedding)  # ???
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, 1)

    def forward(self, x, input_mask):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x


class ESMAttention1dMean(nn.Module):
    """Attention1d removed, leaving a basic linear model"""

    def __init__(self, d_embedding):  # [batch x embedding (1280)]  --> [batch x 1]
        super(ESMAttention1dMean, self).__init__()
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, 1)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels (vocab size, e.g. 22 for OHE)
        :param out_channels: output channels (e.g. 1024 for OHE)
        :param kernel_size: the kernel width (e.g. 5)
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2  # 2 for OHE
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias, padding=padding
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask  # x has shape [batch_size, sequence_length, vocab_size] (e.g. [25, 265, 22] for gb1_1 OHE training)
        return super().forward(x.transpose(1, 2)).transpose(1, 2)  # transpose because it expects [batch_size, vocab_size, sequence_length]


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.linear:
            x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class LinearVariational(nn.Module):
    def __init__(self, in_features, out_features, loss_accumulator, n_batches, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.loss_accumulator = loss_accumulator
        self.n_batches = n_batches

        if getattr(loss_accumulator, "accumulated_kl_div", None) is None:
            loss_accumulator.accumulated_kl_div = 0

        self.w_mu = nn.Parameter(torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.001))
        # proxy for variance
        # log(1 + exp(ρ))◦ eps
        self.w_p = nn.Parameter(torch.FloatTensor(in_features, out_features).normal_(mean=-2.5, std=0.001))
        if self.include_bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            self.b_p = nn.Parameter(torch.zeros(out_features))

    def reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def kl_divergence(self, z, mu_theta, p_theta, prior_sd=1):
        log_prior = distributions.Normal(0, prior_sd).log_prob(z)
        log_p_q = distributions.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z)
        return (log_p_q - log_prior).mean() / self.n_batches

    def forward(self, x):
        w = self.reparameterize(self.w_mu, self.w_p)

        if self.include_bias:
            b = self.reparameterize(self.b_mu, self.b_p)
        else:
            b = 0

        z = x @ w + b

        self.loss_accumulator.accumulated_kl_div += self.kl_divergence(
            w,
            self.w_mu,
            self.w_p,
        )
        if self.include_bias:
            self.loss_accumulator.accumulated_kl_div += self.kl_divergence(
                b,
                self.b_mu,
                self.b_p,
            )
        return z


@dataclass
class KL:
    accumulated_kl_div = 0


class FluorescenceModel(nn.Module):  # TODO: refactor ensemble into this
    def __init__(self, n_tokens, kernel_size, input_size, dropout, input_type="ohe", mve=False, evidential=False, svi=False, n_batches=1):
        super(FluorescenceModel, self).__init__()
        self.kl_loss = KL
        self.encoder = MaskedConv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.esm_conv = nn.Conv1d(1, 1, kernel_size, padding=2)  # in_channels = out_channels = 1
        self.embedding = LengthMaxPool1D(linear=True, in_dim=input_size, out_dim=input_size * 2)
        if mve:
            output_size = 2
        elif evidential:
            output_size = 4
        else:
            output_size = 1
        if svi:
            self.decoder = LinearVariational(input_size * 2, output_size, self.kl_loss, n_batches)
        else:
            self.decoder = nn.Linear(input_size * 2, output_size)
        self.n_tokens = n_tokens  # length of vocab (e.g. 22)
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size  # input vector size (1024 for one hot encodings, 1280 for ESM mean)
        self.input_type = input_type  # choose from "cnn", "esm_mean", or "esm_full"

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def forward(self, x, mask, evidential=False):
        # encoder
        if self.input_type == "ohe":
            x = F.relu(self.encoder(x, input_mask=mask.repeat(self.n_tokens, 1, 1).permute(1, 2, 0)))
            x = x * mask.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        elif self.input_type == "esm_mean":
            x = self.esm_conv(x.unsqueeze(1))  # unsqueeze to make shape be [batch_size, 1, 1280]
        elif self.input_type == "esm_full":
            raise ValueError("ESM full not implemented yet")  # TODO: implement this option using attention layer defined above
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


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, device_ids=[0]):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            self.covar_module, device_ids=device_ids, output_device=device_ids[0]
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def BayesianRidgeRegression(max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2):
    return BayesianRidge(
        n_iter=max_iter,
        tol=tol,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
    )
