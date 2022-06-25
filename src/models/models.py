import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sequence_models.structure import Attention1d
from sklearn.linear_model import BayesianRidge


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


class FluorescenceModel(nn.Module):  # TODO: refactor ensemble, MVE, evidential, SVI, dropout into this
    def __init__(self, n_tokens, kernel_size, input_size, dropout, input_type="ohe"):
        super(FluorescenceModel, self).__init__()
        self.encoder = MaskedConv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.esm_conv = nn.Conv1d(1, 1, kernel_size, padding=2)  # in_channels = out_channels = 1
        self.embedding = LengthMaxPool1D(linear=True, in_dim=input_size, out_dim=input_size * 2)
        self.decoder = nn.Linear(input_size * 2, 1)
        self.n_tokens = n_tokens  # length of vocab (e.g. 22)
        self.dropout = nn.Dropout(dropout)  # TODO: make dropout work at inference time
        self.input_size = input_size  # input vector size (1024 for one hot encodings, 1280 for ESM mean)
        self.input_type = input_type  # choose from "cnn", "esm_mean", or "esm_full"

    def forward(self, x, mask):
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
