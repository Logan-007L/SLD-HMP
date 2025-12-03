import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from models import Orthogonal


class LinQR(nn.Module):
    """Volume-preserving linear layer with learnable QR decomposition."""

    def __init__(self, data_dim):
        super().__init__()
        self.Q = Orthogonal.Orthogonal(d=data_dim)
        self.R = Parameter(torch.Tensor(data_dim, data_dim))
        self.bias = Parameter(torch.Tensor(data_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.R)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, logdet):
        diagR = torch.diag(self.R)
        R = torch.triu(self.R, diagonal=1) + torch.diag(torch.exp(diagR))
        x = x.matmul(R.t())
        x = self.Q(x)
        x = x + self.bias
        logdet = logdet + diagR.sum()
        return x, logdet

    def inverse(self, x, logdet):
        x = x - self.bias
        x = self.Q.inverse(x)
        diagR = torch.diag(self.R)
        R = torch.triu(self.R, diagonal=1) + torch.diag(torch.exp(diagR))
        invR = torch.inverse(R)
        x = x.matmul(invR.t())
        logdet = logdet + diagR.sum()
        return x, logdet


class prelu(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        self.num_parameters = num_parameters
        super(prelu, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input, logdet):
        s = torch.zeros_like(input)
        s[input < 0] = torch.log(self.weight)
        logdet = logdet + torch.sum(s, dim=1)
        return F.prelu(input, self.weight), logdet

    def inverse(self, input, logdet):
        s = torch.zeros_like(input)
        s[input < 0] = torch.log(self.weight)
        logdet = logdet + torch.sum(s, dim=1)
        return F.prelu(input, 1 / self.weight), logdet


class MLP(nn.Module):
    """Simple MLP used inside affine coupling layers."""

    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, data_dim, hidden_dim=64, scale_clip=2.0):
        super().__init__()
        self.data_dim = data_dim
        self.dim1 = data_dim // 2
        self.dim2 = data_dim - self.dim1
        self.scale_clip = scale_clip
        self.net = MLP(in_dim=self.dim1, out_dim=self.dim2 * 2, hidden_dim=hidden_dim)

        perm = torch.randperm(data_dim)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(data_dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x, logdet):
        x_perm = x[:, self.perm]
        x1 = x_perm[:, :self.dim1]
        x2 = x_perm[:, self.dim1:]

        params = self.net(x1)
        log_s, t = params.chunk(2, dim=1)
        log_s = torch.tanh(log_s) * self.scale_clip
        s = torch.exp(log_s)
        y2 = x2 * s + t
        y1 = x1

        z_perm = torch.cat([y1, y2], dim=1)
        z = z_perm[:, self.inv_perm]
        logdet = logdet + torch.sum(log_s, dim=1)
        return z, logdet

    def inverse(self, z, logdet):
        z_perm = z[:, self.perm]
        z1 = z_perm[:, :self.dim1]
        z2 = z_perm[:, self.dim1:]

        params = self.net(z1)
        log_s, t = params.chunk(2, dim=1)
        log_s = torch.tanh(log_s) * self.scale_clip
        s_inv = torch.exp(-log_s)
        x2 = (z2 - t) * s_inv
        x1 = z1

        x_perm = torch.cat([x1, x2], dim=1)
        x = x_perm[:, self.inv_perm]
        logdet = logdet - torch.sum(log_s, dim=1)
        return x, logdet


class LinNF(nn.Module):
    def __init__(self, data_dim, num_layer=5, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(LinQR(data_dim=data_dim))
            self.layers.append(
                AffineCoupling(data_dim=data_dim, hidden_dim=hidden_dim)
            )

    def forward(self, x):
        z = x
        log_det_jacobian = x.new_zeros(x.size(0))
        for layer in self.layers:
            z, log_det_jacobian = layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def inverse(self, z):
        x = z
        log_det_jacobian = z.new_zeros(z.size(0))
        for layer in reversed(self.layers):
            x, log_det_jacobian = layer.inverse(x, log_det_jacobian)
        return x, log_det_jacobian


if __name__ == '__main__':
    bs = 32
    data_dim = 25

    sf = LinNF(data_dim=data_dim)
    sf.double()
    sf.cuda()
    for i in range(10):
        x = torch.randn([bs, data_dim]).double().cuda()

        y1, logdet = sf(x)
        x1, logdet = sf.inverse(y1)
        err = (x1 / x - 1).abs().max()
        print(err)
        print(1)
