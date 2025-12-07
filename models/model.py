#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class ST_GCNN_layer_down(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        time_dim,
        joints_dim,
        dropout,
        bias=True,
        version=0,
        pose_info=None,
    ):
        super(ST_GCNN_layer_down, self).__init__()
        self.kernel_size = kernel_size
        padding = (0, 0)

        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)

        if isinstance(stride, list):
            stride_t, stride_v = stride
        else:
            stride_t = stride
            stride_v = stride

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride_t, stride_v),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        # x = x + res
        x = self.prelu(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class GraphConv(nn.Module):
    """adapted from : https://github.com/tkipf/gcn"""

    def __init__(self, in_len, out_len, in_node_n=66, out_node_n=66, bias=True):
        super(GraphConv, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_node_n = in_node_n
        self.out_node_n = out_node_n
        self.weight = nn.Parameter(torch.FloatTensor(in_len, out_len))
        self.att = nn.Parameter(torch.FloatTensor(in_node_n, out_node_n))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_len))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        features = torch.matmul(input, self.weight)
        output = (
            torch.matmul(features.permute(0, 2, 1).contiguous(), self.att)
            .permute(0, 2, 1)
            .contiguous()
        )

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_len)
            + " -> "
            + str(self.out_len)
            + ") ("
            + str(self.in_node_n)
            + " -> "
            + str(self.out_node_n)
            + ")"
        )


class GraphConvBlock(nn.Module):
    def __init__(
        self,
        in_len,
        out_len,
        in_node_n,
        out_node_n,
        dropout_rate=0,
        leaky=0.1,
        bias=True,
        residual=False,
    ):
        super(GraphConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.resual = residual

        self.out_len = out_len

        self.gcn = GraphConv(
            in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_node_n * out_len)
        self.act = nn.Tanh()
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate)

    def forward(self, input):
        x = self.gcn(input)
        b, vc, t = x.shape
        x = self.bn(x.view(b, -1)).view(b, vc, t)
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.drop(x)

        if self.resual:
            return x + input
        else:
            return x


class ResGCB(nn.Module):
    def __init__(
        self,
        in_len,
        out_len,
        in_node_n,
        out_node_n,
        dropout_rate=0,
        leaky=0.1,
        bias=True,
        residual=False,
    ):
        super(ResGCB, self).__init__()
        self.resual = residual
        self.gcb1 = GraphConvBlock(
            in_len,
            in_len,
            in_node_n=in_node_n,
            out_node_n=in_node_n,
            dropout_rate=dropout_rate,
            bias=bias,
            residual=False,
        )
        self.gcb2 = GraphConvBlock(
            in_len,
            out_len,
            in_node_n=in_node_n,
            out_node_n=out_node_n,
            dropout_rate=dropout_rate,
            bias=bias,
            residual=False,
        )

    def forward(self, input):
        x = self.gcb1(input)
        x = self.gcb2(x)

        if self.resual:
            return x + input
        else:
            return x


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution."""

    def __init__(self, time_dim, joints_dim):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim, joints_dim)
        )
        stdv = 1.0 / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        x = torch.einsum("nctv,tvw->nctw", (x, self.A))
        return x.contiguous()


class ConvTemporalGraphicalV1(nn.Module):
    """Graph convolution with pose prior."""

    def __init__(self, time_dim, joints_dim, pose_info):
        super(ConvTemporalGraphicalV1, self).__init__()
        parents = pose_info["parents"]
        joints_left = list(pose_info["joints_left"])
        joints_right = list(pose_info["joints_right"])
        keep_joints = pose_info["keep_joints"]
        dim_use = list(keep_joints)

        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim, joints_dim)
        )
        stdv = 1.0 / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        self.A_s = torch.zeros((1, joints_dim, joints_dim), requires_grad=False, dtype=torch.float)
        for i, dim in enumerate(dim_use):
            self.A_s[0][i][i] = 1
            if parents[dim] in dim_use:
                parent_index = dim_use.index(parents[dim])
                self.A_s[0][i][parent_index] = 1
                self.A_s[0][parent_index][i] = 1
            if dim in joints_left:
                index = joints_left.index(dim)
                right_dim = joints_right[index]
                right_index = dim_use.index(right_dim)
                if right_dim in dim_use:
                    self.A_s[0][i][right_index] = 1
                    self.A_s[0][right_index][i] = 1

        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        A = self.A * self.A_s.to(x.device)
        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        x = torch.einsum("nctv,tvw->nctw", (x, A))
        return x.contiguous()


class ST_GCNN_layer(nn.Module):
    """Space-time GCN layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        time_dim,
        joints_dim,
        dropout,
        bias=True,
        version=0,
        pose_info=None,
    ):
        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


class SpatialGraphAttention(nn.Module):
    def __init__(self, channels, num_heads=4, attn_dropout=0.1):
        super().__init__()
        num_heads = max(1, min(num_heads, channels))
        if channels % num_heads != 0:
            for candidate in range(num_heads, 0, -1):
                if channels % candidate == 0:
                    num_heads = candidate
                    break
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        N, C, T, V = x.shape
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim, T, V).permute(0, 1, 3, 4, 2)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim, T, V).permute(0, 1, 3, 4, 2)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim, T, V).permute(0, 1, 3, 4, 2)

        attn = torch.einsum("nhtvd,nhtwd->nhtvw", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("nhtvw,nhtwd->nhtvd", attn, v)
        out = out.permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, V)
        out = self.out_proj(out)
        return self.proj_drop(out)


class ST_GAT_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        time_dim,
        joints_dim,
        dropout,
        bias=True,
        version=0,
        pose_info=None,
        num_heads=4,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)
        else:
            raise ValueError(f"Unsupported version {version} for ST_GAT_layer")

        self.spatial_attention = SpatialGraphAttention(
            in_channels, num_heads=num_heads, attn_dropout=dropout
        )

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1),
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x_physical = self.gcn(x)
        x_semantic = self.spatial_attention(x)
        x_fused = x_physical + x_semantic
        x_out = self.tcn(x_fused)
        return self.prelu(x_out + res)


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(256, motion_dim))

    def forward(self, input):
        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out


class Model(nn.Module):
    def __init__(
        self,
        nx,
        ny,
        input_channels,
        st_gcnn_dropout,
        joints_to_consider,
        pose_info,
    ):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.nx = nx
        self.ny = ny
        self.output_len = 20
        self.num_joints = joints_to_consider
        self.anchor_feature_dim = 128
        self.num_D = 30
        if nx == 48:
            self.t_his = 25
            self.t_pred = 100
        elif nx == 42:
            self.t_his = 15
            self.t_pred = 60
        self.nk = 50
        sigma = self._build_skeleton_correlation_matrix(pose_info)
        if sigma is None:
            sigma = torch.empty(0)
        self.register_buffer("skeleton_sigma", sigma)
        if sigma.numel() == 0:
            cholesky = torch.empty(0)
        else:
            jitter = 1e-4 * torch.eye(sigma.shape[0], dtype=sigma.dtype)
            cholesky = torch.linalg.cholesky(sigma + jitter)
        self.register_buffer("skeleton_cholesky", cholesky)

        self.anchor_means = nn.Parameter(torch.zeros(self.nk, self.anchor_feature_dim))
        nn.init.normal_(self.anchor_means, mean=0.0, std=0.02)
        self.structure_proj = nn.Linear(self.num_joints, self.anchor_feature_dim)
        nn.init.xavier_uniform_(self.structure_proj.weight, gain=1e-2)
        nn.init.zeros_(self.structure_proj.bias)
        self.noise_weight = nn.Parameter(torch.tensor(0.1))

        self.st_gcnns_encoder_past_motion = nn.ModuleList()
        self.st_gcnns_encoder_past_motion.append(
            ST_GCNN_layer(
                input_channels,
                128,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_encoder_past_motion.append(
            ST_GCNN_layer(
                128,
                64,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                version=1,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_encoder_past_motion.append(
            ST_GAT_layer(
                64,
                128,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                version=1,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_encoder_past_motion.append(
            ST_GAT_layer(
                128,
                128,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )

        self.st_gcnns_compress = nn.ModuleList()
        joints_dim_lvl1 = max(1, self.num_joints // 2)
        joints_dim_lvl2 = max(1, self.num_joints // 4)
        self.st_gcnns_compress.append(
            ST_GCNN_layer_down(
                256,
                512,
                [2, 2],
                2,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_compress.append(
            ST_GCNN_layer_down(
                512,
                768,
                [2, 2],
                2,
                self.output_len // 2,
                joints_dim_lvl1,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_compress.append(
            ST_GCNN_layer_down(
                768,
                1024,
                [2, 2],
                2,
                self.output_len // 4,
                joints_dim_lvl2,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )

        down_fc = [EqualLinear(1024, 1024, activation=True)]
        down_fc.append(EqualLinear(1024, 512, activation=True))
        down_fc.append(EqualLinear(512, self.num_D))
        self.down_fc = nn.Sequential(*down_fc)

        self.direction = Direction(motion_dim=self.num_D)

        self.st_gcnns_decoder = nn.ModuleList()
        self.st_gcnns_decoder.append(
            ST_GCNN_layer(
                128 + 256,
                128,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                version=1,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_decoder[-1].gcn.A = self.st_gcnns_encoder_past_motion[-2].gcn.A

        self.st_gcnns_decoder.append(
            ST_GCNN_layer(
                128,
                64,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_decoder[-1].gcn.A = self.st_gcnns_encoder_past_motion[-1].gcn.A

        self.st_gcnns_decoder.append(
            ST_GCNN_layer(
                64,
                128,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                version=1,
                pose_info=pose_info,
            )
        )
        self.st_gcnns_decoder[-1].gcn.A = self.st_gcnns_decoder[-3].gcn.A

        self.st_gcnns_decoder.append(
            ST_GCNN_layer(
                128,
                input_channels,
                [3, 1],
                1,
                self.output_len,
                self.num_joints,
                st_gcnn_dropout,
                pose_info=pose_info,
            )
        )

        self.dct_m, self.idct_m = self.get_dct_matrix(self.t_his + self.t_pred)

    def encode_past_motion(self, x_input):
        x_input = x_input.view(x_input.shape[0], x_input.shape[1], -1, 3).permute(1, 3, 0, 2)
        y = torch.zeros(
            (x_input.shape[0], x_input.shape[1], self.t_pred, x_input.shape[3]),
            device=x_input.device,
        )
        x_padding = torch.cat([x_input[:, :, : self.t_his, :], y], dim=2).permute(0, 2, 1, 3)

        N, T, C, V = x_padding.shape
        x_padding = x_padding.reshape([N, T, C * V])

        dct_m = self.dct_m.to(x_input.device)
        idx_pad = list(range(self.t_his)) + [self.t_his - 1] * self.t_pred

        x = (
            torch.matmul(dct_m[: self.output_len], x_padding[:, idx_pad, :])
            .reshape([N, -1, C, V])
            .permute(0, 2, 1, 3)
        )

        for gcn in self.st_gcnns_encoder_past_motion:
            x = gcn(x)
        return x

    def decoding(self, z, condition=None):
        idct_m = self.idct_m.to(z.device)

        condition = condition.view(condition.shape[0], condition.shape[1], -1, 3).permute(1, 3, 0, 2)
        y_condition = torch.zeros(
            (condition.shape[0], condition.shape[1], self.t_pred, condition.shape[3]),
            device=condition.device,
        )
        condition_padding = torch.cat(
            [condition[:, :, : self.t_his, :], y_condition], dim=2
        ).permute(0, 2, 1, 3)
        N, T, C, V = condition_padding.shape

        condition_padding = condition_padding.reshape([N, T, C * V])
        dct_m = self.dct_m.to(condition.device)
        idx_pad = list(range(self.t_his)) + [self.t_his - 1] * self.t_pred
        condition_p = (
            torch.matmul(dct_m[: self.output_len], condition_padding[:, idx_pad, :])
            .reshape([N, -1, C, V])
            .permute(0, 2, 1, 3)
        )
        if condition_p.shape[0] != z.shape[0]:
            condition_p = condition_p.repeat_interleave(self.nk, dim=0)

        for gcn in self.st_gcnns_decoder:
            z = gcn(z)

        output = z + condition_p
        N, C, N_fre, V = output.shape

        output = output.permute(0, 2, 1, 3).reshape([N, -1, C * V])
        outputs = (
            torch.matmul(idct_m[:, : self.output_len], output)
            .reshape([N, -1, C, V])
            .permute(1, 0, 3, 2)
            .contiguous()
            .view(-1, N, C * V)
        )

        return outputs

    def forward(self, x, z=None, epoch=None):
        bs = x.shape[1]
        z = self.encode_past_motion(x).repeat_interleave(self.nk, dim=0)
        anchors_input = self._generate_motion_queries(bs, z.device)
        z1 = torch.cat((anchors_input, z), dim=1)

        for gcn in self.st_gcnns_compress:
            z1 = gcn(z1)
        z1 = z1.mean(-1).mean(-1).view(bs * self.nk, -1)
        alpha = self.down_fc(z1)
        directions = self.direction(alpha)

        N, C, T, V = z.shape
        feature = torch.cat(
            (directions.unsqueeze(2).unsqueeze(3).repeat(1, 1, T, V), z), dim=1
        )

        outputs = self.decoding(feature, x)

        return outputs, feature, feature

    def _build_skeleton_correlation_matrix(self, pose_info):
        if pose_info is None:
            return None
        parents = pose_info.get("parents")
        keep_joints = pose_info.get("keep_joints")
        if parents is None:
            return None
        if keep_joints is None or len(keep_joints) == 0:
            keep_joints = list(range(self.num_joints))
        keep_joints = list(map(int, keep_joints))
        if len(keep_joints) > self.num_joints:
            keep_joints = keep_joints[: self.num_joints]
        joint_map = {joint: idx for idx, joint in enumerate(keep_joints)}
        num_joints = len(joint_map)
        if num_joints == 0:
            return None
        adjacency = torch.zeros((num_joints, num_joints), dtype=torch.float32)
        for joint in keep_joints:
            joint_idx = joint_map[joint]
            parent = parents[joint]
            if parent == -1 or parent not in joint_map:
                continue
            parent_idx = joint_map[parent]
            adjacency[joint_idx, parent_idx] = 1.0
            adjacency[parent_idx, joint_idx] = 1.0
        adjacency = adjacency + torch.eye(num_joints, dtype=torch.float32)
        eigenvalues = torch.linalg.eigvalsh(adjacency)
        lambda_min = torch.min(eigenvalues)
        lambda_max = torch.max(eigenvalues)
        denom = (lambda_max - lambda_min).clamp_min(1e-6)
        sigma = adjacency - lambda_min * torch.eye(num_joints, dtype=torch.float32)
        sigma = sigma / denom
        return sigma

    def _sample_structured_noise(self, total_queries, device):
        if self.skeleton_cholesky.numel() == 0:
            joint_noise = torch.randn(total_queries, self.num_joints, device=device)
        else:
            L = self.skeleton_cholesky.to(device)
            epsilon = torch.randn(total_queries, self.num_joints, device=device)
            joint_noise = torch.matmul(epsilon, L.t())
        mapped_noise = self.structure_proj(joint_noise)
        return mapped_noise

    def _generate_motion_queries(self, batch_size, device):
        total_queries = batch_size * self.nk
        anchors_mean = (
            self.anchor_means.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(total_queries, self.anchor_feature_dim)
        )
        anchors_mean = anchors_mean.to(device)
        structured_noise = self._sample_structured_noise(total_queries, device)
        anchors = anchors_mean + self.noise_weight * structured_noise
        anchors = anchors.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.output_len, self.num_joints)
        return anchors

    def get_dct_matrix(self, N, is_torch=True):
        dct_m = np.eye(N, dtype=np.float32)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        if is_torch:
            dct_m = torch.from_numpy(dct_m)
            idct_m = torch.from_numpy(idct_m)
        return dct_m, idct_m
