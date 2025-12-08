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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True,
                 version=0,
                 pose_info=None):
        
        super(ST_GCNN_layer_down,self).__init__()
        self.kernel_size = kernel_size
        # assert self.kernel_size[0] % 2 == 1
        # assert self.kernel_size[1] % 2 == 1
        # padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        padding = (0,0)
        
        if version == 0:
            self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim,joints_dim,pose_info=pose_info)
        if type(stride) != list: 
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
        else:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    (self.kernel_size[0], self.kernel_size[1]),
                    (stride[0], stride[1]),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )

                
        
        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        # x=x+res
        x=self.prelu(x)
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
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

class GraphConv(nn.Module):
    """
        adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
        """

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
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        '''
        b, cv, t
        '''

        features = torch.matmul(input, self.weight)  # 35 -> 256
        output = torch.matmul(features.permute(0, 2, 1).contiguous(), self.att).permute(0, 2, 1).contiguous()  # 66 -> 66

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('+ str(self.in_len) + ' -> ' + str(self.out_len) + ')' + ' ('+ str(self.in_node_n) + ' -> ' + str(self.out_node_n) + ')'

class GraphConvBlock(nn.Module):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(GraphConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.resual = residual

        self.out_len = out_len

        self.gcn = GraphConv(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, bias=bias)
        self.bn = nn.BatchNorm1d(out_node_n * out_len)
        self.act = nn.Tanh()
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate)

    def forward(self, input):
        '''

        Args:
            input: b, cv, t

        Returns:

        '''
        x = self.gcn(input)
        b, vc, t = x.shape
        x = self.bn(x.view(b, -1)).view(b, vc, t)
        # x = self.bn(x.view(b, -1, 3, t).permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous().view(b, vc, t)
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.drop(x)

        if self.resual:
            return x + input
        else:
            return x


class ResGCB(nn.Module):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(ResGCB, self).__init__()
        self.resual = residual
        self.gcb1 = GraphConvBlock(in_len, in_len, in_node_n=in_node_n, out_node_n=in_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)
        self.gcb2 = GraphConvBlock(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)


    def forward(self, input):
        '''

        Args:
            x: B,CV,T

        Returns:

        '''

        x = self.gcb1(input)
        x = self.gcb2(x)

        if self.resual:
            return x + input
        else:
            return x

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous() 


class ConvTemporalGraphicalV1(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim,
                 pose_info
    ):
        super(ConvTemporalGraphicalV1,self).__init__()
        parents=pose_info['parents']
        joints_left=list(pose_info['joints_left'])
        joints_right=list(pose_info['joints_right'])
        keep_joints=pose_info['keep_joints']
        dim_use = list(keep_joints)
        # print(dim_use)
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim, joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
        self.A_s = torch.zeros((1,joints_dim,joints_dim), requires_grad=False, dtype=torch.float)
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
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous() 


class ST_GCNN_layer(nn.Module):
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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True,
                 version=0,
                 pose_info=None):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        if version == 0:
            self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim,joints_dim,pose_info=pose_info)

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

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
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

        attn = torch.einsum('nhtvd,nhtwd->nhtvw', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('nhtvw,nhtwd->nhtvd', attn, v)
        out = out.permute(0, 1, 4, 2, 3).contiguous().view(N, -1, T, V)
        out = self.out_proj(out)
        return self.proj_drop(out)

class ST_GAT_layer(nn.Module):
    def __init__(self,
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
                 num_heads=4):

        super().__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)

        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)
        else:
            raise ValueError(f'Unsupported version {version} for ST_GAT_layer')

        self.spatial_attention = SpatialGraphAttention(in_channels, num_heads=num_heads, attn_dropout=dropout)

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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1), bias=bias),
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


class ST_GAT_Compress_layer(nn.Module):
    def __init__(self,
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
                 num_heads=4):

        super().__init__()

        def _to_pair(val):
            if isinstance(val, int):
                return (val, val)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return (val[0], val[1])
            raise ValueError("kernel_size/stride 必须为整数或长度为2的序列")

        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride)
        padding = tuple((k - 1) // 2 for k in self.kernel_size)

        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)
        else:
            raise ValueError(f'Unsupported version {version} for ST_GAT_Compress_layer')

        self.spatial_attention = SpatialGraphAttention(in_channels, num_heads=num_heads, attn_dropout=dropout)

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                self.kernel_size,
                self.stride,
                padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if self.stride != (1, 1) or in_channels != out_channels:
            residual_layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1), bias=bias),
                nn.BatchNorm2d(out_channels),
            ]
            if self.stride != (1, 1):
                residual_layers.append(nn.AvgPool2d(kernel_size=self.stride, stride=self.stride))
            self.residual = nn.Sequential(*residual_layers)
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

        #创建一个形状为 (256, motion_dim) 的随机权重参数矩阵
        self.weight = nn.Parameter(torch.randn(256, motion_dim))

    def forward(self, input):
        # input: (bs*t) x 256

        weight = self.weight + 1e-8
        #执行QR分解获取正交基向量（通过 torch.qr() ），确保生成的方向向量相互正交
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            #将输入转换为对角矩阵，与正交基矩阵进行矩阵乘法运算，然后求和，生成方向向量
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out


class Model(nn.Module):
    def __init__(self, nx, ny,input_channels,st_gcnn_dropout,
                 joints_to_consider,
                 pose_info):
        super(Model, self).__init__()
        self.nx = nx
        self.ny = ny
        self.output_len = 20
        self.num_joints = joints_to_consider
        self.num_D = 30
        if nx == 48:
            self.t_his = 25
            self.t_pred = 100
        elif nx == 42:
            self.t_his = 15
            self.t_pred = 60
        self.nk = 50
        self.num_anchor = self.nk
        self.anchor_feature_dim = 128
        self.feature_dim = self.anchor_feature_dim
        # self.anchor_input = nn.ParameterDict()
        # stdv_anchor = 1. / math.sqrt(128)
        # for i in range(self.num_anchor):
        #     self.anchor_input[f'anchor_{i}'] = nn.Parameter(torch.FloatTensor(1, 128).uniform_(-stdv_anchor, stdv_anchor))
        
        sigma = self._build_skeleton_correlation_matrix(pose_info)
        if sigma is None:
            sigma = torch.empty(0)
        self.register_buffer("skeleton_sigma", sigma)
        if sigma.numel() == 0:
            cholesky = torch.empty(0)
        else:
            jitter = 1e-4 * torch.eye(sigma.shape[0], dtype=sigma.dtype)
            cholesky = torch.linalg.cholesky(sigma + jitter)
        self.register_buffer("L_cholesky", cholesky)

        self.anchor_means = nn.Parameter(torch.zeros(self.nk, self.anchor_feature_dim))
        nn.init.normal_(self.anchor_means, mean=0.0, std=0.02)
        self.structure_proj = nn.Sequential(
            nn.Linear(self.num_joints, 64),
            nn.ReLU(),
            nn.Linear(64, self.anchor_feature_dim),
        )
        nn.init.xavier_uniform_(self.structure_proj[2].weight, gain=1e-2)
        nn.init.zeros_(self.structure_proj[2].bias)
        self.noise_weight = nn.Parameter(torch.tensor(0.001))


        self.st_gcnns_encoder_past_motion=nn.ModuleList()
        #0  input_channels = 3
        self.st_gcnns_encoder_past_motion.append(ST_GCNN_layer(input_channels,128,[3,1],1,self.output_len,
                                           joints_to_consider,st_gcnn_dropout,pose_info=pose_info))
        #1
        self.st_gcnns_encoder_past_motion.append(ST_GCNN_layer(128,64,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))
        #2
        self.st_gcnns_encoder_past_motion.append(ST_GAT_layer(64,128,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))
        #3
        self.st_gcnns_encoder_past_motion.append(ST_GAT_layer(128,128,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info))  
        
        self.st_gcnns_compress=nn.ModuleList()
        #0
        self.st_gcnns_compress.append(ST_GAT_Compress_layer(256,512,[2,2],2,self.output_len,
                                               joints_to_consider,st_gcnn_dropout,  pose_info=pose_info))
        #2
        self.st_gcnns_compress.append(ST_GAT_Compress_layer(512,768,[2,2],2,self.output_len//2,
                                               joints_to_consider//2,st_gcnn_dropout, pose_info=pose_info))

        self.st_gcnns_compress.append(ST_GAT_Compress_layer(768,1024,[2,2],2,self.output_len//4,
                                               joints_to_consider//4,st_gcnn_dropout, pose_info=pose_info))
       
       
        down_fc = [EqualLinear(1024, 1024,activation=True)]
        for i in range(1):
            down_fc.append(EqualLinear(1024, 512,activation=True))

        down_fc.append(EqualLinear(512, self.num_D))
        self.down_fc = nn.Sequential(*down_fc) 
       
        
        self.direction = Direction(motion_dim=self.num_D)
        
        self.st_gcnns_decoder=nn.ModuleList()

        #4
        self.st_gcnns_decoder.append(ST_GCNN_layer(128+256,128,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info)) 
        self.st_gcnns_decoder[-1].gcn.A = self.st_gcnns_encoder_past_motion[-2].gcn.A
        
        #5
        self.st_gcnns_decoder.append(ST_GCNN_layer(128,64,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info))   
        self.st_gcnns_decoder[-1].gcn.A = self.st_gcnns_encoder_past_motion[-1].gcn.A
        #6
        self.st_gcnns_decoder.append(ST_GCNN_layer(64,128,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))
        self.st_gcnns_decoder[-1].gcn.A = self.st_gcnns_decoder[-3].gcn.A
        #7
        self.st_gcnns_decoder.append(ST_GCNN_layer(128,input_channels,[3,1],1,self.output_len,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info))
        

        self.dct_m, self.idct_m = self.get_dct_matrix(self.t_his + self.t_pred)
        

    
    def encode_past_motion(self,x_input):
        #x_input: [t_full, bs, V*C]
       
        # [t_full, bs, V*C] -> [t_full, bs, V, C] -> [bs, c, t_full, v]
        x_input = x_input.view(x_input.shape[0], x_input.shape[1], -1, 3).permute(1, 3, 0, 2)
        y = torch.zeros((x_input.shape[0], x_input.shape[1], self.t_pred, x_input.shape[3])).to(x_input.device) 
        # [bs, c, t_full, v] -> [bs, t_full, c, v]
        x_padding = torch.cat([x_input[:,:,:self.t_his,], y], dim=2).permute(0, 2, 1, 3)
    
        N, T, C, V = x_padding.shape # [bs, t_full, C, V] = [16, 75, 3, 14]
        
        # [bs, t_full, C, V] -> [bs, t_full, C*V]
    
        x_padding = x_padding.reshape([N, T, C * V])
        
        
        dct_m = self.dct_m.to(x_input.device)
        idx_pad = list(range(self.t_his)) + [self.t_his - 1] * self.t_pred #t_his = 14,0-14
       
        # [bs, t_full, C*V] -> [bs, t_full, C*V] 
        x_pad = torch.matmul(dct_m[:self.output_len], x_padding[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)
        x = x_pad # [N, C, T, V] =[16,3,20,14] 关节点= 14 ouput_len = 20
        
        #每次变化的维度只有坐标维度
        for gcn in (self.st_gcnns_encoder_past_motion): #0-3 layer
            x = gcn(x)
        N, C, T, V = x.shape # [16, 128, 20, 14]
        
        return x

    def decoding(self,z,condition=None):
        """
            z：前面encoder和compress的输出特征
            condition：原始输入x
        """
        idct_m = self.idct_m.to(z.device)
   
        condition = condition.view(condition.shape[0], condition.shape[1], -1, 3).permute(1, 3, 0, 2) #[16, 3, 75, 14] (T_his, bs, Num_Joints, 3) -> [bs, t_full, 3, Num_joints] 
        y_condition = torch.zeros((condition.shape[0], condition.shape[1], self.t_pred, condition.shape[3])).to(condition.device) 
        condition_padding = torch.cat([condition[:,:,:self.t_his,:], y_condition], dim=2).permute(0, 2, 1, 3)
        N, T, C, V = condition_padding.shape #16，75，3，14
        
        condition_padding = condition_padding.reshape([N, T, C * V]) #[16, 75, 42]
        dct_m = self.dct_m.to(condition.device) #[75, 75]
        idx_pad = list(range(self.t_his)) + [self.t_his - 1] * self.t_pred # len(idx_pad) = 75
        condition_p = torch.matmul(dct_m[:self.output_len], condition_padding[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)#[800, 3, 20, 14]
        if condition_p.shape[0] != z.shape[0]:
            condition_p = condition_p.repeat_interleave(self.nk, dim=0) #这里nk = 50, [N*50, 3, 20, 14]
        
        #z初始 = 800, 384, 20, 14, 最终 z = [800, 3, 20, 14]
        for gcn in (self.st_gcnns_decoder): #0-3 layer
            z = gcn(z) 
        
        output = (z + condition_p) 
        N, C, N_fre, V = output.shape #[800, 3, 20, 14]
        
        output = output.permute(0, 2, 1, 3).reshape([N, -1, C * V]) #[800, 20, 42]

        outputs = torch.matmul(idct_m[:, :self.output_len], output).reshape([N, -1, C, V]).permute(1, 0, 3, 2).contiguous().view(-1,N,C*V)
       
        return outputs #[75, 800, 42]
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
        if self.L_cholesky.numel() == 0:
            joint_noise = torch.randn(total_queries, self.num_joints, device=device)
        else:
            epsilon = torch.randn(total_queries, self.num_joints, device=device)
            L = self.L_cholesky.to(device)
            joint_noise = torch.matmul(epsilon, L.t())
        mapped_noise = self.structure_proj(joint_noise)
        return mapped_noise

    def forward(self, x, z=None,epoch=None):
        bs = x.shape[1]
        #将编码后的Z进行重复，生成多个候选预测（nk表示候选预测的数量）
        z = self.encode_past_motion(x).repeat_interleave(self.nk,dim=0)
        #构建锚点均值并复制到批次维度
        total_queries = bs * self.nk
        anchors_mean = (
            self.anchor_means.unsqueeze(0)
            .expand(bs, -1, -1)
            .reshape(total_queries, self.anchor_feature_dim)
            .to(z.device)
        )
        #采样结构化噪声并通过结构投影映射到特征空间
        structured_noise = self._sample_structured_noise(total_queries, z.device)
        anchors_input = anchors_mean + self.noise_weight * structured_noise
        anchors_input = anchors_input.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.output_len, self.num_joints)
        z1 = torch.cat((anchors_input, z), dim=1)

        #原文中提到的QLP
        for gcn in (self.st_gcnns_compress): #0-3 layer
            z1 = gcn(z1)
        #生成语义方向
        z1 = z1.mean(-1).mean(-1).view(bs*self.nk,-1)
        alpha = self.down_fc(z1)
        #再通过 Direction 类转换为语义方向向量( directions )。这些方向向量代表了动作预测中的不同语义属性（如速度、方向等）
        directions = self.direction(alpha)
        
        N, C, T, V = z.shape
        #最终特征构建 ：将语义方向向量与原始Z进行拼接
        feature = torch.cat((directions.unsqueeze(2).unsqueeze(3).repeat(1, 1, T, V),z),dim=1)

        
        outputs = self.decoding(feature, x)
       
        return outputs , feature, feature
    
   
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



