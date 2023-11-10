import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os, sys

ckconv_source = os.path.join(os.getcwd(), '..')
if ckconv_source not in sys.path:
    sys.path.append(ckconv_source)

import numpy as np
import torch
from torch.nn.utils import weight_norm
from ckconv import*
# from demo_ckconv import CKConv
import ckconv.nn.functional as ckconv_f

from matplotlib import pyplot as plt


in_channels = 1
out_channels = 10
hidden_channels = 32
bias = True
omega_0 = 30.0



def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
    torch.nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(torch.nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = torch.nn.Linear(embedding_dim, projection_dim)
        self.projection2 = torch.nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(torch.nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        torch.nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = torch.nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):

        # print("DIFFCSDI x has input shape ", x.shape)

        B, inputdim, K, L = x.shape #inputdim = 2:::: x.shape: (16,2,2,51)=(B,2,K,L)

        x = x.reshape(B, inputdim, K * L)
        # print("DIFFCSDI  x after input projection", x.shape)

        x = self.input_projection(x)
        # print("DIFFCSDI x after input projection", x.shape)

        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        # print("DIFFCSDI x reshape after relu", x.shape)


        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # print("DIFFCSDI x after Residual", x.shape)


        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        # print("DIFFCSDI x after ALL NET Residual", x.shape)

        x = x.reshape(B, self.channels, K * L)
        # print("DIFFCSDI x after prior to output reshape ", x.shape)

        x = self.output_projection1(x)  # (B,channel,K*L)
        # print("DIFFCSDI x after output projection ", x.shape)
        x = F.relu(x)
        # print("DIFFCSDI x after Filter output projection ", x.shape)

        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        # print("DIFFCSDI x as final ouutput of model ", x.shape)
        # print("#######################################")
        # print("#######################################")

        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = torch.nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)


        # self.ckconv_layer_time = ckconv.CKConv(in_channels=64, out_channels=64, hidden_channels=8, bias=True, omega_0=3.0,
        #                            activation_function='Sine', norm_type='', dim_linear=1, weight_dropout=0.0)
        #
        # self.ckconv_layer_feature = ckconv.CKConv(in_channels=64, out_channels=64, hidden_channels=8, bias=True, omega_0=3.0,
        #                            activation_function='Sine', norm_type='', dim_linear=1, weight_dropout=0.0)

        self.ckconv_layer_time = ckconv.CKConv(in_channels=16, out_channels=16, hidden_channels=16, bias=True, omega_0=9.0,
                                   activation_function='Sine', norm_type='', dim_linear=1, weight_dropout=0.0)

        self.ckconv_layer_feature = ckconv.CKConv(in_channels=16, out_channels=16, hidden_channels=16, bias=True, omega_0=9.0,
                                   activation_function='Sine', norm_type='', dim_linear=1, weight_dropout=0.0)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_ckconv_time(self, y, base_shape):
        B, channel, K, L = base_shape
        # self.ckconv_layer_time.rel_positions = None
        if K * L < 100:
            return y
        y_shape0 = y.shape #16 64 102  #16 64 1680
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y_shape1 = y #32 64 51 #560 64 48
        # nnn = y.permute(2, 0, 1) #51,32,64 #48 560 64
        nnn = y.permute(0,1,2) #51,32,64 #48 560 64
        y = self.ckconv_layer_time(nnn)
        # y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y_shape2 = y.shape         # 32,64,51  # 48 64 64
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        y_shape3 = y.shape         # 16,64,102
        return y

    # B I L        B 16    K: 2  35   L 51  48
    # 51 32 64             48 560 64
    #
    # B  O  L
    # 32 64 51             506 64 64

    # def forward_time(self, y, base_shape):
    #
    #     B, channel, K, L = base_shape
    #
    #     if L == 1:
    #         return y
    #     y_shape0 = y.shape
    #     y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
    #     y_shape1 = y.shape
    #     nnn = y.permute(2, 0, 1) #51,32,64
    #     ya = self.time_layer(nnn)#51,32,64
    #     ay = ya.permute(1, 2, 0)#32,64,51
    #     y = ay
    #     y_shape2 = y.shape
    #     y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
    #     y_shape3 = y.shape
    #     return y

    def forward_ckconv_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K*L < 100:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)

        ccc = y.permute(2, 0, 1)
        y_fll = ccc.permute(1, 2, 0)
        y_fl = self.ckconv_layer_feature(y_fll)
        op = y_fl.permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)


        return y
    #
    # def forward_feature(self, y, base_shape):
    #     B, channel, K, L = base_shape
    #     if K == 1:
    #         return y
    #     y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
    #     ttt = y.permute(2, 0, 1)
    #     y = self.feature_layer(ttt)
    #
    #     ppp = y.permute(1, 2, 0)
    #     y = ppp.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
    #
    #     return y

    def forward(self, x, cond_info, diffusion_emb):

        # print("residual input x shape", x.shape)
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        # print("diffusion input diffusion_emb shape", diffusion_emb.shape)


        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)


        y = x + diffusion_emb # y: 16 64 102

        y = self.forward_ckconv_time(y, base_shape) #y: 16 64 102 # for physio: 102: 48*35:    16 64 1680
        y = self.forward_ckconv_feature(y, base_shape)  # y: 16 64 102
        #
        # y = self.forward_ckconv_time(y, base_shape) #y: 16 64 102
        # # y = self.forward_time(y, base_shape)
        # y = self.forward_ckconv_feature(y, base_shape)  # y: 16 64
        #
        # y = self.forward_ckconv_time(y, base_shape) #y: 16 64 102
        # y = self.forward_ckconv_feature(y, base_shape) #y: 16 64 102 # (B,channel,K*L)
        # # y = self.forward_feature(y, base_shape)  # (B,channel,K*L)




        y = self.mid_projection(y)  # (B,2*channel,K*L)



        _, cond_dim, _, _ = cond_info.shape  #  (B, cond_dim, K, L) (16, 145, 2, 51)  ; 145 = feature emb+ time emb +1 = 16+128+1
        cond_info = cond_info.reshape(B, cond_dim, K * L)

        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        # print(" Y shape before cond_info add  ", y.shape)
        # print(" cond_info shape add  ", cond_info.shape)

        y = y + cond_info

        # print(" Y shape after add  ", y.shape)


        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        # print(" y after filtering shape ", y.shape)

        y = self.output_projection(y)
        # print(" y final output projection ", y.shape)


        residual, skip = torch.chunk(y, 2, dim=1)

        # print(" y shape after residual  projection ", residual.shape)
        #


        x = x.reshape(base_shape)
        # print(" x shape as base shape ", x.shape)

        residual = residual.reshape(base_shape)
        # print(" residual as   base shape ", residual.shape)
        # print("........................")

        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip







#################


#
# diff_CSDI(
#   (diffusion_embedding): DiffusionEmbedding(
#     (projection1): Linear(in_features=128, out_features=128, bias=True)
#     (projection2): Linear(in_features=128, out_features=128, bias=True)
#   )
#   (input_projection): Conv1d(2, 64, kernel_size=(1,), stride=(1,))
#   (output_projection1): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
#   (output_projection2): Conv1d(64, 1, kernel_size=(1,), stride=(1,))
#   (residual_layers): ModuleList(
#     (0-3): 4 x ResidualBlock(
#       (diffusion_projection): Linear(in_features=128, out_features=64, bias=True)
#       (cond_projection): Conv1d(145, 128, kernel_size=(1,), stride=(1,))
#       (mid_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
#       (output_projection): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
#       (time_layer): TransformerEncoder(
#         (layers): ModuleList(
#           (0): TransformerEncoderLayer(
#             (self_attn): MultiheadAttention(
#               (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
#             )
#             (linear1): Linear(in_features=64, out_features=64, bias=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#             (linear2): Linear(in_features=64, out_features=64, bias=True)
#             (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (dropout1): Dropout(p=0.1, inplace=False)
#             (dropout2): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#       (feature_layer): TransformerEncoder(
#         (layers): ModuleList(
#           (0): TransformerEncoderLayer(
#             (self_attn): MultiheadAttention(
#               (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
#             )
#             (linear1): Linear(in_features=64, out_features=64, bias=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#             (linear2): Linear(in_features=64, out_features=64, bias=True)
#             (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (dropout1): Dropout(p=0.1, inplace=False)
#             (dropout2): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#     )
#   )
# )

