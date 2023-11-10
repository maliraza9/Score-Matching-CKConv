# Append .. to path
import os, sys

ckconv_source = os.path.join(os.getcwd(), '..')
if ckconv_source not in sys.path:
    sys.path.append(ckconv_source)

import numpy as np
import torch
from torch.nn.utils import weight_norm

import ckconv

from matplotlib import pyplot as plt



in_channels = 10
out_channels = 10
hidden_channels = 32
activation_function = 'Sine'
norm_type = ''
dim_linear = 1
bias = True
omega_0 = 30.5
weight_dropout = 0.0

ckconv_example = ckconv.CKConv(in_channels,
                              out_channels,
                              hidden_channels,
                              activation_function,
                              norm_type,
                              dim_linear,
                              bias,
                              omega_0,
                              weight_dropout,
                              )


# Example:
in_channels = 10
out_channels = 10
kernelnet_hidden_channels = 32
kernelnet_activation_function = 'Sine'
kernelnet_norm_type = ''
dim_linear = 1
bias = True
omega_0 = 30.5
dropout = 0.1
weight_dropout = 0.0

ckconv_example = ckconv.CKBlock(in_channels,
                                out_channels,
                                kernelnet_hidden_channels,
                                kernelnet_activation_function,
                                kernelnet_norm_type,
                                dim_linear,
                                bias,
                                omega_0,
                                dropout,
                                weight_dropout,
                               )


class CKCNN_backbone(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_blocks: int,
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,
            dropout: float,
            weight_dropout: float,
    ):
        super().__init__()

        # Add num_blocks CKBlocks to a sequential called self.backbone
        blocks = []
        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else hidden_channels
            blocks.append(
                ckconv.nn.CKBlock(
                    block_in_channels,
                    hidden_channels,
                    kernelnet_hidden_channels,
                    kernelnet_activation_function,
                    kernelnet_norm_type,
                    dim_linear,
                    bias,
                    omega_0,
                    dropout,
                    weight_dropout,
                )
            )
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.backbone(x)



class CKCNN(CKCNN_backbone):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
        )

        self.finallyr = torch.nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        return out


# Construct network:
in_channels = 3
out_channels = 10
hidden_channels = 20
num_blocks = 2
kernelnet_hidden_channels = 32
kernelnet_activation_function = 'Sine'
kernelnet_norm_type = ''
dim_linear = 1
bias = True
omega_0 = 30.5
dropout = 0.1
weight_dropout = 0.0

network = CKCNN(in_channels,
                out_channels,
                hidden_channels,
                num_blocks,
                kernelnet_hidden_channels,
                kernelnet_activation_function,
                kernelnet_norm_type,
                dim_linear,
                bias,
                omega_0,
                dropout,
                weight_dropout,
               )
network.to('cuda')


batch_size = 10
in_channels = 3
signal_length = 1000
in_signal = torch.rand([batch_size, in_channels, signal_length])

out = network(in_signal.to('cuda'))
out.shape

figsize = (5, 12)

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize)

counter = 0
for m in network.modules():
    if isinstance(m, ckconv.CKConv):
        axs[counter].set_title('Kernel at layer {}'.format(counter + 1))
        axs[counter].plot(m.conv_kernel.view(-1, signal_length)[0].detach().cpu().numpy())
        counter = counter + 1

plt.tight_layout()
plt.show()