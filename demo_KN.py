# # Append .. to path
# import os, sys
#
#
# ckconv_source = os.path.join(os.getcwd(), '..')
# if ckconv_source not in sys.path:
#     sys.path.append(ckconv_source)
#
# import numpy as np
# import torch
# from torch.nn.utils import weight_norm
#
# import ckconv
# import ckconv.nn.functional as ckconv_f
#
# from matplotlib import pyplot as plt
#
#
#
# # This is a simplified version of the code.
# # A complete version can be found in `ckconv/nn/ckconv.py`.
#
# class KernelNet(torch.nn.Module):
#     def __init__(
#             self,
#             out_channels: int,
#             hidden_channels: int,
#             bias: bool,
#             omega_0: float,
#     ):
#         super().__init__()
#         '''
#         KernelNets are implemented as a 3-layer MLP with sine nonlinearities.
#
#         Args:
#
#          - out_channels: Output channels of KernelNet. For a Conv1D with Nin input channels and Nout output
#                          channels, out_channels is Nin * Nout.
#          - hidden_channels: The hidden dimension of KernelNet.
#          - bias: If the layers will use bias.
#          - omega_0: A prior on the variation of the kernel we want to model. More information on w_0 can be
#                     found in Sec. 4.3 of the paper, and the SIREN paper of Sitzman et. al. (Sec. 3.2., Appx. 1.5)
#                     https://arxiv.org/abs/2006.09661.
#
#         '''
#
#         ActivationFunction = ckconv.nn.Sine
#         Linear = ckconv.nn.Linear1d  # Implements a Linear layer in terms of 1x1 Convolutions.
#         Multiply = ckconv.nn.misc.Multiply  # Multiplies the input by a constant
#
#         # The input of the network is a vector of relative positions. That is, input_dimension = 1.
#         self.kernel_net = torch.nn.Sequential(
#             # 1st layer
#             weight_norm(Linear(1, hidden_channels, bias=bias)),
#             Multiply(omega_0),
#             ActivationFunction(),
#             # 2nd Layer
#             weight_norm(Linear(hidden_channels, hidden_channels, bias=bias)),
#             Multiply(omega_0),
#             ActivationFunction(),
#             # 3rd Layer
#             weight_norm(Linear(hidden_channels, out_channels, bias=bias)),
#         )
#
#         # initialize the kernel function
#         self.initialize(
#             mean=0.0,
#             variance=0.01,
#             bias_value=0.0,
#             omega_0=omega_0,
#         )
#
#     def initialize(self, mean, variance, bias_value, omega_0):
#
#         # Initialization of SIRENs (Please refer to https://arxiv.org/abs/2006.09661 for details).
#         net_layer = 1
#         for (i, m) in enumerate(self.modules()):
#             if (
#                     isinstance(m, torch.nn.Conv1d)
#                     or isinstance(m, torch.nn.Conv2d)
#                     or isinstance(m, torch.nn.Linear)
#             ):
#                 if net_layer == 1:
#                     m.weight.data.uniform_(-1, 1)  # Normally (-1, 1) / in_dim but we only use 1D inputs.
#                     net_layer += 1
#                 else:
#                     m.weight.data.uniform_(
#                         -np.sqrt(6.0 / m.weight.shape[1]) / omega_0,
#                         # the in_size is dim 2 in the weights of Linear and Conv layers
#                         np.sqrt(6.0 / m.weight.shape[1]) / omega_0,
#                     )
#
#                 # Important! Bias is not defined in original SIREN implementation
#                 if m.bias is not None:
#                     m.bias.data.uniform_(-1.0, 1.0)
#
#     def forward(self, x):
#         return self.kernel_net(x)