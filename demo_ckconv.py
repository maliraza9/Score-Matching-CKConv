# # Append .. to path
# import os, sys
#
# ckconv_source = os.path.join(os.getcwd(), '..')
# if ckconv_source not in sys.path:
#     sys.path.append(ckconv_source)
# device = 'cuda:0'
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
# from demo_KN import KernelNet
#
#
# class CKConv(torch.nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             hidden_channels: int,
#             bias: bool,
#             omega_0: float,
#     ):
#         super().__init__()
#
#         # Conv. Kernel is now parameterized by a KernelNet
#         self.Kernel = KernelNet(
#             out_channels * in_channels,
#             hidden_channels,
#             bias,
#             omega_0,
#         )
#
#         if bias:
#             self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
#             self.bias.data.fill_(value=0.0)
#         else:
#             self.bias = None
#
#         self.rel_positions = None
#         self.register_buffer("conv_kernel", torch.zeros(in_channels), persistent=False)
#
#     def forward(self, x):
#         '''
#         The forward pass consists of 3 main steps:
#
#         1. The first time the CKConv sees an input, it will check its dimensionality. Based on that,
#            we create a vector of (normalized) relative positions of equal length.
#
#         2. We now pass the vector of relative positions trhough self.Kernel. This will create the
#            convolutional kernel of the layer on the fly.
#
#         3. We then compute the convolution. As the convolutional kernel is often large, we use a
#            Fourier Convolution to reduce computation complexity.
#
#         '''
#
#         # Construct kernel (Step 1)
#         x_shape = x.shape
#         rel_pos = self.handle_rel_positions(x)
#
#         # Pass relative prositions through self.Kernel (Step 2)
#         conv_kernel = self.Kernel(rel_pos).view(-1, x_shape[1], *x_shape[2:])
#         self.conv_kernel = conv_kernel
#
#         # Compute the convolution
#         return ckconv_f.causal_fftconv(x, conv_kernel, self.bias, double_precision=False)
#
#     def handle_rel_positions(self, x):
#         if (self.rel_positions is None):
#             self.rel_positions = (torch.linspace(-1.0, 1.0, x.shape[-1]).unsqueeze(0).unsqueeze(0))
#             # -> Of form (batch_size=1, in_channels=1, x_dimension=x.shape[-1])
#         return self.rel_positions