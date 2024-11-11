# # Copyright Niantic 2019. Patent Pending. All rights reserved.
# #
# # This software is licensed under the terms of the Monodepth2 licence
# # which allows for non-commercial use only, the full terms of which are made
# # available in the LICENSE file.

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import ConvBlock, Conv3x3  # Ensure these are correctly imported from your project

# class DepthDecoder2(nn.Module):
#     def __init__(self, num_ch_enc, scales, dropout_rate=0.5, upsample_mode="nearest"):
#         super(DepthDecoder2, self).__init__()
#         self.scales = scales
#         self.dropout_rate = dropout_rate
#         self.upsample_mode = upsample_mode
#         self.sigmoid = nn.Sigmoid()
#         self.convs = nn.ModuleDict()

#         # Define upconv layers
#         for i in range(4):
#             in_channels = num_ch_enc[4 - i] if i == 0 else num_ch_enc[4 - i] // 2
#             out_channels = num_ch_enc[4 - i] // 2
#             self.convs[f"upconv{i}"] = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout_rate)
#             )

#         # Define dispconv layers for each scale
#         for s in self.scales:
#             self.convs[f"dispconv{s}"] = nn.Conv2d(num_ch_enc[-1] // 2, 1, kernel_size=3, stride=1, padding=1)
#     def forward(self, features):
#         # Check initial feature map shape
#         x = features[-1]
#         print(f"Initial feature map shape: {x.shape}")  # Expecting [batch_size, channels, height, width]
#         print("Shape of initial feature map (x):", x.shape)
#         # if len(x.shape) != 4:
#         #     x = x.view(1, *x.shape)
#         #     raise ValueError(f"Expected a 4D tensor for `x`, but got shape: {x.shape}")
        
#         if len(x.shape) == 1:
#             batch_size = 12 
#             channels, height, width = 512, 135, 240  # use actual expected dimensions
#             x = x.view(batch_size, channels, height, width)

#         # Process through each upconv layer
#         for i in range(4):
#             x = self.convs[f"upconv{i}"](x)
#             print(f"Shape after upconv{i}: {x.shape}")  # Debugging output
#             # Adjust interpolation based on mode
#             if self.upsample_mode in ["bilinear", "bicubic"]:
#                 x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
#             else:
#                 x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode)
#             print(f"Shape after upsampling in upconv{i}: {x.shape}")  # Debugging output

#         outputs = {}
#         for s in self.scales:
#             disp = self.sigmoid(self.convs[f"dispconv{s}"](x))
#             outputs[("disp", s)] = disp
#             print(f"Shape of disp at scale {s}: {disp.shape}")  # Debugging output

#         return outputs

    # def forward(self, features):
    #     # Start from the last feature map in encoder outputs
    #     x = features[-1]
    #     print(f"Initial feature map shape: {x.shape}")  # Debugging output

    #     # Process through each upconv layer
    #     for i in range(4):
    #         x = self.convs[f"upconv{i}"](x)
    #         print(f"Shape after upconv{i}: {x.shape}")  # Debugging output
            
    #         # Adjust interpolation based on mode
    #         if self.upsample_mode in ["bilinear", "bicubic"]:
    #             x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
    #         else:
    #             x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode)
    #         print(f"Shape after upsampling in upconv{i}: {x.shape}")  # Debugging output

    #     # Apply dispconv layers for each scale
    #     outputs = {}
    #     for s in self.scales:
    #         disp = self.sigmoid(self.convs[f"dispconv{s}"](x))
    #         outputs[("disp", s)] = disp
    #         print(f"Shape of disp at scale {s}: {disp.shape}")  # Debugging output

    #     return outputs



# import numpy as np
# import torch
# import torch.nn as nn
# from collections import OrderedDict
# from layers import ConvBlock, Conv3x3  # Ensure these are correctly imported from your project

# class DepthDecoder2(nn.Module):
#     def __init__(self, num_ch_enc, scales, dropout_rate=0.5, upsample_mode="nearest"):
#         super(DepthDecoder2, self).__init__()
#         self.scales = scales
#         self.dropout_rate = dropout_rate
#         self.upsample_mode = upsample_mode
#         self.sigmoid = nn.Sigmoid()
#         self.convs = nn.ModuleDict()

#         for i in range(4):
#             self.convs[f"upconv{i}"] = nn.Sequential(
                
#                 nn.Conv2d(num_ch_enc[i], num_ch_enc[i] // 2, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm2d(num_ch_enc[i] // 2),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout_rate)
#             )
            

#         for s in self.scales:
#             self.convs[f"dispconv{s}"] = nn.Conv2d(num_ch_enc[-1] // 2, 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, features):
#         x = features[-1]

#         for i in range(4):
#             x = self.convs[f"upconv{i}"](x)
#             print(x.shape)  # Add this line before every convolutional layer
#             # Adjust interpolation based on mode
#             if self.upsample_mode in ["bilinear", "bicubic"]:
#                 x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
#             else:
#                 x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode)

#         outputs = {}
#         for s in self.scales:
#             outputs[("disp", s)] = self.sigmoid(self.convs[f"dispconv{s}"](x))

#         return outputs


# class DepthDecoder2(nn.Module):
#     def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, dropout_rate=0.5):
#         super(DepthDecoder2, self).__init__()

#         self.num_output_channels = num_output_channels
#         self.use_skips = use_skips
#         self.upsample_mode = 'nearest'  # Set upsample mode here
#         self.scales = scales
#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = np.array([16, 32, 64, 128, 256])

#         # Initialize decoder layers with dropout
#         self.convs = OrderedDict()
#         self.dropout = nn.Dropout(dropout_rate)

#         for i in range(4):
#             self.convs[f"upconv{i}"] = nn.Sequential(
#                 nn.Conv2d(num_ch_enc[i], num_ch_enc[i] // 2, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm2d(num_ch_enc[i] // 2),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout_rate)
#             )
        
#         for i in range(4, -1, -1):
#             num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

#             num_ch_in = self.num_ch_dec[i]
#             if self.use_skips and i > 0:
#                 num_ch_in += self.num_ch_enc[i - 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

#         for s in self.scales:
#             self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_features):
#         self.outputs = {}
#         x = input_features[-1]

#         for i in range(4, -1, -1):
#             x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)

#             if self.use_skips and i > 0:
#                 x = torch.cat([x, input_features[i - 1]], dim=1)

#             x = self.convs[("upconv", i, 1)](x)
#             if i in self.scales:
#                 self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

#         return self.outputs


# class DepthDecoder2(nn.Module):
    # def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, dropout_rate=0.5):
    #     super(DepthDecoder2, self).__init__()

    #     self.num_output_channels = num_output_channels
    #     self.use_skips = use_skips
    #     self.upsample_mode = 'nearest'
    #     self.scales = scales

    #     self.num_ch_enc = num_ch_enc
    #     self.num_ch_dec = np.array([16, 32, 64, 128, 256])

    #     # Decoder layers with dropout
    #     self.convs = OrderedDict()
    #     for i in range(4, -1, -1):
    #         # First upconv layer for each level
    #         num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
    #         num_ch_out = self.num_ch_dec[i]
    #         self.convs[("upconv", i, 0)] = nn.Sequential(
    #             ConvBlock(num_ch_in, num_ch_out),
    #             nn.Dropout(dropout_rate)  # Apply dropout after the first ConvBlock
    #         )

    #         # Second upconv layer with optional skip connections
    #         num_ch_in = self.num_ch_dec[i]
    #         if self.use_skips and i > 0:
    #             num_ch_in += self.num_ch_enc[i - 1]
    #         num_ch_out = self.num_ch_dec[i]
    #         self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

    #     # Disparity prediction layers
    #     for s in self.scales:
    #         self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

    #     # Create the decoder as an nn.ModuleList
    #     self.decoder = nn.ModuleList(list(self.convs.values()))
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, input_features):
    #     self.outputs = {}

    #     # Start decoding
    #     x = input_features[-1]
    #     for i in range(4, -1, -1):
    #         x = self.convs[("upconv", i, 0)](x)  # Apply first upconv with dropout
    #         x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode)

    #         # Skip connections
    #         if self.use_skips and i > 0:
    #             x = torch.cat([x, input_features[i - 1]], 1)

    #         x = self.convs[("upconv", i, 1)](x)  # Apply second upconv without dropout

    #         # Disparity output at specified scales
    #         if i in self.scales:
    #             self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

    #     return self.outputs

