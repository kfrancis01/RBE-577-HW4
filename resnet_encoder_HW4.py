# # Copyright Niantic 2019. Patent Pending. All rights reserved.
# #
# # This software is licensed under the terms of the Monodepth2 licence
# # which allows for non-commercial use only, the full terms of which are made
# # available in the LICENSE file.

# resnet_encoder_HW4.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResNetMultiImageInput(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, dropout_rate=0.5):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout_rate = dropout_rate

        # Define ResNet layers with dropout
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def forward(self, x):
        features = []

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.maxpool(x)

        # ResNet layers with dropout
        x = self.layer1(x)
        features.append(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.layer2(x)
        features.append(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.layer3(x)
        features.append(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.layer4(x)
        features.append(x)
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        return features


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, dropout_rate=0.5):
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, dropout_rate=dropout_rate)
    
    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded, strict=False)
    return model


class ResnetEncoder2(nn.Module):
    """ResNet encoder with optional dropout"""
    def __init__(self, num_layers, pretrained, num_input_images=1, dropout_rate=0.5):
        super(ResnetEncoder2, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")
        
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, dropout_rate)
        else:
            self.encoder = resnets[num_layers](pretrained)
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        self.dropout_rate = dropout_rate

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225  # Normalize input
        features = self.encoder(x)
        return features


# from __future__ import absolute_import, division, print_function

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.utils.model_zoo as model_zoo

# class ResNetMultiImageInput(models.ResNet):
#     def __init__(self, block, layers, num_classes=1000, num_input_images=1, dropout_rate=0.5):
#         super(ResNetMultiImageInput, self).__init__(block, layers)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.convs = nn.ModuleDict()
        
#         # Define ResNet layers and dropout rate
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.dropout_rate = dropout_rate

#     def forward(self, x):
#         self.outputs = {}

#         # Process input features through encoder
#         features = self.encoder(x)  # Pass features through ResNet

#         # Decoder processing (using features from encoder)
#         x = features[-1]  # Use the last feature map from the encoder
        
#         for i in range(4, -1, -1):
            
#             x = self.convs[f"upconv_{i}_0"](x)
            
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)
#             x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
#             x = self.maxpool(x)
        
#         return self.outputs
        
#         # Apply dropout after each layer if needed
#         features = []
#         features.append(self.layer1(x))
#         features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#         features.append(self.layer2(features[-1]))
#         features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#         features.append(self.layer3(features[-1]))
#         features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#         features.append(self.layer4(features[-1]))
#         features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#         return features

# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, dropout_rate=0.5):
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, dropout_rate=dropout_rate)
    
#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded, strict=False)
#     return model

# class ResnetEncoder2(nn.Module):
#     def __init__(self, num_layers, pretrained, num_input_images=1, dropout_rate=0.5):
#         super(ResnetEncoder2, self).__init__()
#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}

#         self.convs = nn.ModuleDict({
#         "upconv_4_0": nn.Conv2d(512, 256, kernel_size=3, padding=1),
#         "upconv_4_1": nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         })
        
#         if num_layers not in resnets:
#             raise ValueError(f"{num_layers} is not a valid number of resnet layers")
        
#         if num_input_images > 1:
#             self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, dropout_rate)
#         else:
#             self.encoder = resnets[num_layers](pretrained)
        
#         if num_layers > 34:
#             self.num_ch_enc[1:] *= 4
#         self.dropout_rate = dropout_rate

#     # def forward(self, input_image):
#     #     x = (input_image - 0.45) / 0.225
#     #     x = self.encoder.conv1(x)
#     #     x = self.encoder.bn1(x)
#     #     x = self.encoder.relu(x)
#     #     x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
#     #     x = self.encoder.maxpool(x)
        
#     #     features = []
#     #     features.append(self.encoder.layer1(x))
#     #     features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#     #     features.append(self.encoder.layer2(features[-1]))
#     #     features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#     #     features.append(self.encoder.layer3(features[-1]))
#     #     features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#     #     features.append(self.encoder.layer4(features[-1]))
#     #     features[-1] = nn.functional.dropout(features[-1], p=self.dropout_rate, training=self.training)
        
#     #     return features
#     def forward(self, input_features):
#         self.outputs = {}

#         # Decoder processing
#         x = input_features[-1]
#         for i in range(4, -1, -1):
#             # x = self.convs[("upconv", i, 0)](x)
#             x = self.convs[f"upconv_{i}_0"](x)
            
#             # Adjust interpolate based on mode
#             if self.upsample_mode == "nearest":
#                 x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode)
#             else:
#                 x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)
            
#             if self.use_skips and i > 0:
#                 x = torch.cat([x, input_features[i - 1]], 1)
            
#             # x = self.convs[("upconv", i, 1)](x)
#             x = self.convs[f"upconv_{i}_1"](x)
            
#             if i in self.scales:
#                 self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

#         return self.outputs


# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.utils.model_zoo as model_zoo

# class ResNetMultiImageInput(models.ResNet):
#     def __init__(self, block, layers, num_classes=1000, num_input_images=1, dropout_rate=0.5):
#         super(ResNetMultiImageInput, self).__init__(block, layers)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.maxpool(x)
        
#         # Extract features from each ResNet layer
#         features = []
#         features.append(self.layer1(x))
#         features.append(self.layer2(features[-1]))
#         features.append(self.layer3(features[-1]))
#         features.append(self.layer4(features[-1]))
        
#         return features

# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, dropout_rate=0.5):
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, dropout_rate=dropout_rate)
    
#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded, strict=False)
#     return model

# class ResnetEncoder2(nn.Module):
#     def __init__(self, num_layers, pretrained, num_input_images=1, dropout_rate=0.5):
#         super(ResnetEncoder2, self).__init__()
#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}

#         if num_layers not in resnets:
#             raise ValueError(f"{num_layers} is not a valid number of resnet layers")
        
#         if num_input_images > 1:
#             self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, dropout_rate)
#         else:
#             self.encoder = resnets[num_layers](pretrained)
        
#         if num_layers > 34:
#             self.num_ch_enc[1:] *= 4

#     def forward(self, input_image):
#         x = (input_image - 0.45) / 0.225
#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         x = self.encoder.relu(x)
#         x = self.encoder.dropout(x)
#         x = self.encoder.maxpool(x)
        
#         features = []
#         features.append(self.encoder.layer1(x))
#         features.append(self.encoder.layer2(features[-1]))
#         features.append(self.encoder.layer3(features[-1]))
#         features.append(self.encoder.layer4(features[-1]))
        
#         return features


# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.utils.model_zoo as model_zoo
# import torch.nn.functional as F

# class ResNetMultiImageInput(models.ResNet):
#     def __init__(self, block, layers, num_classes=1000, num_input_images=1, dropout_rate=0.5):
#         super(ResNetMultiImageInput, self).__init__(block, layers)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout_rate)  # Add dropout here
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # Define resnet layers
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Apply dropout only in forward pass
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)  # Apply dropout after activation
#         x = self.maxpool(x)
        
#         # Pass through ResNet layers
#         x = self.layer1(x)
#         x = self.dropout(x)
#         x = self.layer2(x)
#         x = self.dropout(x)
#         x = self.layer3(x)
#         x = self.dropout(x)
#         x = self.layer4(x)
#         x = self.dropout(x)
        
#         return x


# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, dropout_rate=0.5):
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, dropout_rate=dropout_rate)
    
#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded, strict=False)
#     return model

# class ResnetEncoder2(nn.Module):
#     def __init__(self, num_layers, pretrained, num_input_images=1, dropout_rate=0.5):
#         super(ResnetEncoder2, self).__init__()
#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}

#         if num_layers not in resnets:
#             raise ValueError(f"{num_layers} is not a valid number of resnet layers")
        
#         if num_input_images > 1:
#             self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, dropout_rate)
#         else:
#             self.encoder = resnets[num_layers](pretrained)
        
#         if num_layers > 34:
#             self.num_ch_enc[1:] *= 4

#         # Define separate dropout layer for the forward pass
#         self.dropout = nn.Dropout(dropout_rate)

#     # def forward(self, input_image):
#     #     x = (input_image - 0.45) / 0.225  # Normalize input
#     #     x = self.encoder.conv1(x)
#     #     x = self.encoder.bn1(x)
#     #     x = self.encoder.relu(x)
#     #     x = self.dropout(x)  # Apply dropout here
#     #     x = self.encoder.maxpool(x)
        
#     #     self.features = []
#     #     x = self.encoder.layer1(x)
#     #     self.features.append(self.dropout(x))  # Apply dropout and save feature
#     #     x = self.encoder.layer2(x)
#     #     self.features.append(self.dropout(x))
#     #     x = self.encoder.layer3(x)
#     #     self.features.append(self.dropout(x))
#     #     x = self.encoder.layer4(x)
#     #     self.features.append(self.dropout(x))
        
#     #     return self.features
    
#     def forward(self, input_features):
#         self.outputs = {}
#         x = input_features[-1]  # Start from the last encoder feature map

#         # Start decoding and upsampling
#         for i in range(4, -1, -1):
#             # Upsample x to match the size of the feature map it will be concatenated with
#             x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False)

#             # Concatenate with skip connection if using skips and if not the first layer
#             if self.use_skips and i > 0:
#                 # Ensure sizes match by checking shape, then concatenate
#                 if x.size(2) != input_features[i - 1].size(2) or x.size(3) != input_features[i - 1].size(3):
#                     x = nn.functional.interpolate(x, size=(input_features[i - 1].size(2), input_features[i - 1].size(3)),
#                                                 mode=self.upsample_mode, align_corners=False)
#                 x = torch.cat([x, input_features[i - 1]], dim=1)
            
#             # Apply convolution and store results
#             x = self.convs[("upconv", i, 1)](x)
            
#             # Generate disparity map for scales that need output
#             if i in self.scales:
#                 self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

#         return self.outputs


# from __future__ import absolute_import, division, print_function

# import numpy as np

# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.utils.model_zoo as model_zoo

# class ResNetMultiImageInput(models.ResNet):
#     def __init__(self, block, layers, num_classes=1000, num_input_images=1, dropout_rate=0.5):
#         super(ResNetMultiImageInput, self).__init__(block, layers)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(dropout_rate)  # Add dropout layer here
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # Define resnet layers with dropout
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # Apply initialization for conv and batch norm layers
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     # def forward(self, x):
#     def forward(self, input_image):
#         # x = self.conv1(x)
#         # x = self.bn1(x)
#         # x = self.relu(x)
#         # x = self.dropout(x)  # Apply dropout after activation
#         # x = self.maxpool(x)
#         # x = self.layer1(x)
#         # x = self.dropout(x)  # Optionally add dropout after each layer
#         # x = self.layer2(x)
#         # x = self.dropout(x)
#         # x = self.layer3(x)
#         # x = self.dropout(x)
#         # x = self.layer4(x)
#         # x = self.dropout(x)
#         # return x
        
#         self.features = []
#         x = (input_image - 0.45) / 0.225

#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         x = F.relu(x)
#         x = self.encoder.maxpool(x)
#         x = self.dropout(x)  # Apply dropout after the first layer

#         # Extract features from each ResNet layer with dropout after each if desired
#         x = self.encoder.layer1(x)
#         x = self.dropout(x)
#         self.features.append(x)

#         x = self.encoder.layer2(x)
#         x = self.dropout(x)
#         self.features.append(x)

#         x = self.encoder.layer3(x)
#         x = self.dropout(x)
#         self.features.append(x)

#         x = self.encoder.layer4(x)
#         x = self.dropout(x)
#         self.features.append(x)

#         return self.features

# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, dropout_rate=0.5):
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, dropout_rate=dropout_rate)
    
#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded, strict=False)
#     return model

# class ResnetEncoder2(nn.Module):
#     def __init__(self, num_layers, pretrained, num_input_images=1, dropout_rate=0.5):
#         super(ResnetEncoder2, self).__init__()
#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}

#         if num_layers not in resnets:
#             raise ValueError(f"{num_layers} is not a valid number of resnet layers")
        
#         if num_input_images > 1:
#             self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, dropout_rate)
#         else:
#             self.encoder = resnets[num_layers](pretrained)
        
#         if num_layers > 34:
#             self.num_ch_enc[1:] *= 4

#     def forward(self, input_image):
#         x = (input_image - 0.45) / 0.225
#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         x = self.encoder.relu(x)
#         x = self.encoder.dropout(x)  # Apply dropout if defined in the custom ResNet
#         x = self.encoder.maxpool(x)
        
#         self.features = []
#         self.features.append(self.encoder.layer1(x))
#         self.features.append(self.encoder.layer2(self.features[-1]))
#         self.features.append(self.encoder.layer3(self.features[-1]))
#         self.features.append(self.encoder.layer4(self.features[-1]))
        
#         return self.features

# class ResNetMultiImageInput(models.ResNet):
#     def __init__(self, block, layers, num_classes=1000, num_input_images=1):
#         super(ResNetMultiImageInput, self).__init__(block, layers)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(
#             num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
#     assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat(
#             [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded)
#     return model


# class ResnetEncoder(nn.Module):
#     def __init__(self, num_layers, pretrained, num_input_images=1):
#         super(ResnetEncoder, self).__init__()

#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])

#         resnets = {18: models.resnet18,
#                    34: models.resnet34,
#                    50: models.resnet50,
#                    101: models.resnet101,
#                    152: models.resnet152}

#         if num_layers not in resnets:
#             raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

#         if num_input_images > 1:
#             self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
#         else:
#             self.encoder = resnets[num_layers](pretrained)

#         if num_layers > 34:
#             self.num_ch_enc[1:] *= 4

#     def forward(self, input_image):
#         self.features = []
#         x = (input_image - 0.45) / 0.225
#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         self.features.append(self.encoder.relu(x))
#         self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
#         self.features.append(self.encoder.layer2(self.features[-1]))
#         self.features.append(self.encoder.layer3(self.features[-1]))
#         self.features.append(self.encoder.layer4(self.features[-1]))

#         return self.features
