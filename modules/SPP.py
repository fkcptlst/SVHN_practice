from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SPPLayer(nn.Module):
    """
    Spatial Pyramid Pooling Layer
    :outshape: (batch_size, channels * (1 + 2*2 + 4*4 + 8*8 + 16*16+ ... + 2^(num_levels-1) * 2^(num_levels-1)))
    """
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        a = min(h, w)
        pooling_layers = []
        for i in range(self.num_levels):
            out_shape = (2**i, 2**i)
            if self.pool_type == 'max_pool':
                tensor = F.adaptive_max_pool2d(x, out_shape).view(bs, -1)
            else:
                tensor = F.adaptive_avg_pool2d(x, out_shape).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

# class SPPLayer(nn.Module):
#     """
#     Spatial Pyramid Pooling Layer
#     :outshape: (batch_size, channels * (1 + 2*2 + 4*4 + 8*8 + 16*16+ ... + 2^(num_levels-1) * 2^(num_levels-1)))
#     """
#     def __init__(self, num_levels, pool_type='max_pool'):
#         super(SPPLayer, self).__init__()
#         self.num_levels = num_levels
#         # self.out_size = out_size
#         self.pool_type = pool_type
#
#     def forward(self, x):
#         bs, c, h, w = x.size()
#         a = min(h, w)
#         pooling_layers = []
#         for i in range(self.num_levels):
#             kernel_size = a // (2 ** i)
#             if self.pool_type == 'max_pool':
#                 tensor = F.max_pool2d(x, kernel_size=kernel_size,
#                                       stride=kernel_size).view(bs, -1)
#             else:
#                 tensor = F.avg_pool2d(x, kernel_size=kernel_size,
#                                       stride=kernel_size).view(bs, -1)
#             pooling_layers.append(tensor)
#         x = torch.cat(pooling_layers, dim=-1)
#         return x

# class SPPLayer(nn.Module):
#     """
#     Spatial Pyramid Pooling Layer
#     :outshape: (batch_size, channels * (1 + 2*2 + 4*4 + 8*8 + 16*16+ ... + 2^(num_levels-1) * 2^(num_levels-1)))
#     """
#     def __init__(self, num_levels, pool_type='max_pool'):
#         super(SPPLayer, self).__init__()
#         self.num_levels = num_levels
#         self.pool_type = pool_type
#
#     def forward(self, x):
#         bs, c, h, w = x.size()
#         a = min(h, w)
#         pooling_layers = []
#         for i in range(self.num_levels):
#             out_shape = (2**i, 2**i)
#             if self.pool_type == 'max_pool':
#                 tensor = F.adaptive_max_pool2d(x, out_shape).view(bs, -1)
#             else:
#                 tensor = F.adaptive_avg_pool2d(x, out_shape).view(bs, -1)
#             pooling_layers.append(tensor)
#         # print(pooling_layers)
#         x = torch.cat(pooling_layers, dim=-1)
#         return x
        # for i in range(self.num_levels):
        #     kernel_size = a // (2 ** i)
        #     if self.pool_type == 'max_pool':
        #         tensor = F.max_pool2d(x, kernel_size=kernel_size,
        #                               stride=kernel_size).view(bs, -1)
        #     else:
        #         tensor = F.avg_pool2d(x, kernel_size=kernel_size,
        #                               stride=kernel_size).view(bs, -1)
        #     pooling_layers.append(tensor)
        # x = torch.cat(pooling_layers, dim=-1)
        # return x
# class DetectionNetSPP(nn.Module):
#     """
#     Expected input size is 64x64
#     """
#     def __init__(self, spp_level=3):
#         super(DetectionNetSPP, self).__init__()
#         self.spp_level = spp_level
#         self.num_grids = 0
#         for i in range(spp_level):
#             self.num_grids += 2**(i*2)
#         print(self.num_grids)
#
#         self.conv_model = nn.Sequential(OrderedDict([
#           ('conv1', nn.Conv2d(3, 128, 3)),
#           ('relu1', nn.ReLU()),
#           ('pool1', nn.MaxPool2d(2)),
#           ('conv2', nn.Conv2d(128, 128, 3)),
#           ('relu2', nn.ReLU()),
#           ('pool2', nn.MaxPool2d(2)),
#           ('conv3', nn.Conv2d(128, 128, 3)),
#           ('relu3', nn.ReLU()),
#           ('pool3', nn.MaxPool2d(2)),
#           ('conv4', nn.Conv2d(128, 128, 3)),
#           ('relu4', nn.ReLU())
#         ]))
#
#         self.spp_layer = SPPLayer(spp_level)
#
#         self.linear_model = nn.Sequential(OrderedDict([
#           ('fc1', nn.Linear(self.num_grids*128, 1024)),
#           ('fc1_relu', nn.ReLU()),
#           ('fc2', nn.Linear(1024, 2)),
#         ]))
#
#     def forward(self, x):
#         x = self.conv_model(x)
#         x = self.spp_layer(x)
#         x = self.linear_model(x)
#         return x

# if __name__ == '__main__':
#     model = SPPLayer(3)
#     model2 = SPPLayer(3)
#     print(model)
#     x = torch.randn(1, 3, 64, 64)
#     y = model(x)
#     y2 = model2(x)
#     print(y.shape)
#     print(y2.shape)
#     assert y.shape == y2.shape
#     assert torch.allclose(y, y2)
#     x = torch.randn(1, 3, 32, 32)
#     y = model(x)
#     y2 = model2(x)
#     assert y.shape == y2.shape
#     assert torch.allclose(y, y2)
#     print(y.shape)
#     x = torch.randn(1, 3, 128, 128)
#     y = model(x)
#     y2 = model2(x)
#     assert y.shape == y2.shape
#     assert torch.allclose(y, y2)
#     print(y.shape)
#     x = torch.randn(1, 3, 255, 255)
#     y = model(x)
#     y2 = model2(x)
#     print(y.shape)
#     print(y2.shape)
#     assert y.shape == y2.shape
#     assert torch.allclose(y, y2)
#     print("Test passed")