import torch.nn as nn
import torch.nn.functional as F
import torch
from .SPP import SPPLayer

class STNLayer(nn.Module):
    """
    Spatial Transformer Network
    """
    def __init__(self, spp_num_levels=3, adaptive_pooling_shape=(54,54), spp_pool_type='max_pool'):
        super(STNLayer, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d(adaptive_pooling_shape), # TODO need check
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.spp_layer = SPPLayer(num_levels=spp_num_levels, pool_type=spp_pool_type)

        linear_input_size = sum((4 ** i) * 3 for i in range(spp_num_levels))
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            # nn.Linear(10 * 3 * 3, 32),
            nn.Linear(linear_input_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        # xs = xs.view(-1, 10 * 3 * 3)  # reshape?

        xs = self.spp_layer(xs)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
