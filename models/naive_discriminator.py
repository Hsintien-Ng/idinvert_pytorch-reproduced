import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.02))

        repeat_num = int(np.log2(image_size)) - 1
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.LeakyReLU(0.02))
            # layers.append(ResidualBlock(curr_dim * 2, curr_dim * 2))
            curr_dim = curr_dim * 2

        # kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src


if __name__ == '__main__':
    D = Discriminator(image_size=256).cuda()
    x = torch.rand(10, 3, 256, 256).cuda()
    print(D(x).shape)