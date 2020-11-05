import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

import numpy as np
from math import sqrt


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
                         weight[:, :, 1:, 1:]
                         + weight[:, :, :-1, 1:]
                         + weight[:, :, 1:, :-1]
                         + weight[:, :, :-1, :-1]
                 ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            padding,
            kernel_size2=None,
            padding2=None,
            downsample=False,
            fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = min(group_size, x.shape[0])
    s = x.shape
    y = x.view([group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3]])
    y -= torch.mean(y, dim=0, keepdim=True)
    y = torch.mean(y * y, dim=0)
    y = torch.sqrt(y + 1e-8)
    y = torch.mean(y, dim=2, keepdim=True)
    y = torch.mean(y, dim=3, keepdim=True)
    y = torch.mean(y, dim=4, keepdim=True)
    y = torch.mean(y, dim=2)
    y = y.repeat([group_size, 1, s[2], s[3]])
    return torch.cat([x, y], dim=1)


class ConvBlockOut(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, label_size=0):
        super().__init__()
        self.conv = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel, padding=padding),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Linear(512 * 4 * 4, 512, bias=False)
        self.fc2 = nn.Linear(512, max(label_size, 1), bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = x.view([batch_size, -1])
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, structure='fixed', fused=True, from_rgb_activate=False, fmap_base=8192, fmap_decay=1.0,
                 fmap_max=512, mbstd_group_size=4, mbstd_num_features=1, label_size=0):
        super().__init__()

        self.structure = structure
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.progression = nn.ModuleList(
            [
                # ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(nf(8 - 1), nf(7 - 1), 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(nf(7 - 1), nf(6 - 1), 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(nf(6 - 1), nf(5 - 1), 3, 1, downsample=True),  # 32
                ConvBlock(nf(5 - 1), nf(4 - 1), 3, 1, downsample=True),  # 16
                ConvBlock(nf(4 - 1), nf(3 - 1), 3, 1, downsample=True),  # 8
                ConvBlock(nf(3 - 1), nf(2 - 1), 3, 1, downsample=True),  # 4
            ]
        )

        self.block_out = ConvBlockOut(nf(3 - 1) + 1, nf(3 - 1), 3, 1, label_size)

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))
            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        self.n_layer = len(self.progression)

    def forward(self, x):
        if self.structure == 'fixed':
            resolution = x.shape[2]
            assert resolution == 256
            resolution_log2 = int(np.log2(resolution))
            x = self.from_rgb[0](x)
            for res in range(resolution_log2, 2, -1):
                # if res >= 3:   # 8x8 and up
                idx = resolution_log2 - res
                x = self.progression[idx](x)
            # else:
            if self.mbstd_group_size > 1:
                x = minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
            x = self.block_out(x)
            return x

        # elif self.structure


if __name__ == '__main__':
    D = Discriminator()
    a = torch.rand(32, 3, 256, 256)
    out = D(a)
    print(out.shape)

    # a = torch.rand(32, 64, 256, 256)
    # C = ConvBlock(64, 128, 3, 1, downsample=True)
    # out = C(a)
    # print(out.shape)