import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

import numpy as np
from math import sqrt


class ResolutionScalingLayer(nn.Module):
  """Implements the resolution scaling layer.

  Basically, this layer can be used to upsample feature maps from spatial domain
  with nearest neighbor interpolation.
  """

  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class BlurLayer(nn.Module):
    """Implements the blur layer."""
    def __init__(self,
               channels,
               kernel=(1, 2, 1),
               normalize=True,
               flip=False):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, -1)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class DownConvBlock(nn.Module):
    """Implements the convolutional block with downsampling.

    Basically, this block is used as the second convolutional block for each
    resolution, which will execute downsampling.
    """

    def __init__(self,
               resolution,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               fused_scale=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=1.0,
               w_space_dim=512,
               randomize_noise=False):
        """Initializes the class with block settings.

        Args:
            resolution: Spatial resolution of current layer.
            in_channels: Number of channels of the input tensor fed into this block.
            out_channels: Number of channels (kernels) of the output tensor.
            kernel_size: Size of the convolutional kernel.
            stride: Stride parameter for convolution operation.
            padding: Padding parameter for convolution operation.
            dilation: Dilation rate for convolution operation.
            add_bias: Whether to add bias onto the convolutional result.
            fused_scale: Whether to fuse `downsample` and `conv2d` together, resulting in `conv2d`.
            wscale_gain: The gain factor for `wscale` layer.
            wscale_lr_multiplier: The learning rate multiplier factor for `wscale` layer.
            w_space_dim: The dimension of disentangled latent space, w. This is used for style modulation.
            randomize_noise: Whether to add random noise.
        """
        super().__init__()

        self.fused_scale = fused_scale

        if self.fused_scale:
            self.weight = nn.Parameter(
                torch.randn(kernel_size, kernel_size, in_channels, out_channels))

        else:
            self.downsample = ResolutionScalingLayer(scale_factor=0.5)
            self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=1,
                                bias=add_bias)

        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.blur = BlurLayer(channels=in_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.blur(x)
        if self.fused_scale:
            kernel = self.weight * self.scale
            kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
            kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                        kernel[1:, :-1] + kernel[:-1, :-1])
            kernel = kernel.permute(2, 3, 0, 1)
            x = F.conv2d(x, kernel, stride=2, padding=1)
        else:
            x = self.downsample(x)
            x = self.conv(x) * self.scale
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """Implements the normal convolutional block.

    Basically, this block is used as the second convolutional block for each
    resolution.
    """

    def __init__(self,
               resolution,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=1.0,
               w_space_dim=512,
               randomize_noise=False):
        """Initializes the class with block settings.

        Args:
          resolution: Spatial resolution of current layer.
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels (kernels) of the output tensor.
          kernel_size: Size of the convolutional kernel.
          stride: Stride parameter for convolution operation.
          padding: Padding parameter for convolution operation.
          dilation: Dilation rate for convolution operation.
          add_bias: Whether to add bias onto the convolutional result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.
          w_space_dim: The dimension of disentangled latent space, w. This is used
            for style modulation.
          randomize_noise: Whether to add random noise.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=1,
                              bias=add_bias)
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = wscale_gain / np.sqrt(fan_in) * wscale_lr_multiplier
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x) * self.scale
        x = self.act(x)
        return x


class WScaleLayer(nn.Module):
    """Implements the layer to scale weight variable and add bias.

    NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
    layer), and only scaled with a constant number, which is not trainable in
    this layer. However, the bias variable is trainable in this layer.
    """

    def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               gain=np.sqrt(2.0),
               lr_multiplier=1.0):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain / np.sqrt(fan_in) * lr_multiplier
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.lr_multiplier = lr_multiplier

    def forward(self, x):
        if x.ndim == 4:
            return x * self.scale + self.bias.view(1, -1, 1, 1) * self.lr_multiplier
        if x.ndim == 2:
            return x * self.scale + self.bias.view(1, -1) * self.lr_multiplier
        raise ValueError(f'The input tensor should be with shape [batch_size, '
                         f'channel, height, width], or [batch_size, channel]!\n'
                         f'But {x.shape} is received!')


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer, weight-scale layer,
    and activation layer in sequence.
    """

    def __init__(self,
               in_channels,
               out_channels,
               add_bias=False,
               wscale_gain=np.sqrt(2.0),
               wscale_lr_multiplier=0.01,
               activation_type='lrelu'):
        """Initializes the class with block settings.

        Args:
          in_channels: Number of channels of the input tensor fed into this block.
          out_channels: Number of channels of the output tensor.
          add_bias: Whether to add bias onto the fully-connected result.
          wscale_gain: The gain factor for `wscale` layer.
          wscale_lr_multiplier: The learning rate multiplier factor for `wscale`
            layer.
          activation_type: Type of activation. Support `linear` and `lrelu`.

        Raises:
          NotImplementedError: If the input `activation_type` is not supported.
        """
        super().__init__()
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels,
                            bias=add_bias)
        self.wscale = WScaleLayer(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  gain=wscale_gain,
                                  lr_multiplier=wscale_lr_multiplier)
        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

    def forward(self, x):
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.wscale(x)
        x = self.activate(x)
        return x


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = min(group_size, x.shape[0])
    s = x.shape
    y = x.view([group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])
    y -= torch.mean(y, dim=0, keepdim=True)
    y = torch.mean(y * y, dim=0)
    y = torch.sqrt(y + 1e-8)
    y = torch.mean(y, dim=2, keepdim=True)
    y = torch.mean(y, dim=3, keepdim=True)
    y = torch.mean(y, dim=4, keepdim=True)
    y = torch.mean(y, dim=2)
    y = y.repeat([group_size, 1, s[2], s[3]])
    return torch.cat([x, y], dim=1)


class block(nn.Module):
    def __init__(self, resolution, in_channel, out_channel, mbstd_group_size=4, mbstd_num_features=1, label_size=0):
        super(block, self).__init__()
        self.resolution_log2 = int(np.log2(resolution))
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        if self.resolution_log2 >= 3:
            self.Conv0 = ConvBlock(resolution, in_channel, in_channel, add_bias=True)
            self.Conv1_down = DownConvBlock(resolution, in_channel, out_channel, add_bias=True)
        else:
            if self.mbstd_group_size > 1:
                conv0_inchannel = in_channel + 1
            else:
                conv0_inchannel = in_channel
            self.Conv0 = ConvBlock(resolution, conv0_inchannel, in_channel, add_bias=True)
            self.Dense0 = DenseBlock(resolution * resolution * in_channel, out_channel, add_bias=True)
            self.Dense1 = DenseBlock(out_channel, max(label_size, 1), wscale_gain=1., add_bias=True, activation_type='linear')

    def forward(self, x):
        if self.resolution_log2 >= 3:
            x = self.Conv0(x)
            x = self.Conv1_down(x)
        else:
            if self.mbstd_group_size > 1:
                x = minibatch_stddev_layer(x, self.mbstd_group_size, self.mbstd_num_features)
            x = self.Conv0(x)
            x = self.Dense0(x)
            x = self.Dense1(x)
        return x


class FirstConvBlock(nn.Module):
    """Implements the last convolutional block.

    Basically, this block converts RGB image to feature map.
    """

    def __init__(self, out_channels, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              bias=False)
        self.scale = 1 / np.sqrt(in_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv(x) * self.scale
        x = x + self.bias.view(1, -1, 1, 1)
        return x


class FixedDiscriminator(nn.Module):
    def __init__(self, resolution, fmaps_base=8192, fmaps_max=512):
        super(FixedDiscriminator, self).__init__()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(self.resolution))
        self.add_module(f'fromrgb', FirstConvBlock(self.get_nf(self.resolution_log2-1)))
        for block_idx in range(self.resolution_log2, 1, -1):
            self.add_module(f'block{block_idx}', block(self.resolution, self.get_nf(block_idx-1), self.get_nf(block_idx-2)))
            self.resolution = self.resolution // 2

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(int(self.fmaps_base / (2.0 ** res)), self.fmaps_max)

    def forward(self, x):
        x = self.__getattr__(f'fromrgb')(x)
        for block_idx in range(self.resolution_log2, 1, -1):
            x = self.__getattr__(f'block{block_idx}')(x)
        return x

if __name__ == '__main__':
    x = torch.rand(16, 3, 256, 256).cuda()
    model = FixedDiscriminator(256).cuda()
    out = model(x)
    print(out.shape)
