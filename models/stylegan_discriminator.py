# python 3.7
"""Contains the generator class of StyleGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch
import torch.nn as nn

from .base_discriminator import BaseDiscriminator
from .stylegan_discriminator_network import FixedDiscriminator

__all__ = ['StyleGANDiscriminator']


class StyleGANDiscriminator(BaseDiscriminator):
    """Defines the generator class of StyleGAN.

    Different from conventional GAN, StyleGAN introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self, model_name, logger=None, gpu_ids=None):
        self.gan_type = 'stylegan'
        super().__init__(model_name, logger, gpu_ids)
        # Data Parallel
        self.net.to(self.run_device)
        if self.gpu_ids is not None:
            assert len(self.gpu_ids) > 1
            self.net = nn.DataParallel(self.net, self.gpu_ids)

    def build(self):
        self.fmaps_base = getattr(self, 'fmaps_base', 8192)
        self.fmaps_max = getattr(self, 'fmaps_max', 512)
        self.net = FixedDiscriminator(
            resolution=self.resolution,
            fmaps_base=self.fmaps_base,
            fmaps_max=self.fmaps_max)

    def _encode(self, images):
        if not isinstance(images, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
        if (images.ndim != 4 or images.shape[0] <= 0 or
                images.shape[0] > self.batch_size or images.shape[1:] != (
                        self.image_channels, self.resolution, self.resolution)):
            raise ValueError(f'Input images should be with shape [batch_size, '
                             f'channel, height, width], where '
                             f'`batch_size` no larger than {self.batch_size}, '
                             f'`channel` equals to {self.image_channels}, '
                             f'`height` and `width` equal to {self.resolution}!\n'
                             f'But {images.shape} is received!')

        xs = self.to_tensor(images.astype(np.float32))
        codes = self.net(xs)
        # assert codes.shape == (images.shape[0], np.prod(self.encode_dim))
        # codes = codes.view(codes.shape[0], *self.encode_dim)
        results = {
            'image': images,
            'code': self.get_value(codes),
        }

        if self.use_cuda:
            torch.cuda.empty_cache()

        return results

    def encode(self, images, **kwargs):
        return self.batch_run(images, self._encode)