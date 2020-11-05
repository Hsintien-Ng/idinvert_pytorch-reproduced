from datasets.celebahq import CelebaHQ
from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel
# from models.stylegan_discriminator import Discriminator
# from models.naive_discriminator import Discriminator
from models.stylegan_discriminator import StyleGANDiscriminator
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
import torch.autograd as autograd
from torch.utils.data import DataLoader


def div_loss_(D, real_x, fake_x, p=2, cuda=False):
    # if cuda:
    #     alpha = torch.rand((real_x.shape[0], 1, 1, 1)).cuda()
    # else:
    #     alpha = torch.rand((real_x.shape[0], 1, 1, 1))
    # x_ = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    x_ = real_x.requires_grad_(True)
    y_ = D.net(x_)
    # cal f'(x)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # grad = grad.view(x_.shape[0], -1)
    # div = (grad.norm(2, dim=1) ** p).mean()
    div = (grad * grad).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    div = torch.mean(div)
    return div

def div_loss(D, x, y, r1_gamma=10.0, cuda=False):
    x_ = x.requires_grad_(True)
    y_ = D.net(x_)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad * grad
    grad = grad.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    loss = grad.mean()
    return loss

def GAN_loss(scores_out, real=True):
    if real:
        return torch.mean(F.softplus(-scores_out))
    else:
        return torch.mean(F.softplus(scores_out))

def training_loop(
        config,
        dataset_args         = {},
        E_lr_args            = EasyDict(),
        D_lr_args            = EasyDict(),
        opt_args             = EasyDict(),
        E_loss_args          = EasyDict(),
        D_loss_args          = EasyDict(),
        logger               = None,
        writer               = None,
        image_snapshot_ticks = 50,
        max_epoch            = 50
):
    # parse
    loss_pix_weight = E_loss_args.loss_pix_weight
    loss_feat_weight = E_loss_args.loss_feat_weight
    loss_adv_weight = E_loss_args.loss_adv_weight
    loss_real_weight = D_loss_args.loss_real_weight
    loss_fake_weight = D_loss_args.loss_fake_weight
    loss_gp_weight = D_loss_args.loss_gp_weight
    loss_ep_weight = D_loss_args.loss_ep_weight
    E_learning_rate = E_lr_args.learning_rate
    D_learning_rate = D_lr_args.learning_rate

    # construct dataloader
    train_dataset = CelebaHQ(dataset_args, train=True)
    val_dataset = CelebaHQ(dataset_args, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)

    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids)
    E = StyleGANEncoder(config.model_name, logger, gpu_ids=config.gpu_ids)
    F = PerceptualModel(min_val=G.min_val, max_val=G.max_val, gpu_ids=config.gpu_ids)
    D = StyleGANDiscriminator(config.model_name, logger, gpu_ids=config.gpu_ids)
    G.net.synthesis.eval()
    E.net.train()
    F.net.eval()
    D.net.train()
    encode_dim = [G.num_layers, G.w_space_dim]

    # optimizer
    optimizer_E = torch.optim.Adam(E.net.parameters(), lr=E_learning_rate, **opt_args)
    optimizer_D = torch.optim.Adam(D.net.parameters(), lr=D_learning_rate, **opt_args)
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D, gamma=D_lr_args.decay_rate)

    global_step = 0
    for epoch in range(max_epoch):
        E_loss_rec = 0.
        E_loss_adv = 0.
        E_loss_feat = 0.
        D_loss_real = 0.
        D_loss_fake = 0.
        D_loss_grad = 0.
        learning_rate = lr_scheduler_E.get_lr()[0]
        for step, items in enumerate(train_dataloader):
            E.net.train()
            x = items
            x = x.float().cuda()
            batch_size = x.shape[0]
            z = E.net(x).view(batch_size, *encode_dim)
            x_rec = G.net.synthesis(z)

            # ===============================
            #         optimizing D
            # ===============================

            x_real = D.net(x)
            x_fake = D.net(x_rec.detach())
            loss_real = GAN_loss(x_real, real=True)
            loss_fake = GAN_loss(x_fake, real=False)
            # gradient div
            loss_gp = div_loss_(D, x, x_rec.detach(), cuda=config.cuda)
            # loss_gp = div_loss(D, x, x_real)

            D_loss_real += loss_real.item()
            D_loss_fake += loss_fake.item()
            D_loss_grad += loss_gp.item()
            log_message = f'D-[real:{loss_real.cpu().detach().numpy():.3f}, ' \
                          f'fake:{loss_fake.cpu().detach().numpy():.3f}, ' \
                          f'gp:{loss_gp.cpu().detach().numpy():.3f}]'
            D_loss = loss_real_weight * loss_real + loss_fake_weight * loss_fake + loss_gp_weight * loss_gp + loss_ep_weight * (loss_real * loss_real)
            D_loss.backward()
            optimizer_D.step()

            # ===============================
            #         optimizing G
            # ===============================
            # Reconstruction loss.
            loss_pix = torch.mean((x - x_rec) ** 2)
            E_loss_rec += loss_pix.item()
            log_message += f', G-[pix:{loss_pix.cpu().detach().numpy():.3f}'

            # Perceptual loss.
            loss_feat = 0.
            if loss_feat_weight:
                x_feat = F.net(x)
                x_rec_feat = F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
                E_loss_feat += loss_feat.item()
                log_message += f', feat:{loss_feat.cpu().detach().numpy():.3f}'

            # adversarial loss.
            loss_adv = 0.
            if loss_adv_weight:
                x_adv = D.net(x_rec)
                loss_adv = GAN_loss(x_adv, real=True)
                E_loss_adv += loss_adv.item()
                log_message += f', adv:{loss_adv.cpu().detach().numpy():.3f}]'

            E_loss = loss_pix_weight * loss_pix + loss_feat_weight * loss_feat + loss_adv_weight * loss_adv
            log_message += f', loss:{E_loss.cpu().detach().numpy():.3f}'
            optimizer_E.zero_grad()
            E_loss.backward()
            optimizer_E.step()

            # pbar.set_description_str(log_message)
            if logger:
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'Step:{step:04d}, '
                             f'lr:{learning_rate:.2e}, '
                             f'{log_message}')
            if writer:
                writer.add_scalar('D/loss_real', loss_real.item(), global_step=global_step)
                writer.add_scalar('D/loss_fake', loss_fake.item(), global_step=global_step)
                writer.add_scalar('D/loss_gp', loss_gp.item(), global_step=global_step)
                writer.add_scalar('D/loss', D_loss.item(), global_step=global_step)
                writer.add_scalar('E/loss_pix', loss_pix.item(), global_step=global_step)
                writer.add_scalar('E/loss_feat', loss_feat.item(), global_step=global_step)
                writer.add_scalar('E/loss_adv', loss_adv.item(), global_step=global_step)
                writer.add_scalar('E/loss', E_loss.item(), global_step=global_step)

            if step % image_snapshot_ticks == 0:
                E.net.eval()
                for val_step, val_items in enumerate(val_dataloader):
                    x_val = val_items
                    x_val = x_val.float().cuda()
                    batch_size_val = x_val.shape[0]
                    x_train = x[:batch_size_val, :, :, :]
                    z_train = E.net(x_train).view(batch_size_val, *encode_dim)
                    x_rec_train = G.net.synthesis(z_train)
                    z_val = E.net(x_val).view(batch_size_val, *encode_dim)
                    x_rec_val = G.net.synthesis(z_val)
                    x_all = torch.cat([x_val, x_rec_val, x_train, x_rec_train], dim=0)
                    if val_step > config.test_save_step:
                        break
                    save_filename = f'epoch_{epoch:03d}_step_{step:04d}_test_{val_step:04d}.png'
                    save_filepath = os.path.join(config.save_images, save_filename)
                    tvutils.save_image(x_all, filename=save_filepath, nrow=config.test_batch_size, normalize=True, scale_each=True)
            
            global_step += 1
            if (global_step + 1) % E_lr_args.decay_step == 0:
                lr_scheduler_E.step()
            if (global_step + 1) % D_lr_args.decay_step == 0:
                lr_scheduler_D.step()

        D_loss_real /= train_dataloader.__len__()
        D_loss_fake /= train_dataloader.__len__()
        D_loss_grad /= train_dataloader.__len__()
        E_loss_rec /= train_dataloader.__len__()
        E_loss_adv /= train_dataloader.__len__()
        E_loss_feat /= train_dataloader.__len__()
        log_message_ep = f'D-[real:{D_loss_real:.3f}, fake:{D_loss_fake:.3f}, gp:{D_loss_grad:.3f}], ' \
                         f'G-[pix:{E_loss_rec:.3f}, feat:{E_loss_feat:.3f}, adv:{E_loss_adv:.3f}]'
        if logger:
            logger.debug(f'Epoch: {epoch:03d}, '
                         f'lr: {learning_rate:.2e}, '
                         f'{log_message_ep}')

        save_filename = f'styleganinv_encoder_epoch_{epoch:03d}'
        save_filepath = os.path.join(config.save_models, save_filename)
        torch.save(E.net.module.state_dict(), save_filepath)

        