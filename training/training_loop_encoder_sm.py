from datasets.celebahq import CelebaHQ
from datasets.afhq import Afhq,RandomSampler
from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel
# from models.stylegan_discriminator import Discriminator
from models.naive_discriminator import Discriminator
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torchvision.utils as tvutils
import torch.autograd as autograd
from torch.utils.data import DataLoader
import itertools



def div_loss(D, real_x, fake_x, p=6, cuda=False):
    if cuda:
        alpha = torch.rand((real_x.shape[0], 1, 1, 1)).cuda()
    else:
        alpha = torch.rand((real_x.shape[0], 1, 1, 1))
    x_ = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    y_ , _ = D(x_)
    # cal f'(x)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True, ####
        only_inputs=True,
    )[0]
    grad = grad.view(x_.shape[0], -1)
    div = (grad.norm(2, dim=1) ** p).mean() # 
    return div


def training_loop(
        config,
        dataset_args         = {},
        lr_args              = EasyDict(),
        opt_args             = EasyDict(),
        E_loss_args          = EasyDict(),
        D_loss_args          = EasyDict(),
        logger               = None,
        writer               = None,
        image_snapshot_ticks = 100, 
        max_epoch            = 40
):
   
    assert max_epoch >= 20, 'max_epoch must be larger than 20.'
    decay_step = max_epoch // 20    

    # parse
    loss_pix_weight = E_loss_args.loss_pix_weight
    loss_feat_weight = E_loss_args.loss_feat_weight
    loss_adv_weight = E_loss_args.loss_adv_weight
    loss_real_weight = D_loss_args.loss_real_weight
    loss_fake_weight = D_loss_args.loss_fake_weight
    loss_gp_weight = D_loss_args.loss_gp_weight
    loss_cam_weight =D_loss_args.loss_cam_weight
    learning_rate = lr_args.learning_rate

    # construct dataloader

    # afhq_train = Afhq(dataset_args, train=True)
    # afhq_test = Afhq(dataset_args, train=False)

    # afhq_dataloader = DataLoader(afhq_train, batch_size=config.train_batch_size, shuffle=True)
    
    # create celeba-hq dataset
    dataset_args.data_root = dataset_args.data_root2
    celeba_train = CelebaHQ(dataset_args,train=True)

    celeba_dataloader = DataLoader(celeba_train,batch_size = config.train_batch_size,shuffle=True)

    celeba_test = CelebaHQ(dataset_args,train=False)

    # construct model
    G = StyleGANGenerator(config.model_name, logger, gpu_ids=config.gpu_ids)
    E = StyleGANEncoder(config.model_name, logger, gpu_ids=config.gpu_ids)
    if config.resume_id is not None:
        # E.net.module.is_gpu = False
        E.weight_path = config.resume_id
        E.load()

    F = PerceptualModel(min_val=G.min_val, max_val=G.max_val, gpu_ids=config.gpu_ids)
    D = Discriminator()
    D = D.cuda()
    D = nn.DataParallel(D, config.gpu_ids)
    encode_dim = [G.num_layers, G.w_space_dim]

    # optimizer
    optimizer_E = torch.optim.Adam(E.net.module.parameters(), lr=learning_rate, **opt_args)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=learning_rate, **opt_args)
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=lr_args.decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D, gamma=lr_args.decay_rate)

    global_step = 0
    for epoch in range(max_epoch):
        E_loss_rec = 0.
        E_loss_adv = 0.
        E_loss_feat = 0.
        D_loss_real = 0.
        D_loss_fake = 0.
        D_loss_grad = 0.
        D_loss_cam = 0.
        learning_rate = lr_scheduler_E.get_last_lr()[0]
        for step, items in enumerate(celeba_dataloader):
            E.net.train()
            
            x = items
            x = x.float().cuda()
            batch_size = x.shape[0]
                
            z = E.net(x).view(batch_size,*encode_dim)
            x_rec = G.net.synthesis(z)

            # ===============================
            #         optimizing D
            # ===============================
            real_adv = D(x)
            fake_adv = D(mix_rec.detach())

            loss_real = - torch.mean(__real__)
            loss_fake = torch.mean(__fake__)

            # gradient div
            loss_gp = div_loss(D, mix_real, mix_rec.detach(), cuda=config.cuda)

            # BCE Loss for classification
            # loss_cam = torch.nn.BCELoss()(cam,target) # 选用BCELoss
            D_loss_real += loss_real.item()
            D_loss_fake += loss_fake.item()
            D_loss_grad += loss_gp.item()
            # D_loss_cam += loss_cam.item()

            log_message = f'D-[real:{loss_real.cpu().detach().numpy():.3f}, ' \
                          f'fake:{loss_fake.cpu().detach().numpy():.3f}, ' \
                          f'gp:{loss_gp.cpu().detach().numpy():.3f}' \
                        #   f'cls:{loss_cam.cpu().detach().numpy():3f}]'

            D_loss = loss_real_weight * loss_real + loss_fake_weight * loss_fake + loss_gp_weight * loss_gp #+ loss_cam_weight * loss_cam # 对生成的两个数据引入分类loss
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            torch.cuda.empty_cache()
            # ===============================
            #         optimizing G
            # ===============================
            # Reconstruction loss.
            loss_pix = torch.mean((mix_real- mix_rec) ** 2)
            E_loss_rec += loss_pix.item()
            log_message += f', G-[pix:{loss_pix.cpu().detach().numpy():.3f}'
            # Perceptual loss.
            loss_feat = 0.
            if loss_feat_weight:
                mix_feat = F.net(mix_real)
                x_rec_feat = F.net(mix_rec)
                loss_feat = torch.mean((mix_feat - x_rec_feat) ** 2)
                E_loss_feat += loss_feat.item()
                log_message += f', feat:{loss_feat.cpu().detach().numpy():.3f}'

            # adversarial loss.
            loss_adv = 0.
            x_adv , cam_adv = D(mix_rec)
            if loss_adv_weight:
                loss_adv = - torch.mean(x_adv)
                E_loss_adv += loss_adv.item()
                log_message += f', adv:{loss_adv.cpu().detach().numpy():.3f}]'
            
            if loss_cam_weight > 0:
                # print(cam_adv.shape,target.shape)
                loss_cam = -torch.nn.BCELoss()(cam_adv,E_target)
                loss_cam.backward() # 额外分支的梯度回传
                E_loss_adv += loss_cam.item()
                log_message += f', cam:{loss_cam.cpu().detach().numpy():.3f}]'
            
            E_loss = loss_pix_weight * loss_pix + loss_feat_weight * loss_feat + loss_adv_weight * loss_adv #+ loss_cam_weight * loss_cam
            log_message += f', loss:{E_loss.cpu().detach().numpy():.3f}'
            optimizer_E.zero_grad()
            E_loss.backward()
            optimizer_E.step()

            # 下面全是test部分

            if writer is not None:
                writer.add_scalar('Discriminator/loss_real', loss_real.item(), global_step)
                writer.add_scalar('Discriminator/loss_fake', loss_fake.item(), global_step)
                writer.add_scalar('Discriminator/loss_gp', loss_gp.item(), global_step)
                writer.add_scalar('Discriminator/loss_D', D_loss.item(), global_step) 

                writer.add_scalar('Encoder/loss_pix', loss_pix.item(), global_step)
                writer.add_scalar('Encoder/loss_feat', loss_fake.item(), global_step)
                if loss_adv_weight:
                    writer.add_scalar('Encoder/loss_adv', loss_adv.item(), global_step)
                writer.add_scalar('Encoder/loss_E', E_loss.item(), global_step)
            global_step += 1
          
            # print(step)
            if step % image_snapshot_ticks == 0:
                if logger is not None:
                    logger.debug(f'Epoch:{epoch:03d}, '
                        f'Step:{step:04d}, '
                        f'lr:{learning_rate:.2e}, '
                        f'{log_message}')

                E.net.eval()

                # 这里改成了随机抽样来展示图片
                idx = afhq_test.get_batch_index(config.test_batch_size)
                test_item = [torch.unsqueeze(afhq_test.__getitem__(id),dim=0) for id in idx]
                test_item = torch.cat(test_item,dim=0)
                test_item = test_item.float().cuda()
                batch_size = test_item.shape[0]
                z = E.net(test_item).view(batch_size,*encode_dim)
                test_rec = G.net.synthesis(z)

                idx = celeba_test.get_batch_index(config.test_batch_size)
                val_item = [torch.unsqueeze(celeba_test.__getitem__(id),dim=0) for id in idx]
                val_item = torch.cat(val_item,dim=0)
                val_item = val_item.float().cuda()
                batch_size = val_item.shape[0]
                z = E.net(val_item).view(batch_size,*encode_dim)
                val_rec = G.net.synthesis(z)

                # 增加训练集里面本身的数据
                mix_real = mix_real[:config.test_batch_size]
                mix_real = mix_real.float().cuda()
                batch_size = mix_real.shape[0]
                z = E.net(mix_real).view(batch_size,*encode_dim)
                mix_rec = G.net.synthesis(z)

                x_all = torch.cat([test_item,test_rec,val_item,val_rec,mix_real,mix_rec], dim=0)

                save_filename = f'epoch_{epoch:03d}_step_{step:04d}_test.png'
                save_filepath = os.path.join(config.save_images, save_filename)
                tvutils.save_image(x_all, fp=str(save_filepath), nrow=4, normalize=True, scale_each=True) #参数做了修改,版本有问题
            
        save_filename = f'encoder_epoch_{epoch:03d}_step{step}'
        save_filepath = os.path.join(config.save_models, save_filename)
        torch.save(E.net.state_dict(), save_filepath)

        D_loss_real /= afhq_dataloader.__len__()
        D_loss_fake /= afhq_dataloader.__len__()
        D_loss_grad /= afhq_dataloader.__len__()
        E_loss_rec /= afhq_dataloader.__len__()
        E_loss_adv /= afhq_dataloader.__len__()
        E_loss_feat /= afhq_dataloader.__len__()
        log_message_ep = f'D-[real:{D_loss_real:.3f}, fake:{D_loss_fake:.3f}, gp:{D_loss_grad:.3f}], ' \
                         f'G-[pix:{E_loss_rec:.3f}, feat:{E_loss_feat:.3f}, adv:{E_loss_adv:.3f}]'
        
        if (epoch+1) % decay_step == 0:
            lr_scheduler_E.step()
            lr_scheduler_D.step()
