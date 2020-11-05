# from datasets.celebahq import Config
from training.misc import EasyDict
from training.training_loop_encoder import training_loop
from utils.logger import setup_logger
from tensorboardX import SummaryWriter

import os
import argparse
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'

def main():
    parser = argparse.ArgumentParser(description='Training the in-domain encoder through pytorch')
    parser.add_argument('--data_root', type=str, default='/mnt/ssd2/xintian/datasets/celeba_hq/',
                        help='path to training data (.txt path file)')
    # parser.add_argument('--decoder_pkl', type=str, default='',
    #                     help='path to the stylegan generator, which serves as a decoder here.')
    # parser.add_argument('--num_gpus', type=int, default=3,
    #                     help='Number of GPUs to use during training (defaults: 8)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='the image size in training dataset (defaults; 256)')
    parser.add_argument('--model_name', type=str, default='styleganinv_ffhq256',
                        help='the name of the model')
    parser.add_argument('--dataset_name', type=str, default='ffhq',
                        help='the name of the training dataset (defaults; ffhq)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=8,
                        help='training batch size')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='to use cuda or not')
    parser.add_argument('--gpu_ids', type=list, default=[0,1,2,3],
                        help='list of gpus')
    parser.add_argument('--test_save_step', type=int, default=0,
                        help='how much step to be saved when inference')
    parser.add_argument('--save_root', type=str, default='/mnt/ssd2/xintian/idinvert_pytorch')
    args = parser.parse_args()

    current_time = datetime.now().strftime('%b%d_%H-%M')
    # prefix = 'FFHQ-CelebAHQ-InitEncoder_naiveD'
    prefix = 'FFHQ-CelebAHQ-InitEncoder_StyleD'
    # prefix = 'FFHQ-CelebAHQ-PretrainEncoder_naiveD'
    args.save_images = os.path.join(args.save_root, prefix + current_time, 'save_images')
    args.save_models = os.path.join(args.save_root, prefix + current_time, 'save_models')
    args.save_logs = os.path.join(args.save_root, prefix + current_time, 'save_logs')
    if not os.path.exists(args.save_images):
        os.makedirs(args.save_images)
    if not os.path.exists(args.save_models):
        os.makedirs(args.save_models)
    if not os.path.exists(args.save_logs):
        os.makedirs(args.save_logs)
    writer = SummaryWriter(os.path.join(args.save_root, prefix + current_time))

    class Config:
        data_root = args.data_root
        size = 256
        min_val = -1.0
        max_val = 1.0
    datasets_args = Config()

    opt_args = EasyDict(betas=(0.9, 0.99), eps=1e-8)
    E_lr_args = EasyDict(learning_rate=0.00001, decay_step=3000, decay_rate=0.8, stair=False)
    D_lr_args = EasyDict(learning_rate=0.00001, decay_step=3000, decay_rate=0.8, stair=False)

    E_loss_args = EasyDict(loss_pix_weight=1.0, loss_feat_weight=0.00005, loss_adv_weight=0.1)
    D_loss_args = EasyDict(loss_real_weight=1.0, loss_fake_weight=1.0, loss_gp_weight=5.0, loss_ep_weight=0.001)

    logger = setup_logger(args.save_logs, 'inversion.log', 'inversion_logger')
    logger.info(f'Loading model.')

    training_loop(args, datasets_args, E_lr_args, D_lr_args, opt_args, E_loss_args, D_loss_args, logger, writer)


if __name__ == '__main__':
    main()
