import argparse
import os
import torch.nn as nn
import torch.utils.data as util_data

import rest_v2
import pre_process as prep
from util import *
from data_list import ImageList_land_au

import numpy as np
import torch


from timm.models import create_model

import random
import warnings

warnings.filterwarnings('ignore')


def set_random_seed(SEED=4):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    # fix the seed for reproducibility
    set_random_seed(config.seed)

    use_gpu = torch.cuda.is_available()
    
    ## prepare data
    dsets = {}
    dset_loaders = {}
    dsets['test'] = ImageList_land_au(crop_size=config.crop_size, phase='test', path=config.test_path_prefix,
                      transform=prep.image_test(crop_size=config.crop_size),
                      target_transform=prep.land_transform(img_size=config.crop_size,
                                                           flip_reflect=np.loadtxt(
                                                               config.flip_reflect)))
    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    ## set network modules
    net = create_model(
        config.net,
        pretrained=False,
        au_num=config.au_num,
        drop_path_rate=config.drop_path,
        causal_dim=config.causal_dim
    )
    if use_gpu:
        net = net.cuda()

    if not os.path.exists(config.write_path_prefix + config.net + '/' + config.run_name):
        os.makedirs(config.write_path_prefix + config.net + '/' + config.run_name)
    if not os.path.exists(config.write_res_prefix + config.net + '/' + config.run_name):
        os.makedirs(config.write_res_prefix + config.net + '/' + config.run_name)

    if config.start_epoch <= 0:
        raise (RuntimeError('start_epoch should be larger than 0\n'))

    if config.pred_AU:
        res_file = open(
            config.write_res_prefix + config.net + '/' + config.run_name + '/offline_AU_pred_' + str(config.start_epoch) + '.txt', 'w')
        res_file2 = open(
            config.write_res_prefix + config.net + '/' + config.run_name + '/offline_AU_pred_' + str(
                config.start_epoch) + '_details.txt', 'w')

    net.train(False)

    for epoch in range(config.start_epoch, config.epochs + 1):
        state = torch.load(
            config.write_path_prefix + config.net + '/' + config.run_name + '/net_' + str(epoch) + '.pth')
        if epoch >= config.inter_epoch:
            net.load_state_dict(state['model'])
            K_data = state['K_data']
            V_data = state['V_data']
            P_data = state['P_data']
            if use_gpu:
                K_data, V_data = K_data.float().cuda(), V_data.float().cuda()
        else:
            net.load_state_dict(state)

        if config.pred_AU:
            if epoch >= config.inter_epoch:
                f1score_arr, acc_arr = AU_detection_eval(dset_loaders['test'], net, data_infos=[K_data, V_data], use_gpu=use_gpu)
            else:
                f1score_arr, acc_arr = AU_detection_eval(dset_loaders['test'], net,
                                                         use_gpu=use_gpu)

            print('epoch =%d, f1 score mean=%f, accuracy mean=%f' % (epoch, f1score_arr.mean(), acc_arr.mean()))
            print('%d\t%f\t%f' % (epoch, f1score_arr.mean(), acc_arr.mean()), file=res_file)
            print(f1score_arr, acc_arr, file=res_file2)
        if config.vis_attention:
            if not os.path.exists(config.write_res_prefix + config.net + '/' + config.run_name + '/vis_map/' + str(epoch)):
                os.makedirs(config.write_res_prefix + config.net + '/' + config.run_name + '/vis_map/' + str(epoch))
            if not os.path.exists(config.write_res_prefix + config.net + '/' + config.run_name + '/overlay_vis_map/' + str(epoch)):
                os.makedirs(config.write_res_prefix + config.net + '/' + config.run_name + '/overlay_vis_map/' + str(epoch))

            vis_attention_former(dset_loaders['test'], net, config.write_res_prefix, config.net + '/' + config.run_name, epoch,
                          config.au_num, config.dataset_name, config.shrink, config.sigma, use_gpu=use_gpu)
        if config.cal_flops:
            if epoch >= config.inter_epoch:
                eval_flops_former(dset_loaders['test'], net, epoch, data_infos=[K_data, V_data], use_gpu=use_gpu)
            else:
                eval_flops_former(dset_loaders['test'], net, epoch, use_gpu=use_gpu)

    if config.pred_AU:
        res_file.close()
        res_file2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--eval_batch_size', type=int, default=20, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--causal_dim', type=int, default=512)

    parser.add_argument('--pred_AU', type=str2bool, default=True)
    parser.add_argument('--vis_attention', type=str2bool, default=False)
    parser.add_argument('--cal_flops', type=str2bool, default=False)

    parser.add_argument('--net', type=str, default='restv2_tiny_ac2d')
    parser.add_argument('--run_name', type=str, default='BP4D_combine_1_2')
    parser.add_argument('--dataset_name', type=str, default='BP4D')

    # Directories.
    parser.add_argument('--shrink', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=3)
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--train_path_prefix', type=str, default='data/list/BP4D_combine_1_2')
    parser.add_argument('--test_path_prefix', type=str, default='data/list/BP4D_part3')

    # ----------------------------From ResT---------------------------------
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--inter_epoch', type=int, default=1, help='starting epoch')

    # Model parameters
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    

    parser.add_argument('--seed', default=4, type=int)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)
