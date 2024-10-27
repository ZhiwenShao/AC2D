import argparse
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as util_data

import rest_v2
import pre_process as prep
from util import *
from data_list import ImageList_land_au

import numpy as np
import torch

from timm.models import create_model
from optim_factory import create_optimizer

import random
import warnings

warnings.filterwarnings('ignore')


def BCEWithLogitsLoss_PNWeight(input, target, p_n_weight, weight=None, size_average=True, reduce=True):
    r"""Function that measures Binary Cross Entropy between target and output with p_n_weight
    logits.

    Modified from pytorch BCEWithLogitsLoss.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on :attr:`size_average`. When :attr:`reduce`
                is ``False``, returns a loss per input/target element instead and ignores
                :attr:`size_average`. Default: ``True``

    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    # make all the negative value to positive and positive values to 0
    max_val = (-input).clamp(min=0)
    loss = (p_n_weight - 1) * target * (1 + (- input).exp()).log() + \
        input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = torch.relu(loss)
    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def attention_KLDiv_loss(input, target, size_average=True, reduce=True):
    classify_loss = nn.KLDivLoss(reduction='batchmean', size_average=size_average, reduce=reduce)

    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_target = target[:, i, :]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


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

    ## set loss criterion
    use_gpu = torch.cuda.is_available()
    au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt', dtype=float))
    au_p_n_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_p_n_weight.txt', dtype=float))
    if use_gpu:
        au_weight = au_weight.float().cuda()
        au_p_n_weight = au_p_n_weight.float().cuda()
    else:
        au_weight = au_weight.float()
        au_p_n_weight = au_p_n_weight.float()

    map_regress_criterion = nn.MSELoss()

    ## prepare data
    dsets = {}
    dset_loaders = {}

    dsets['train'] = ImageList_land_au(crop_size=config.crop_size, path=config.train_path_prefix,
                                        transform=prep.image_train(crop_size=config.crop_size),
                                        target_transform=prep.land_transform(img_size=config.crop_size,
                                                                             flip_reflect=np.loadtxt(
                                                                                 config.flip_reflect)))

    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)

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

    pretrain_net = create_model(
        config.net,
        pretrained=False,
        au_num=config.bp4d_au_num,
        drop_path_rate=config.drop_path,
        causal_dim=config.causal_dim
    )

    print('loading pretrained model ...')
    pretrain_state = torch.load(
        config.write_path_prefix + config.pretrain_net_path + '/net_' + str(
            config.pretrain_epoch) + '.pth')
    pretrain_net.load_state_dict(pretrain_state['model'])
    K_data = pretrain_state['K_data']
    V_data = pretrain_state['V_data']
    P_data = pretrain_state['P_data']
    config.inter_epoch = 0

    net.stem = pretrain_net.stem
    net.patch_2 = pretrain_net.patch_2
    net.stage1 = pretrain_net.stage1
    net.stage2 = pretrain_net.stage2
    net.aus_patch_3[0] = pretrain_net.aus_patch_3[0]
    net.aus_patch_3[1] = pretrain_net.aus_patch_3[1]
    net.aus_patch_3[2] = pretrain_net.aus_patch_3[2]
    net.aus_patch_3[3] = pretrain_net.aus_patch_3[3]
    net.aus_stage3_start[0] = pretrain_net.aus_stage3_start[0]
    net.aus_stage3_start[1] = pretrain_net.aus_stage3_start[1]
    net.aus_stage3_start[2] = pretrain_net.aus_stage3_start[2]
    net.aus_stage3_start[3] = pretrain_net.aus_stage3_start[3]
    net.aus_stage3_interm[0] = pretrain_net.aus_stage3_interm[0]
    net.aus_stage3_interm[1] = pretrain_net.aus_stage3_interm[1]
    net.aus_stage3_interm[2] = pretrain_net.aus_stage3_interm[2]
    net.aus_stage3_interm[3] = pretrain_net.aus_stage3_interm[3]
    net.aus_stage3_end[0] = pretrain_net.aus_stage3_end[0]
    net.aus_stage3_end[1] = pretrain_net.aus_stage3_end[1]
    net.aus_stage3_end[2] = pretrain_net.aus_stage3_end[2]
    net.aus_stage3_end[3] = pretrain_net.aus_stage3_end[3]
    net.aus_norm[0] = pretrain_net.aus_norm[0]
    net.aus_norm[1] = pretrain_net.aus_norm[1]
    net.aus_norm[2] = pretrain_net.aus_norm[2]
    net.aus_norm[3] = pretrain_net.aus_norm[3]
    net.aus_avg_pool[0] = pretrain_net.aus_avg_pool[0]
    net.aus_avg_pool[1] = pretrain_net.aus_avg_pool[1]
    net.aus_avg_pool[2] = pretrain_net.aus_avg_pool[2]
    net.aus_avg_pool[3] = pretrain_net.aus_avg_pool[3]
    net.aus_head[0] = pretrain_net.aus_head[0]
    net.aus_head[1] = pretrain_net.aus_head[1]
    net.aus_head[2] = pretrain_net.aus_head[2]
    net.aus_head[3] = pretrain_net.aus_head[3]

    if config.dataset_name=='DISFA':
        print('initializing for DISFA ...')
        net.aus_patch_3[5] = pretrain_net.aus_patch_3[6]
        net.aus_patch_3[6] = pretrain_net.aus_patch_3[10]
        net.aus_patch_3[7] = pretrain_net.aus_patch_3[9]
        net.aus_stage3_start[5] = pretrain_net.aus_stage3_start[6]
        net.aus_stage3_start[6] = pretrain_net.aus_stage3_start[10]
        net.aus_stage3_start[7] = pretrain_net.aus_stage3_start[9]
        net.aus_stage3_interm[5] = pretrain_net.aus_stage3_interm[6]
        net.aus_stage3_interm[6] = pretrain_net.aus_stage3_interm[10]
        net.aus_stage3_interm[7] = pretrain_net.aus_stage3_interm[9]
        net.aus_stage3_end[5] = pretrain_net.aus_stage3_end[6]
        net.aus_stage3_end[6] = pretrain_net.aus_stage3_end[10]
        net.aus_stage3_end[7] = pretrain_net.aus_stage3_end[9]
        net.aus_norm[5] = pretrain_net.aus_norm[6]
        net.aus_norm[6] = pretrain_net.aus_norm[10]
        net.aus_norm[7] = pretrain_net.aus_norm[9]
        net.aus_avg_pool[5] = pretrain_net.aus_avg_pool[6]
        net.aus_avg_pool[6] = pretrain_net.aus_avg_pool[10]
        net.aus_avg_pool[7] = pretrain_net.aus_avg_pool[9]
        net.aus_head[5] = pretrain_net.aus_head[6]
        net.aus_head[6] = pretrain_net.aus_head[10]
        net.aus_head[7] = pretrain_net.aus_head[9]

    if config.start_epoch > 0:
        print('resuming model from epoch %d' %(config.start_epoch))
        state = torch.load(
            config.write_path_prefix + config.net + '/' + config.run_name + '/net_' + str(
                config.start_epoch) + '.pth')
        if config.start_epoch>= config.inter_epoch:
            net.load_state_dict(state['model'])
            K_data = state['K_data']
            V_data = state['V_data']
            P_data = state['P_data']
        else:
            net.load_state_dict(state)

    if use_gpu:
        net = net.cuda()

    print(net)

    num_training_steps_per_epoch = len(dset_loaders['train'])
    config.lr = config.lr * config.train_batch_size / 256

    
    ## set optimizer
    optimizer = create_optimizer(config, net)

    loss_scaler = NativeScalerWithGradNormCount()
    
    print("Use Cosine LR scheduler")
    lr_schedule_values = cosine_scheduler(
        config.lr, config.min_lr, config.epochs, num_training_steps_per_epoch,
        warmup_epochs=config.warmup_epochs, warmup_steps=config.warmup_steps,
    )

    if config.weight_decay_end is None:
        config.weight_decay_end = config.weight_decay
    wd_schedule_values = cosine_scheduler(
        config.weight_decay, config.weight_decay_end, config.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if not os.path.exists(config.write_path_prefix + config.net + '/' + config.run_name):
        os.makedirs(config.write_path_prefix + config.net + '/' + config.run_name)
    if not os.path.exists(config.write_res_prefix + config.net + '/' + config.run_name):
        os.makedirs(config.write_res_prefix + config.net + '/' + config.run_name)

    res_file = open(
        config.write_res_prefix + config.net + '/' + config.run_name + '/AU_pred_' + str(config.start_epoch) + '.txt', 'w')
    res_file2 = open(
        config.write_res_prefix + config.net + '/' + config.run_name + '/AU_pred_' + str(config.start_epoch) + '_details.txt', 'w')

    ## train
    count = 0

    for epoch in range(config.start_epoch, config.epochs + 1):
        if epoch > config.start_epoch:
            print('taking snapshot ...')
            if epoch>= config.inter_epoch:
                state = {
                    'model': net.state_dict(),
                    'K_data': K_data,
                    'V_data': V_data,
                    'P_data': P_data
                }
                torch.save(state,
                           config.write_path_prefix + config.net + '/' + config.run_name + '/net_' + str(
                               epoch) + '.pth')
            else:
                torch.save(net.state_dict(),
                           config.write_path_prefix + config.net + '/' + config.run_name + '/net_' + str(
                               epoch) + '.pth')

        # eval in the train
        if epoch > config.start_epoch:
            print('testing ...')
            net.train(False)

            if epoch >= config.inter_epoch:
                if use_gpu:
                    K_data, V_data = K_data.float().cuda(), V_data.float().cuda()
                f1score_arr, acc_arr = AU_detection_eval(dset_loaders['test'], net, data_infos=[K_data, V_data ], use_gpu=use_gpu)
            else:
                f1score_arr, acc_arr = AU_detection_eval(dset_loaders['test'], net,
                                                         use_gpu=use_gpu)
            print('epoch =%d, f1 score mean=%f, accuracy mean=%f' %
                  (epoch, f1score_arr.mean(), acc_arr.mean()))
            print('%d\t%f\t%f' % (epoch, f1score_arr.mean(), acc_arr.mean()), file=res_file)
            print(f1score_arr, acc_arr, file=res_file2)
            res_file.close()
            res_file2.close()
            res_file = open(
                config.write_res_prefix + config.net + '/' + config.run_name + '/AU_pred_' + str(
                    config.start_epoch) + '.txt', 'a')
            res_file2 = open(
        config.write_res_prefix + config.net + '/' + config.run_name + '/AU_pred_' + str(config.start_epoch) + '_details.txt', 'a')   

            net.train(True)

        if epoch >= config.epochs:
            break

        start_steps = epoch * num_training_steps_per_epoch
        optimizer.zero_grad()
        for i, batch in enumerate(dset_loaders['train']):

            step = i // config.update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if lr_schedule_values is not None or wd_schedule_values is not None and i % config.update_freq == 0:
                for j, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            if i % config.display == 0 and count > 0:
                print('[epoch = %d][iter = %d][total_loss = %f][loss_au = %f][loss_attention = %f]' % (epoch, i,
                                                                                                       total_loss.data.cpu().numpy(),
                                                                                                       loss_au.data.cpu().numpy(),
                                                                                                       loss_attention.data.cpu().numpy()))
                print('learning rate = %f' % (optimizer.param_groups[0]['lr']))
                print('weight decay = %f' % (optimizer.param_groups[0]['weight_decay']))
                print('the number of training iterations is %d' % (count))

            img, land, au = batch

            if use_gpu:
                img, land, au = img.cuda(), land.float().cuda(), au.float().cuda()
            else:
                land, au = land.float(), au.float()

            with torch.cuda.amp.autocast():
                if epoch > config.inter_epoch-1:
                    if use_gpu:
                        K_data, V_data = K_data.float().cuda(), V_data.float().cuda()
                    data_infos=[K_data, V_data]
                else:
                    data_infos=None
                
                aus_feature, aus_attention, aus_output = net([img, data_infos])
                aus_feature = aus_feature.data.cpu().numpy()

                coord = prepare_coord(land, config.au_num, config.dataset_name)
                gt_attention = render_gaussian_heatmap(coord, config.crop_size, config.au_num * 2,
                                                        shrink=config.shrink,
                                                        sigma=config.sigma, use_gpu=use_gpu)

                gt_attention_1 = gt_attention[:, :, :, 0:gt_attention.size(3):2]
                gt_attention_2 = gt_attention[:, :, :, 1:gt_attention.size(3):2]
                gt_attention = torch.max(gt_attention_1, gt_attention_2)
                gt_attention = gt_attention.permute(0, 3, 1, 2)
                gt_attention = gt_attention.reshape(gt_attention.size(0), gt_attention.size(1), -1)
                gt_attention = F.normalize(gt_attention, p=1, dim=2)

                aus_attention = aus_attention.reshape(aus_attention.size(0), aus_attention.size(1), -1, gt_attention.size(2))
                aus_attention = aus_attention.mean(dim=2)

                aus_attention = torch.log(aus_attention)
                loss_attention = attention_KLDiv_loss(aus_attention, gt_attention)
                loss_au = BCEWithLogitsLoss_PNWeight(aus_output, au, au_p_n_weight, au_weight)


                if epoch >= config.inter_epoch - 1:
                    if i == 0:
                        all_feature = np.sum(aus_feature, axis=0)
                        sample_num = aus_feature.shape[0]
                    else:
                        all_feature = all_feature + np.sum(aus_feature, axis=0)
                        sample_num = sample_num + aus_feature.shape[0]

                total_loss = config.lambda_au * loss_au + config.lambda_attention * loss_attention

                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                total_loss /= config.update_freq
                grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.clip_grad,
                                        parameters=net.parameters(), create_graph=is_second_order,
                                        update_grad=(i + 1) % config.update_freq == 0)
                if (i + 1) % config.update_freq == 0:
                    optimizer.zero_grad()

            torch.cuda.synchronize()

            count = count + 1

        if epoch >= config.inter_epoch - 1:
            P_data = 1 / sample_num
            K_data = torch.from_numpy(all_feature/sample_num).to(torch.float32)
            V_data = P_data * K_data

    res_file.close()
    res_file2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--train_batch_size', type=int, default=24, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=20, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--causal_dim', type=int, default=512)

    parser.add_argument('--net', type=str, default='restv2_tiny_ac2d')
    parser.add_argument('--run_name', type=str, default='DISFA_combine_1_2')
    parser.add_argument('--dataset_name', type=str, default='DISFA')
    parser.add_argument('--bp4d_au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--pretrain_epoch', type=int, default=11, help='pretraining epoch')
    parser.add_argument('--pretrain_net_path', type=str,
                        default='restv2_tiny_ac2d/BP4D_combine_1_3')

    # Training configuration.
    parser.add_argument('--shrink', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=3)
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_attention', type=float, default=12800, help='weight for landmark detection loss')
    parser.add_argument('--display', type=int, default=100, help='iteration gaps for displaying')

    # Directories.
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--train_path_prefix', type=str, default='data/list/v6/DISFA_combine_1_2')
    parser.add_argument('--test_path_prefix', type=str, default='data/list/v6/DISFA_part3')

    #----------------------------From ResT---------------------------------
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--inter_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (If None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3)')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--seed', default=4, type=int)

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)
