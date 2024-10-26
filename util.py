import torch
import torch.nn.functional as F
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

import time
from collections import defaultdict, deque
import datetime
from timm.utils import get_state_dict

from pathlib import Path

import torch.nn as nn
import torch.distributed as dist
from torch import inf

from fvcore.nn import FlopCountAnalysis

def str2bool(v):
    return v.lower() in ('true')

def tensor2img(img):
    img = img.data.numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0))+ 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def save_img(img, name, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path + name + '.png')
    return img

def AU_detection_eval(loader, net, data_infos=None, use_gpu=True):
    missing_label = 9
    for i, batch in enumerate(loader):
        img, land, label = batch
        if use_gpu:
            img, label = img.cuda(), label.cuda()

        aus_output = net([img, data_infos])

        if type(aus_output) is tuple:
            aus_output = torch.sigmoid(aus_output[-1])
        else:
            aus_output = torch.sigmoid(aus_output)
        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_label = label.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_label.data.numpy()

    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    return f1score_arr, acc_arr


def eval_flops_former(loader, net, epoch, data_infos=None, use_gpu=True):
    for i, batch in enumerate(loader):
        img, land, label = batch
        if i > 0:
            break
        print('Epoch '+str(epoch)+', Batch '+str(i))
        if use_gpu:
            img, label = img.cuda(), label.cuda()

        input = [img, data_infos]
        net_flops = FlopCountAnalysis(net, input)
        num_flops = net_flops.total()
        print('flops:', num_flops)
        print(net_flops.by_operator())

        # sum(p.numel() for p in model.parameters() if p.requires_grad)
        para_num = sum(p.numel() for p in net.parameters())
        print('para num:', para_num)


def vis_attention_former(loader, net, write_path_prefix, net_name, epoch, au_num, dataset_name, shrink, sigma, crop_size=176, alpha = 0.5, use_gpu=True):
    for i, batch in enumerate(loader):
        img, land, label = batch
        if use_gpu:
            img, land, label = img.cuda(), land.float().cuda(), label.cuda()
        else:
            land = land.float()

        coord = prepare_coord(land, au_num, dataset_name)
        gt_attention = render_gaussian_heatmap(coord, crop_size, au_num * 2,
                                               shrink=shrink,
                                               sigma=sigma, use_gpu=use_gpu)

        gt_attention_1 = gt_attention[:, :, :, 0:gt_attention.size(3):2]
        gt_attention_2 = gt_attention[:, :, :, 1:gt_attention.size(3):2]
        gt_attention = torch.max(gt_attention_1, gt_attention_2)
        gt_attention = gt_attention.permute(0, 3, 1, 2).contiguous()
        tmp = gt_attention.view(gt_attention.size(0), gt_attention.size(1), -1)
        tmp = F.normalize(tmp, p=1, dim=2)

        input = [img, None]
        aus_output = net(input)
        if type(aus_output) is not tuple:
            raise (RuntimeError('output attention is required\n'))
        else:
            output_attention = aus_output[-2]
            output_attention = output_attention.reshape(output_attention.size(0), output_attention.size(1), -1,
                                                  tmp.size(2))
            #Compute average self-attention over channels, you can also select certain channels to show 
            output_tmp = output_attention.mean(dim=2)


        att_max, _ =tmp.max(dim=2)
        att_max = att_max.unsqueeze(2)
        att_max = att_max.repeat(1, 1, tmp.size(2))
        tmp = tmp * 1.0 / att_max
        gt_attention = tmp.view(gt_attention.size(0), gt_attention.size(1), gt_attention.size(2), gt_attention.size(3))

        output_tmp = output_tmp * 1.0 / att_max
        output_attention = output_tmp.view(gt_attention.size(0), gt_attention.size(1), gt_attention.size(2), gt_attention.size(3))

        spatial_attention = output_attention


        if i == 0:
            all_input = img.data.cpu().float()
            all_label = label.data.cpu()
            all_spatial_attention = spatial_attention.data.cpu().float()
        else:
            all_input = torch.cat((all_input, img.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.data.cpu()), 0)
            all_spatial_attention = torch.cat((all_spatial_attention, spatial_attention.data.cpu().float()), 0)

    for i in range(all_spatial_attention.shape[0]):
        background = save_img(all_input[i], 'input', write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_')
        flag = True
        for j in range(all_spatial_attention.shape[1]):
            fig, ax = plt.subplots()
            cax = ax.imshow(all_spatial_attention[i,j], cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #        cbar = fig.colorbar(cax)
            fig.savefig(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
            if all_label[i, j]:
                if flag:
                    combine_spatial_attention = all_spatial_attention[i, j].unsqueeze(0)
                    flag = False
                else:
                    combine_spatial_attention = torch.cat(
                        (combine_spatial_attention, all_spatial_attention[i, j].unsqueeze(0)), 0)
        combine_spatial_attention = combine_spatial_attention.max(dim=0)
        combine_spatial_attention = combine_spatial_attention[0]

        cax = ax.imshow(combine_spatial_attention, cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
        ax.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        #        cbar = fig.colorbar(cax)
        fig.savefig(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                    '/' + str(i) + '_au_' + str(all_spatial_attention.shape[1]) + '.png', bbox_inches='tight', pad_inches=0)
        for j in range(all_spatial_attention.shape[1]+1):
            overlay = Image.open(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png')
            overlay = overlay.resize(background.size, Image.ANTIALIAS)
            background = background.convert('RGBA')
            overlay = overlay.convert('RGBA')
            new_img = Image.blend(background, overlay, alpha)
            new_img.save(write_path_prefix + net_name + '/overlay_vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', 'PNG')


def prepare_coord(land, au_num, dataset_name):

    coord = np.zeros((land.size(0), au_num * 2, 2))
    land = land.data.cpu().numpy()
    for i in range(land.shape[0]):
        land_array = land[i,:]
        str_dt = np.append(land_array[0:len(land_array):2], land_array[1:len(land_array):2])
        arr2d = np.array(str_dt).reshape((2, int(str_dt.shape[0]/2)))
        ruler = abs(arr2d[0, 22] - arr2d[0, 25])

        coord[i, 0:8, :] = np.array([[arr2d[0, 4], arr2d[1, 4] - ruler / 2], [arr2d[0, 5], arr2d[1, 5] - ruler / 2],# au1
                                     [arr2d[0, 1], arr2d[1, 1] - ruler / 3], [arr2d[0, 8], arr2d[1, 8] - ruler / 3],# au2
                                     [arr2d[0, 2], arr2d[1, 2] + ruler / 3], [arr2d[0, 7], arr2d[1, 7] + ruler / 3],# au4
                                     [arr2d[0, 24], arr2d[1, 24] + ruler], [arr2d[0, 29], arr2d[1, 29] + ruler],# au6
        ])
        if dataset_name=='BP4D':
            coord[i, 8:24, :] = np.array([[arr2d[0, 21], arr2d[1, 21]], [arr2d[0, 26], arr2d[1, 26]],# au7
                                          [arr2d[0, 43], arr2d[1, 43]], [arr2d[0, 45], arr2d[1, 45]],# au10
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],# au12 au14 au15
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],
                                          [arr2d[0, 39], arr2d[1, 39] + ruler / 2], [arr2d[0, 41], arr2d[1, 41] + ruler / 2],# au17
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]],# au23 au24
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]]
            ])
        elif dataset_name=='DISFA':
            coord[i, 8:16, :] = np.array([[arr2d[0, 15], arr2d[1, 15] - ruler / 2], [arr2d[0, 17], arr2d[1, 17] - ruler / 2],# au9
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],# au12
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]],# au25
                                          [arr2d[0, 39], arr2d[1, 39] + ruler / 2], [arr2d[0, 41], arr2d[1, 41] + ruler / 2]# au26


            ])
        elif dataset_name=='GFT':
            coord[i, 8:20, :] = np.array([[arr2d[0, 43], arr2d[1, 43]], [arr2d[0, 45], arr2d[1, 45]],# au 10
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],# au12 au14 au15
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]],# au23 au24
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]]

            ])
        elif dataset_name=='Aff-Wild2_2021version':
            coord[i, 8:24, :] = np.array([[arr2d[0, 21], arr2d[1, 21]], [arr2d[0, 26], arr2d[1, 26]],# au7
                                          [arr2d[0, 43], arr2d[1, 43]], [arr2d[0, 45], arr2d[1, 45]],# au10
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],# au12 au15
                                          [arr2d[0, 31], arr2d[1, 31]], [arr2d[0, 37], arr2d[1, 37]],
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]],# au23 au24 au25
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]],
                                          [arr2d[0, 34], arr2d[1, 34]], [arr2d[0, 40], arr2d[1, 40]],
                                          [arr2d[0, 39], arr2d[1, 39] + ruler / 2], [arr2d[0, 41], arr2d[1, 41] + ruler / 2]# au26
            ])

    return coord

def render_gaussian_heatmap(coord, crop_size, num_AU_points, shrink = 2, sigma = 2, use_gpu=True):
    output_shape = (crop_size // shrink, crop_size // shrink)
    x = torch.tensor([i for i in range(output_shape[1])])
    y = torch.tensor([i for i in range(output_shape[0])])
    coord = torch.tensor(coord)
    if use_gpu:
        x, y, coord = x.cuda(), y.cuda(), coord.cuda()

    xx, yy = torch.meshgrid(x, y)
    xx, yy = xx.T, yy.T
    xx = torch.reshape(xx.float(), (1, *output_shape, 1))
    yy = torch.reshape(yy.float(), (1, *output_shape, 1))
    x = torch.floor(torch.reshape(coord[:, :, 0], [-1, 1, 1, num_AU_points]) / crop_size * output_shape[1] + 0.5)
    y = torch.floor(torch.reshape(coord[:, :, 1], [-1, 1, 1, num_AU_points]) / crop_size * output_shape[0] + 0.5)
    heatmap = torch.exp(
        -(((xx - x) / float(sigma)) ** 2) / float(2) - (((yy - y) / float(sigma)) ** 2) / float(
            2))
    # print(heatmap.max()) # is 1
    # return heatmap * 255.
    return heatmap


#----------------Borrow from ResTv2--------------------- 
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model_without_ddp.load_state_dict(state_dict)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str):  # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, dim, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.dim = (dim,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.dim, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
