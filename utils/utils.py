import os
import torch
import torch.nn as nn
import os.path as osp
import numpy as np
import random
import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AllMeters:
    '''
    update and log a list of AverageMeter objects
    '''

    def __init__(self, name_list) -> None:
        self.meter_dict = {}
        self.name_list = name_list
        for name in name_list:
            self.meter_dict[name] = AverageMeter()
        self.last_time = time.time()

    def add(self, name_list):
        for name in name_list:
            if name in self.meter_dict:
                print('Already in!')
                raise NameError
            self.meter_dict[name] = AverageMeter()
            self.name_list.append(name)

    def update(self, k, v, n=1):
        self.meter_dict[k].update(v, n)

    def update_list(self, k_list, v_list):
        for k, v in zip(k_list, v_list):
            self.meter_dict[k].update(v)

    def get(self, k):
        return self.meter_dict[k].avg

    def reset(self):
        self.meter_dict = {}
        for name in self.name_list:
            self.meter_dict[name] = AverageMeter()

    def tb_log(self, writer, step, prefix='Train'):
        for name in self.name_list:
            tb_name = prefix + '/' + name
            v = self.get(name)
            writer.add_scalar(tb_name, v, step)

    def log_str(self, step, args):
        s = args.source.upper()[0]
        t = args.target.upper()[0]
        cur_time = time.time()
        log = '%s -> %s Iter %d Time %.2f ' % (s, t, step, cur_time - self.last_time)
        for k in self.name_list:
            v = self.get(k)
            log += '%s %.4f ' % (k, v)
        self.last_time = time.time()
        return log


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_options(args):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    os.makedirs(args.log_dir, exist_ok=True)
    file_name = osp.join(args.log_dir, 'opt.txt')
    with open(file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
