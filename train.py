import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm

from loaders.domainnet import build_dataset
from loaders.office_home import build_dataset_officehome
from loaders.visda import build_dataset_visda
from utils.utils import set_seed, weights_init, print_options, AllMeters
from utils.lr_schedule import InvLr
from utils.ema import ModelEMA
from model.basenet import Predictor_deep
from model.resnet import resnet34
from utils.losses import Prototype_t, loss_unl

from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser(description='ProML')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--bs', default=24, type=int)
    parser.add_argument('--bs_unl_multi', default=2, type=int)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--source', type=str, default='painting')
    parser.add_argument('--target', type=str, default='real')
    parser.add_argument('--seed', type=int, default=123456, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--arch', type=str, default='resnet',
                        help='which network to use')
    parser.add_argument('--data_root', type=str, default='/data/domainnet')
    parser.add_argument('--unl_transform', type=str, default='fixmatch')
    parser.add_argument('--labeled_transform', type=str, default='labeled')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')
    parser.add_argument('--num_classes', type=int, default=126)
    parser.add_argument('--dataset', type=str, default='multi',
                        choices=['multi', 'office', 'office_home', 'visda'],
                        help='the name of dataset')
    parser.add_argument('--base_path', type=str, default='./data/txt/multi/')
    parser.add_argument('--threshold1', default=0.95, type=float,
                        help='pseudo label threshold1')
    parser.add_argument('--threshold2', default=0.4, type=float,
                        help='pseudo label threshold2')
    parser.add_argument('--T', type=float, default=0.1, metavar='T',
                        help='temperature (default: 0.1)')
    parser.add_argument('--log_dir', type=str, default='./logs/domainnet/contras_cls/debug')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--test_interval', type=float, default=500)
    parser.add_argument('--print_interval', type=float, default=50)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--warm_steps', type=int, default=100)
    
    args = parser.parse_args()
    return args


def train(source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, net_G, net_F, optimizer, scheduler):
    global args, writer
    net_G.train()
    net_F.train()

    loss_name_list = ['L_base', 'L_batch', 'L_fixmatch', 'L_inter', 'L_intra']
    train_meters = AllMeters(loss_name_list)
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    target_unl_iter = iter(target_loader_unl)

    criterion = nn.CrossEntropyLoss().cuda()
    best_acc_val, cur_acc_test = 0, 0
    best_acc = 0

    net_G_ema = ModelEMA(net_G, decay=0.99)
    net_F_ema = ModelEMA(net_F, decay=0.99)

    proto_t = Prototype_t(C=args.num_classes, dim=args.inc)

    for batch_idx in range(args.num_steps):
        lambda_warm = 1 if batch_idx > args.warm_steps else 0
        try:
            data_batch_source = source_iter.next()
        except:
            source_iter = iter(source_loader)
            data_batch_source = source_iter.next()

        try:
            data_batch_target = target_iter.next()
        except:
            target_loader.dataset.shuffle_repeat()
            target_iter = iter(target_loader)
            data_batch_target = target_iter.next()

        try:
            data_batch_unl = target_unl_iter.next()
        except:
            target_unl_iter = iter(target_loader_unl)
            data_batch_unl = target_unl_iter.next()

        imgs_s_w = data_batch_source[0].cuda()
        gt_s = data_batch_source[1].cuda()

        imgs_t_w = data_batch_target[0].cuda()
        gt_t = data_batch_target[1].cuda()

        imgs_tu_w, imgs_tu_s = data_batch_unl[0][0].cuda(), data_batch_unl[0][1].cuda()
        gt_tu = data_batch_unl[1].cuda()

        data = torch.cat((imgs_s_w, imgs_t_w), 0)
        target = torch.cat((gt_s, gt_t), 0)
        output = net_G(data)

        feat_t_con, out1 = net_F(output)
        feat_s, feat_t = feat_t_con.chunk(2)
        L_base = criterion(out1, target)
        
        L_intra, L_inter, L_fixmatch, L_batch, unl_mask, unl_pseudo_label, feat_tu_w = loss_unl(net_G, net_F, feat_t_con, feat_s, gt_s, imgs_tu_w, imgs_tu_s, proto_t, batch_idx, args)

        if batch_idx <= args.warm_steps:
            Loss = L_base + L_batch + L_fixmatch
        else:
            Loss = L_base + L_batch + L_fixmatch + L_inter + L_intra
        
        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(batch_idx)
        net_G_ema.update(net_G)
        net_F_ema.update(net_F)

        proto_t.update(feat_t, gt_t, feat_tu_w, batch_idx, unl_mask, unl_pseudo_label, args, norm=True)
        
        loss_name_list = ['L_base', 'L_batch', 'L_fixmatch', 'L_inter', 'L_intra']
        loss_value_list = [L_base.item(), L_batch.item(), L_fixmatch.item(), L_inter.item(), L_intra.item()]
        train_meters.update_list(loss_name_list, loss_value_list)

        if batch_idx % args.print_interval == 0:
            log_str = train_meters.log_str(batch_idx, args)
            print(log_str)

        if (batch_idx + 1) % args.test_interval == 0:
            train_meters.tb_log(writer, batch_idx)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], batch_idx)

            acc_val = test(target_loader_val, net_G_ema.ema, net_F_ema.ema, batch_idx)
            acc1, acc2,  per_acc1, per_acc2 = test_multi(batch_idx, target_loader_test,
                                                         net_G, net_F,
                                                         net_G_ema.ema, net_F_ema.ema)
            
            writer.add_scalar('Test/acc', acc1, batch_idx)
            writer.add_scalar('Test/acc_ema', acc2, batch_idx)
            writer.add_scalar('Test/mAcc', per_acc1, batch_idx)
            writer.add_scalar('Test/mAcc_ema', per_acc2, batch_idx)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                cur_acc_test = acc1
                cur_acc_test_ema = acc2
                if args.save:
                    net_F_path = os.path.join(args.log_dir, 'ckpt', 'Best_F.pth')
                    net_F_ema_path = os.path.join(args.log_dir, 'ckpt', 'Best_F_ema.pth')
                    net_G_path = os.path.join(args.log_dir, 'ckpt', 'Best_G.pth')
                    net_G_ema_path = os.path.join(args.log_dir, 'ckpt', 'Best_G_ema.pth')
                    torch.save(net_G.state_dict(), net_G_path)
                    torch.save(net_F.state_dict(), net_F_path)
                    torch.save(net_G_ema.ema.state_dict(), net_G_ema_path)
                    torch.save(net_F_ema.ema.state_dict(), net_F_ema_path)
            
            writer.add_scalar('Test/BestAcc', cur_acc_test, batch_idx)
            writer.add_scalar('Test/BestAcc_ema', cur_acc_test_ema, batch_idx)

            train_meters.reset()


@torch.no_grad()
def test(test_loader, net_G, net_F, iter_idx):
    global args, writer
    net_G.eval()
    net_F.eval()

    correct = 0
    total = 0
    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
        _, outputs = net_F(net_G(inputs))

        outputs = torch.softmax(outputs, dim=1)
        max_prob, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    net_G.train()
    net_F.train()
    acc = correct / total * 100.
    return acc


def test_multi(batch_idx1, test_loader, net_G1, net_F1, net_G2, net_F2):
    global args, writer
    net_G1.eval()
    net_F1.eval()
    net_G2.eval()
    net_F2.eval()

    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0
    
    total = 0

    predicted_list1 = []
    predicted_list2 = []
    predicted_list = []
    target_list = []
    
    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        try:
            inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
        except:
            inputs, targets = data_batch[0][0].cuda(), data_batch[1].cuda()
        _, outputs1 = net_F1(net_G1(inputs))
        _, outputs2 = net_F2(net_G2(inputs))
        
        l = 0.5
        outputs1 = torch.softmax(outputs1, dim=1)
        outputs2 = torch.softmax(outputs2, dim=1)
        
        max_prob, predicted1 = outputs1.max(1)
        max_prob, predicted2 = outputs2.max(1)
        total += targets.size(0)

        correct1 += predicted1.eq(targets).sum().item()
        correct2 += predicted2.eq(targets).sum().item()

        predicted_list1 += predicted1.cpu().numpy().tolist()
        predicted_list2 += predicted2.cpu().numpy().tolist()
        target_list += targets.cpu().numpy().tolist()

    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(target_list, predicted_list1, normalize='true')
    cm2 = confusion_matrix(target_list, predicted_list2, normalize='true')

    per_acc1 = cm1.diagonal().mean() * 100.
    per_acc2 = cm2.diagonal().mean() * 100.

    net_G1.train()
    net_F1.train()
    net_G2.train()
    net_F2.train()
    acc1 = correct1 / total * 100.
    acc2 = correct2 / total * 100.
    print('acc1 %.2f acc2 %.2f ' % (acc1, acc2))
    print('mAcc1 %.2f mAcc2 %.2f ' % (per_acc1, per_acc2))

    return acc1, acc2,  per_acc1, per_acc2


def get_optim_params(model, lr):
    lr_list, lr_multi_list = [], []
    for name, param in model.named_parameters():
        if 'fc' in name:
            lr_multi_list.append(param)
        else:
            lr_list.append(param)
    return [{'params': lr_list, 'lr': lr},
            {'params': lr_multi_list, 'lr': 10 * lr}]


def main():
    global args, writer
    args = get_args()

    print_options(args)

    if args.dataset == 'multi':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset(args)
    elif args.dataset == 'office_home':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset_officehome(args)
    elif args.dataset == 'visda':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset_visda(args)

    set_seed(args.seed)

    net_G = resnet34().cuda()
    net_F = Predictor_deep(num_class=args.num_classes, inc=512, temp=args.T)
    weights_init(net_F)
    net_F.cuda()
    args.inc = 512

    optimizer = optim.SGD(get_optim_params(net_G, args.lr) + net_F.get_optim_params(args.lr),
                          momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = InvLr(optimizer)

    tb_dir = os.path.join(args.log_dir, 'tb_log')
    save_dir = os.path.join(args.log_dir, 'ckpt')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    train(source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,
          net_G, net_F, optimizer, scheduler)


if __name__ == '__main__':
    main()
