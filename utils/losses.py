import numpy as np
import torch
import torch.nn.functional as F
import ot
import torch.nn as nn
import scipy
from torch.autograd import Variable
import math

def loss_unl(net_G, net_F, feat_t_con, feat_s, gt_s, imgs_tu_w, imgs_tu_s, proto_t, batch_idx, args):
    feat_tu = net_G(torch.cat((imgs_tu_w, imgs_tu_s), dim=0))
    feat_tu_con, logits_tu = net_F(feat_tu)
    feat_tu_w, feat_tu_s = feat_tu_con.chunk(2)
    logits_tu_w, logits_tu_s = logits_tu.chunk(2)
    prob_tu_w = F.softmax(logits_tu_w, dim=1)
    prob_tu_s = F.softmax(logits_tu_s, dim=1)
    prob_tu_ws = sharpen(prob_tu_w)
    prob_tu_ss = sharpen(prob_tu_s)
    
    cos_wt = F.softmax(cosine_similarity(feat_tu_w, proto_t.mo_pro) / args.T, dim=1)
    cos_st = F.softmax(cosine_similarity(feat_tu_s, proto_t.mo_pro) / args.T, dim=1)

    L_intra, pseudo_label_ot = ot_loss(proto_t, feat_tu_w, feat_tu_s, args)

    pseu_tu_w = F.softmax(logits_tu_w.detach(), dim=1)
    max_probs, targets_u = torch.max(pseu_tu_w, dim=1)
    unl_mask1 = max_probs.ge(args.threshold1)
    unl_mask2 = max_probs.ge(args.threshold2)
    pseudo_label_ot[unl_mask1] = targets_u[unl_mask1]

    L_fixmatch = 1. * (F.cross_entropy(logits_tu_s, unl_pseudo_label, reduction='none') * unl_mask1).mean()

    if batch_idx > args.warm_steps:
        unl_pseudo_label = pseudo_label_ot
        unl_mask = unl_mask2.float().unsqueeze(1).detach()
        L_batch = 0.2 * (contras_cls(prob_tu_ws, prob_tu_ss) + contras_cls(cos_wt, cos_st))
        L_inter = loss_align(feat_s, gt_s, proto_t, args)
    else:
        unl_pseudo_label = targets_u
        unl_mask = unl_mask1.float().unsqueeze(1).detach()
        L_batch = 0.2 * contras_cls(prob_tu_ws, prob_tu_ss)
        L_inter = 0.
    return L_intra, L_inter, L_fixmatch, L_batch, unl_mask, unl_pseudo_label, feat_tu_w


def loss_align(feat_s, gt_s, proto_t, args):
    dist_st = cosine_similarity(feat_s, proto_t.mo_pro) / args.T
    L_align = (F.cross_entropy(dist_st, gt_s)).mean()
    return L_align


def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())
    return similarity
        
        
def sharpen(p, T = 0.25):
    u = p ** (1/T)
    return u / u.sum(dim=1, keepdim=True)


def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2
    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss


def ot_loss(proto_t, feat_tu_w, feat_tu_s, args):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_t.mo_pro, feat_tu_w)
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    pseudo_label_ot = pred_ot.clone().detach()
    Lm = center_loss_cls(proto_t.mo_pro, feat_tu_s, pred_ot, num_classes=args.num_classes)
    return Lm, pseudo_label_ot


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def ot_mapping(M):
    reg1 = 1
    reg2 = 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


def center_loss_cls(centers, x, labels, num_classes=65):
    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm2 = F.normalize(centers)
    x = F.normalize(x)
    distmat =  - 1. * x @ centers_norm.t() + 1
    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))
    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    return loss


class Prototype_t:
    def __init__(self, C=65, dim=512, m=0.9):
        self.mo_pro = torch.zeros(C, dim).cuda()
        self.batch_pro = torch.zeros(C, dim).cuda()
        self.m = m

    @torch.no_grad()
    def update(self, featt, lblt, feat_tu_w, i_iter, unl_mask, unl_pseudo_label, args, norm=False):
        if i_iter < 20:
            momentum = 0
        else:
            momentum = self.m

        if i_iter <= (20 + args.warm_steps)/2:
            for i_cls in torch.unique(lblt):
                featt_i = featt[lblt == i_cls, :]
                featt_i_center = featt_i.mean(dim=0, keepdim=True)
                self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * momentum + featt_i_center * (1 - momentum)

        #unlabel update 
        if i_iter > (20 + args.warm_steps)/2:
            unl_pseudo_label1 = torch.cat([lblt, unl_pseudo_label], dim=0)
            feat_tu_w_for_pro1 = torch.cat([featt, feat_tu_w], dim=0)
            unl_mask1 = torch.cat([torch.ones(lblt.shape[0], 1).cuda(), unl_mask], dim=0)
            for i_cls in torch.unique(unl_pseudo_label1):
                featt_i = feat_tu_w_for_pro1[unl_pseudo_label1 == i_cls * unl_mask1.squeeze(), :]
                if featt_i.shape[0]:
                    featt_i_center = featt_i.mean(dim=0, keepdim=True)
                    self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * momentum + featt_i_center * (1 - momentum)
            
        if norm:
            self.mo_pro = F.normalize(self.mo_pro)