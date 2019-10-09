"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model.fc import FCNet
from model.bc import BCNet
import numpy as np


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2, .5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse,
                                        dropout=dropout, k=3),
                                  name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False,
                    mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        # if visualize:
        #     logits,bc_out = self.logits(v,q) # b x g x v x q
        # else:
        #     logits = self.logits(v,q) # b x g x v x q

        logits = self.logits(v, q)  # b x g x v x q
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(
                                                                logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        p = nn.functional.softmax(
            logits.view(-1, self.glimpse, v_num * q_num), 2)
        p = p.view(-1, self.glimpse, v_num, q_num)
        # if visualize:
        #     return p,logits, bc_out
        if not logit:
            return p, logits
        return logits
