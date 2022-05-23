import torch.nn as nn
import torch.nn.functional as F
import torch


def calculate_attentive_distillation_loss(f_map_s, f_map_t):
    """
    Args:
        f_map_s(Tensor): Bs*C*H*W, student's feature map
        f_map_t(Tensor): Bs*C*H*W, teacher's feature map
    """
    temp = 0.5

    S_attention_t, C_attention_t = get_attention(f_map_s, temp)
    S_attention_s, C_attention_s = get_attention(f_map_t, temp)

    loss_att = get_loss(f_map_s, f_map_t, C_attention_t, S_attention_t)
    loss_ad = get_ad_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
    combined_loss = loss_att + loss_ad
    return combined_loss


def get_attention(f_map, temp):
    """ preds: Bs*C*W*H """
    N, C, H, W = f_map.shape

    value = torch.abs(f_map)
    # Bs*W*H
    fea_map = value.mean(axis=1, keepdim=True)
    S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

    # Bs*C
    channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
    C_attention = C * F.softmax(channel_map / temp, dim=1)

    return S_attention, C_attention


def get_loss(f_map_s, f_map_t, C_t, S_t):
    loss_mse = nn.MSELoss(reduction='mean')

    C_t = C_t.unsqueeze(dim=-1)
    C_t = C_t.unsqueeze(dim=-1)

    S_t = S_t.unsqueeze(dim=1)

    fea_t = torch.mul(f_map_t, torch.sqrt(S_t))
    fea_t = torch.mul(fea_t, torch.sqrt(C_t))

    fea_s = torch.mul(f_map_s, torch.sqrt(S_t))
    fea_s = torch.mul(fea_s, torch.sqrt(C_t))

    loss = loss_mse(fea_s, fea_t)
    return loss


def get_ad_loss(C_s, C_t, S_s, S_t):
    loss_mse = nn.L1Loss(reduction='mean')
    ad_loss = loss_mse(C_s, C_t) + loss_mse(S_s, S_t)

    return ad_loss

    return context
