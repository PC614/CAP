
"""
方法R&P的代码，参考一下几个链接
通过随机padding和平滑的方法来进行防御
属于input transforamtion的一种

"""

import torch
import torch.nn.functional as F

def randomize_input(xs, prob=0.5, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
    p = torch.rand(1).item()
    if p <= prob:
        out = random_resize_pad(xs, crop_lst)
        return out
    else:
        return xs

def random_resize_pad(xs, crop_lst):
    rand_cur = torch.randint(low=0, high=len(crop_lst), size=(1,)).item()
    crop_size = 1 - crop_lst[rand_cur]
    pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
    pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

    if len(xs.shape) == 4:
        bs, c, w, h = xs.shape
    elif len(xs.shape) == 5:
        bs, fs, c, w, h = xs.shape
    w_, h_ = int(crop_size * w), int(crop_size * h)
    out = F.interpolate(xs, size=[w_, h_], mode='bicubic', align_corners=False)

    pad_left = int(pad_left * (w - w_))
    pad_top = int(pad_top * (h - h_))
    out = F.pad(out, [pad_left, w - pad_left - w_, pad_top, h - pad_top - h_], value=0)

    return out