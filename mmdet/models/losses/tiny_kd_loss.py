import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.utils.weight_init import normal_init, constant_init
from mmcv.runner import BaseModule

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l2_loss(pred, target):
    """L2 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.pow(pred - target, 2)
    return loss

@torch.no_grad()
def get_mask(gt_bboxes, img_metas, feat_shape):
    img_h, img_w = 0, 0
    for img_meta in img_metas:
        h, w, _ = img_meta['pad_shape']
        img_h = max(h, img_h)
        img_w = max(w, img_w)

    N, C, H, W = feat_shape

    mask_fg = torch.zeros(N, img_h, img_w, dtype=torch.float, device=gt_bboxes[0].device, requires_grad=False)

    wmin, wmax, hmin, hmax = [], [], [], []
    for i, (boxes, img_meta) in enumerate(zip(gt_bboxes, img_metas)):
        wmin.append(torch.floor(boxes[:, 0]).int())
        wmax.append(torch.ceil(boxes[:, 2]).int())
        hmin.append(torch.floor(boxes[:, 1]).int())
        hmax.append(torch.ceil(boxes[:, 3]).int())

        area = img_h * img_w / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))
        for j in range(len(boxes)):
            mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                torch.maximum(mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

        mask_fg[i] = mask_fg[i] / len(boxes)

    mask_fg = mask_fg.unsqueeze(dim=1)
    max_pool = nn.AdaptiveMaxPool2d((H, W))
    mask_fg = max_pool(mask_fg)
    mask_fg = mask_fg / torch.sum(mask_fg, dim=(1, 2, 3), keepdim=True)

    mask_bg = torch.where(mask_fg > 0, 0., 1.)
    for i in range(N):
        if torch.sum(mask_bg[i]):
            mask_bg[i] /= torch.sum(mask_bg[i])

    mask_fg = mask_fg / (N * C)
    mask_bg = mask_bg / (N * C)
    return mask_fg, mask_bg


# REF: https://github.com/open-mmlab/mmcv
class Attention(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,  # TODO
                 mode=['value', 'conv_out', 'identity'],
                 temp=None,
                 info_prefix='',
                 **kwargs):
        super(Attention, self).__init__()

        self.in_channels = in_channels
        self.dim = max(in_channels // 1, 1)  # reduction = 1

        self.mode = mode
        self.info_prefix = info_prefix

        if temp is None:
            temp = self.dim ** 0.5
        self.temp = float(temp)

        self.query = ConvModule(
            self.in_channels,
            self.dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None
        )
        self.key = ConvModule(
            self.in_channels,
            self.dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None
        )

        if 'value' in self.mode:
            self.value = ConvModule(
                self.in_channels,
                self.dim,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None
            )
        else:
            self.value = None

        if 'conv_out' in self.mode:
            self.conv_out = ConvModule(
                self.dim,
                self.in_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
        else:
            self.conv_out = None

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.value, self.query, self.key]:
            if m is not None:
                normal_init(m.conv, std=std)

        if self.conv_out is not None:
            if zeros_init:
                if self.conv_out.norm_cfg is None:
                    constant_init(self.conv_out.conv, 0)
                else:
                    constant_init(self.conv_out.norm, 0)
            else:
                if self.conv_out.norm_cfg is None:
                    normal_init(self.conv_out.conv, std=std)
                else:
                    normal_init(self.conv_out.norm, std=std)

    def forward(self, x, mask=None):
        # x: [N, C, H, W]
        n = x.size(0)

        # query: [N, HxW, C]
        # key:   [N, C, HxW]
        # value: [N, HxW, C]
        query = self.query(x).view(n, self.dim, -1).permute(0, 2, 1)
        key = self.key(x).view(n, self.dim, -1)

        # attention: [N, HxW, HxW]
        attn = torch.matmul(query, key)
        attn /= self.temp
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)

        value = self.value(x).view(n, self.dim, -1).permute(0, 2, 1)
        # y: [N, HxW, C]
        y = torch.matmul(attn, value)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.dim, *x.size()[2:])

        if 'conv_out' in self.mode:
            y = self.conv_out(y)

        if 'identity' in self.mode:
            output = x + y
        else:
            output = y
        return output


class PatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16, stride=None,
                 padding=0, shift_aug=False, **kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.padding = padding
        self.stride = patch_size if stride is None else stride

        self.unfold_params = dict(kernel_size=self.patch_size, padding=self.padding, stride=self.stride)
        self.fold_params = dict(kernel_size=self.patch_size, padding=self.padding, stride=self.stride)

        self.attention = Attention(in_channels=in_channels, **kwargs)

        self.shift_aug = shift_aug

    @staticmethod
    def get_pad_value(length, kernel_size, stride):
        if length + stride <= kernel_size:
            pad = kernel_size - length
        else:
            pad = (length - kernel_size) % stride
            pad = stride - pad if pad != 0 else 0
        return pad

    def pad_input(self, x, pad=None):
        B, C, H, W = x.shape

        if pad is None:
            pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0

            if self.shift_aug:
                pad_t = torch.randint(self.patch_size, [1]) if H > self.patch_size else 0
                pad_l = torch.randint(self.patch_size, [1]) if W > self.patch_size else 0

            pad_b = self.get_pad_value(H + pad_t, self.patch_size, self.stride)
            pad_r = self.get_pad_value(W + pad_l, self.patch_size, self.stride)

            pad = [pad_l, pad_r, pad_t, pad_b]  # left, right, top, bottom
        else:
            pad_l, pad_r, pad_t, pad_b = pad

        x = F.pad(x, pad, mode='constant', value=0)
        out_shape = (pad_t, pad_t + H, pad_l, pad_l + W)
        return x, pad, out_shape

    def get_scalar(self, x, out_shape):
        H, W = x.shape[2:]
        x = nn.Unfold(**self.unfold_params)(x)
        x = nn.Fold((H, W), **self.fold_params)(x)
        x = x[:, :, out_shape[0]:out_shape[1], out_shape[2]:out_shape[3]]
        return x

    def get_attention_mask(self, in_shape, pad):
        B, _, H, W = in_shape
        S = self.patch_size

        mask = torch.zeros(B, 1, H, W).float()
        mask = F.pad(mask, pad, mode='constant', value=-float('inf'))
        mask = nn.Unfold(**self.unfold_params)(mask)

        N = mask.size(-1)
        mask = mask.view(B, 1, S, S, N).permute(0, 4, 1, 2, 3)
        mask = mask.contiguous().view(B * N, 1, S, S)
        mask = mask.view(B * N, 1, -1)

        return mask

    def forward(self, x, pad=None):
        in_shape = x.shape
        x, pad, out_shape = self.pad_input(x, pad)
        B, C, H, W = x.shape
        S = self.patch_size

        attn_mask = self.get_attention_mask(in_shape, pad).to(x.device)
        scalar = self.get_scalar(torch.ones_like(x, dtype=torch.float), out_shape)

        x = nn.Unfold(**self.unfold_params)(x)
        N = x.size(-1)

        x = x.view(B, C, S, S, N).permute(0, 4, 1, 2, 3)
        x = x.contiguous().view(B * N, C, S, S)

        x = self.attention(x, mask=attn_mask)

        x = x.view(B, N, C * S * S).permute(0, 2, 1)
        x = nn.Fold((H, W), **self.fold_params)(x)

        x = x[:, :, out_shape[0]:out_shape[1], out_shape[2]:out_shape[3]]
        x = x / scalar
        return x, pad


@LOSSES.register_module()
class KDLoss(BaseModule):
    def __init__(self,
                 stu_channel=256,
                 tea_channel=256,
                 alpha_fg=1.0,
                 alpha_bg=0.5,
                 lambda1=1.0,
                 lambda2=1.0,
                 loss_name='loss_kd',
                 attn_cfg=dict(),
                 ):
        super().__init__()
        self.loss_name = loss_name

        self.alpha_fg = alpha_fg
        self.alpha_bg = alpha_bg
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.tea_attention = PatchAttention(tea_channel, **attn_cfg)
        self.stu_attention = PatchAttention(stu_channel, **attn_cfg)

        self.adapter1 = nn.Conv2d(stu_channel, tea_channel, kernel_size=3, stride=1, padding=1)
        self.adapter2 = nn.Conv2d(stu_channel, tea_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, stu_x, tea_x, warmup_weight=1.0, tea_img_metas=None, tea_gt_bboxes=None, **kwargs):
        if stu_x.shape != tea_x.shape:
            stu_x = F.interpolate(stu_x, size=tea_x.shape[2:], mode='bilinear', align_corners=False)

        with torch.no_grad():
            fg_mask, bg_mask = get_mask(tea_gt_bboxes, tea_img_metas, feat_shape=tea_x.shape)

        tea_a, pad = self.tea_attention(tea_x)
        stu_a, _ = self.stu_attention(stu_x, pad=pad)

        loss_fea = self.get_loss(self.adapter1(stu_x), tea_x, fg_mask, bg_mask)
        loss_rel = self.get_loss(self.adapter2(stu_a), tea_a, fg_mask, bg_mask)
        loss = self.lambda1 * loss_fea + self.lambda2 * loss_rel
        loss = loss * warmup_weight
        return {self.loss_name: loss}

    def get_loss(self, stu_x, tea_x, fg_mask, bg_mask):
        loss = l2_loss(stu_x, tea_x, reduction='none')
        loss = self.alpha_fg * torch.sum(loss * fg_mask) + self.alpha_bg * torch.sum(loss * bg_mask)
        return loss
