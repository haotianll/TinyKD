import copy
from collections import OrderedDict

import mmcv
import torch
import torch.nn as nn
from mmcv.runner import ModuleDict, ModuleList
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict

from .single_stage import SingleStageDetector
from ..builder import DETECTORS, build_detector, build_loss


@DETECTORS.register_module()
class KD_SingleStage(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,

                 kd_warmup_iters=1000,
                 kd_config=None,
                 teacher_config=None,
                 teacher_ckpt=None,
                 teacher_inherit=[],
                 **kwargs):
        super().__init__(backbone, neck, bbox_head, **kwargs)

        self.kd_config = kd_config

        self.teacher_ckpt = teacher_ckpt
        self.teacher_config = teacher_config
        self.teacher_inherit = teacher_inherit

        self.teacher_model = self.init_teacher_model(teacher_config, teacher_ckpt)

        self.kd_iter = 0
        self.kd_warmup_iters = kd_warmup_iters

        if self.kd_config is not None:
            self.kd_losses, self.kd_positions, self.kd_indices = self.init_kd_losses()

    def init_kd_losses(self):
        loss_modules = ModuleDict()
        loss_positions = dict()
        loss_kd_indices = dict()

        for config in self.kd_config:
            name = config.get('name', 'kd')
            loss_kd = config.get('loss_kd')
            loss_stages = config.get('loss_stages')
            kd_indices = config.get('kd_indices')
            loss_name = config.get('loss_name')
            loss_position = config.get('position', 'neck')

            loss = ModuleList()
            for index in range(loss_stages):
                if index not in kd_indices:
                    loss.append(nn.Identity())
                    continue

                loss_config = copy.deepcopy(loss_kd)

                extra_config = loss_config.pop('extra')
                for k, v in extra_config.items():
                    extra_config[k] = v[index]
                loss_config.update(extra_config)

                loss_config['loss_name'] = loss_name.format(index)
                loss.append(build_loss(loss_config))
            loss_modules.add_module(name, loss)
            loss_positions[name] = loss_position
            loss_kd_indices[name] = kd_indices

        return loss_modules, loss_positions, loss_kd_indices

    def init_teacher_model(self, teacher_config, teacher_ckpt):
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)

        teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(teacher_model, teacher_ckpt, map_location='cpu')

        return teacher_model

    def init_weights(self):
        super().init_weights()

        if self.teacher_inherit is None or len(self.teacher_inherit) == 0:
            return

        tea_checkpoint = _load_checkpoint(self.teacher_ckpt)
        all_weights = []
        for name, weight in tea_checkpoint["state_dict"].items():
            for key in self.teacher_inherit:
                if name.startswith(key):
                    all_weights.append((name, weight))
        state_dict = OrderedDict(all_weights)
        load_state_dict(self, state_dict)

    def train(self, mode=True):
        self.teacher_model.train(False)
        super().train(mode)

    def cuda(self, device=None):
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def __setattr__(self, name, value):
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def preprocess_input(img, img_metas=None, gt_bboxes=None):
        new_img = copy.deepcopy(img)
        new_img_metas = copy.deepcopy(img_metas)
        new_gt_bboxes = copy.deepcopy(gt_bboxes)
        return new_img, new_img_metas, new_gt_bboxes

    def forward_extract_feat(self, img, backbone, neck):
        x = backbone(img)
        x = neck(x)
        return x

    def forward_kd(self, x, teacher_x, **kwargs):
        self.kd_iter += 1
        kwargs.update(dict(warmup_weight=min(1., self.kd_iter / self.kd_warmup_iters)))

        losses = dict()
        for key, loss_modules in self.kd_losses.items():
            position = self.kd_positions.get(key)
            kd_indices = self.kd_indices.get(key)
            if position == 'neck':
                for index, (stu_x, tea_x) in enumerate(zip(x, teacher_x)):
                    if index not in kd_indices:
                        continue
                    loss_func = loss_modules[index]
                    loss = loss_func(stu_x, tea_x.detach(), **kwargs)
                    losses.update(loss)
        return losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        tea_img, tea_img_metas, tea_gt_bboxes = self.preprocess_input(img, img_metas, gt_bboxes)

        x = self.forward_extract_feat(img, self.backbone, self.neck)

        with torch.no_grad():
            teacher_x = self.forward_extract_feat(tea_img, self.teacher_model.backbone, self.teacher_model.neck)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

        if self.kd_config:
            kd_losses = self.forward_kd(x, teacher_x, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels,
                                        tea_img_metas=tea_img_metas, tea_gt_bboxes=tea_gt_bboxes)
            losses.update(kd_losses)
        return losses
