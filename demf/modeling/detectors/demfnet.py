import torch
import warnings

from mmdet.models import DETECTORS
from mmdet.core import bbox2result
from mmdet3d.models import (
    ImVoteNet, Base3DDetector, builder
)
from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class DeMFVoteNet(ImVoteNet):
    def __init__(self,
                 pts_backbone=None,
                 pts_bbox_head=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 img_encoder=None,
                 freeze_img_branch=False,
                 num_sampled_seed=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):

        Base3DDetector.__init__(self, init_cfg=init_cfg)

        # point branch
        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head is not None:
            pts_bbox_head.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head.update(test_cfg=test_cfg.pts)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        # image branch
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_encoder is not None:
            self.img_encoder = builder.build_head(img_encoder)
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg

        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()

        self.num_sampled_seed = num_sampled_seed

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)

        if self.with_pts_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=pts_pretrained)
                
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load img network ckpts into img branch."""
        for key in list(state_dict):
            if not key.startswith('img_bbox_head'):
                continue

            if 'encoder' in key or 'level_embeds' in key:
                key_ = key.replace('img_bbox_head.transformer',
                                   'img_encoder')
                state_dict[key_] = state_dict.pop(key)
            else:
                state_dict.pop(key)
        
        super(ImVoteNet, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
    
    def train(self, mode=True):
        """Overload in order to keep image branch modules in eval mode."""
        super(ImVoteNet, self).train(mode)
        if self.freeze_img_branch:
            self.img_encoder.eval()
            if self.with_img_backbone:
                self.img_backbone.eval()
            if self.with_img_neck:
                self.img_neck.eval()
    
    @torch.no_grad()
    def extract_img_feat(self, img, img_metas):
        """Directly extract features from the img backbone+neck."""
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        if self.img_encoder:
            x = self.img_encoder(x, img_metas)
        return x
    
    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes_ignore=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      **kwargs):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        img_features = self.extract_img_feat(img, img_metas)

        points = torch.stack(points)
        seeds_3d, seed_3d_features, seed_indices = \
            self.extract_pts_feat(points)

        feat_dict = dict(
            seed_points=seeds_3d,
            seed_features=seed_3d_features,
            seed_indices=seed_indices)
        img_dict = dict(
            img_features=img_features,
            img_metas=img_metas,
        )

        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                        pts_semantic_mask, pts_instance_mask, img_metas)
        bbox_preds = self.pts_bbox_head(
            feat_dict, self.train_cfg.pts.sample_mod, img_dict)
        losses = self.pts_bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None,
                     bboxes_2d=None,
                     **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for _img, img_meta in zip(img, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(_img.size()[-2:])
        
        if points is None:
            for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(
                        f'{name} must be a list, but got {type(var)}')

            num_augs = len(img)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(img)}) '
                                 f'!= num of image meta ({len(img_metas)})')

            if num_augs == 1:
                # proposals (List[List[Tensor]]): the outer list indicates
                # test-time augs (multiscale, flip, etc.) and the inner list
                # indicates images in a batch.
                # The Tensor should have a shape Px4, where P is the number of
                # proposals.
                if 'proposals' in kwargs:
                    kwargs['proposals'] = kwargs['proposals'][0]
                return self.simple_test_img_only(
                    img=img[0], img_metas=img_metas[0], **kwargs)
            else:
                # TODO: not implement yet
                assert img[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{img[0].size(0)}'
                # TODO: support test augmentation for predefined proposals
                assert 'proposals' not in kwargs
                return self.aug_test_img_only(
                    img=img, img_metas=img_metas, **kwargs)

        else:
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))

            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.
                    format(len(points), len(img_metas)))

            if num_augs == 1:
                return self.simple_test(
                    points[0],
                    img_metas[0],
                    img[0],
                    bboxes_2d=bboxes_2d[0] if bboxes_2d is not None else None,
                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img, bboxes_2d,
                                     **kwargs)         
    
    def simple_test_img_only(self,
                             img,
                             img_metas,
                             proposals=None,
                             rescale=False):
        feat = self.extract_img_feat(img)
        results_list = self.img_bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.img_bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img=None,
                    bboxes_2d=None,
                    rescale=False,
                    **kwargs):
        img_features = self.extract_img_feat(img, img_metas)

        points = torch.stack(points)
        seeds_3d, seed_3d_features, seed_indices = \
            self.extract_pts_feat(points)
        
        feat_dict = dict(
            seed_points=seeds_3d,
            seed_features=seed_3d_features,
            seed_indices=seed_indices)
        img_dict = dict(
            img_features=img_features,
            img_metas=img_metas,
        )
        bbox_preds = self.pts_bbox_head(
            feat_dict, self.test_cfg.pts.sample_mod, img_dict)
        bbox_list = self.pts_bbox_head.get_bboxes(
            points, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
