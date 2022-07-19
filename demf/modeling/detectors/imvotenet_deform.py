import numpy as np
import torch
import warnings

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.utils import MLP
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.detectors import ImVoteNet
from mmdet.core import bbox2result


def sample_valid_seeds(mask, num_sampled_seed=1024):
    r"""Randomly sample seeds from all imvotes.

    Modified from `<https://github.com/facebookresearch/imvotenet/blob/a8856345146bacf29a57266a2f0b874406fd8823/models/imvotenet.py#L26>`_

    Args:
        mask (torch.Tensor): Bool tensor in shape (
            seed_num*max_imvote_per_pixel), indicates
            whether this imvote corresponds to a 2D bbox.
        num_sampled_seed (int): How many to sample from all imvotes.

    Returns:
        torch.Tensor: Indices with shape (num_sampled_seed).
    """  # noqa: E501
    device = mask.device
    batch_size = mask.shape[0]
    sample_inds = mask.new_zeros((batch_size, num_sampled_seed),
                                 dtype=torch.int64)
    for bidx in range(batch_size):
        # return index of non zero elements
        valid_inds = torch.nonzero(mask[bidx, :]).squeeze(-1)
        if len(valid_inds) < num_sampled_seed:
            # compute set t1 - t2
            t1 = torch.arange(num_sampled_seed, device=device)
            t2 = valid_inds % num_sampled_seed
            combined = torch.cat((t1, t2))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]

            rand_inds = torch.randperm(
                len(difference),
                device=device)[:num_sampled_seed - len(valid_inds)]
            cur_sample_inds = difference[rand_inds]
            cur_sample_inds = torch.cat((valid_inds, cur_sample_inds))
        else:
            rand_inds = torch.randperm(
                len(valid_inds), device=device)[:num_sampled_seed]
            cur_sample_inds = valid_inds[rand_inds]
        sample_inds[bidx, :] = cur_sample_inds
    return sample_inds


@DETECTORS.register_module()
class ImVoteNet_Deformdetr(ImVoteNet):
    r"""`ImVoteNet <https://arxiv.org/abs/2001.10692>`_ for 3D detection."""

    def __init__(self,
                 pts_backbone=None,
                 pts_bbox_heads=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 img_bbox_head=None,
                 img_mlp=None,
                 freeze_img_branch=False,
                 fusion_layer=None,
                 num_sampled_seed=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        Base3DDetector.__init__(self, init_cfg=init_cfg)

        # point branch
        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_heads is not None:
            pts_bbox_head_common = pts_bbox_heads.common
            pts_bbox_head_common.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_common.update(test_cfg=test_cfg.pts)
            pts_bbox_head_joint = pts_bbox_head_common.copy()
            pts_bbox_head_joint.update(pts_bbox_heads.joint)
            pts_bbox_head_pts = pts_bbox_head_common.copy()
            pts_bbox_head_pts.update(pts_bbox_heads.pts)
            pts_bbox_head_img = pts_bbox_head_common.copy()
            pts_bbox_head_img.update(pts_bbox_heads.img)

            self.pts_bbox_head_joint = builder.build_head(pts_bbox_head_joint)
            self.pts_bbox_head_pts = builder.build_head(pts_bbox_head_pts)
            self.pts_bbox_head_img = builder.build_head(pts_bbox_head_img)
            self.pts_bbox_heads = [
                self.pts_bbox_head_joint, self.pts_bbox_head_pts,
                self.pts_bbox_head_img
            ]
            self.loss_weights = pts_bbox_heads.loss_weights

        # image branch
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_bbox_head is not None:
            img_bbox_head.update(train_cfg=train_cfg)
            img_bbox_head.update(test_cfg=test_cfg)
            self.img_bbox_head = builder.build_head(img_bbox_head)
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg
                   
        # fusion
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
            self.max_imvote_per_pixel = fusion_layer.max_imvote_per_pixel

        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()

        if img_mlp is not None:
            self.img_mlp = MLP(**img_mlp)

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

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        if self.with_img_bbox_head:
            for param in self.img_bbox_head.parameters():
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
            if self.with_img_bbox_head:
                self.img_bbox_head.eval()
            if self.with_img_backbone:
                self.img_backbone.eval()
            if self.with_img_neck:
                self.img_neck.eval()
    
    @torch.no_grad()
    def filter(self, ret, thresh=0.09):
        for i, _ret in enumerate(ret):
            scores = _ret[:,-1]
            index = scores > thresh
            ret[i] = _ret[index]
        return ret
    
    @torch.no_grad()
    def extract_bboxes_2d(self,
                          img,
                          img_metas,
                          train=True,
                          bboxes_2d=None,
                          **kwargs):
        """Extract bounding boxes from 2d detector.

        Args:
            img (torch.Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): Image meta info.
            train (bool): train-time or not.
            bboxes_2d (list[torch.Tensor]): provided 2d bboxes,
                not supported yet.

        Return:
            list[torch.Tensor]: a list of processed 2d bounding boxes.
        """
        if bboxes_2d is None:
            x = self.extract_img_feat(img)
            results_list = self.img_bbox_head.simple_test(x, img_metas, rescale=False)
            rets = [
                bbox2result(det_bboxes, det_labels, self.img_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]

            rets_processed = []
            for ret in rets:
                ret = self.filter(ret)
                tmp = np.concatenate(ret, axis=0)
                sem_class = img.new_zeros((len(tmp)))
                start = 0
                for i, bboxes in enumerate(ret):
                    sem_class[start:start + len(bboxes)] = i
                    start += len(bboxes)
                ret = img.new_tensor(tmp)

                # append class index
                ret = torch.cat([ret, sem_class[:, None]], dim=-1)
                inds = torch.argsort(ret[:, 4], descending=True)
                ret = ret.index_select(0, inds)

                # drop half bboxes during training for better generalization
                if train:
                    rand_drop = torch.randperm(len(ret))[:(len(ret) + 1) // 2]
                    rand_drop = torch.sort(rand_drop)[0]
                    ret = ret[rand_drop]

                rets_processed.append(ret.float())
            return rets_processed
        else:
            rets_processed = []
            for ret in bboxes_2d:
                if len(ret) > 0 and train:
                    rand_drop = torch.randperm(len(ret))[:(len(ret) + 1) // 2]
                    rand_drop = torch.sort(rand_drop)[0]
                    ret = ret[rand_drop]
                rets_processed.append(ret.float())
            return rets_processed
    
    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      bboxes_2d=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      **kwargs):
        """Forwarding of train for image branch pretrain or stage 2 train.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img (torch.Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image and point cloud meta info
                dict. For example, keys include 'ori_shape', 'img_norm_cfg',
                and 'transformation_3d_flow'. For details on the values of
                the keys see `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[torch.Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[torch.Tensor]): class indices for each
                2d bounding box.
            gt_bboxes_ignore (None | list[torch.Tensor]): specify which
                2d bounding boxes can be ignored when computing the loss.
            gt_masks (None | torch.Tensor): true segmentation masks for each
                2d bbox, used if the architecture supports a segmentation task.
            proposals: override rpn proposals (2d) with custom proposals.
                Use when `with_rpn` is False.
            bboxes_2d (list[torch.Tensor]): provided 2d bboxes,
                not supported yet.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): 3d gt bboxes.
            gt_labels_3d (list[torch.Tensor]): gt class labels for 3d bboxes.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.

        Returns:
            dict[str, torch.Tensor]: a dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if points is None:
            x = self.extract_img_feat(img)
            losses = self.img_bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
            
            return losses
        else:
            bboxes_2d = self.extract_bboxes_2d(
                img, img_metas, bboxes_2d=bboxes_2d, **kwargs)

            points = torch.stack(points)
            seeds_3d, seed_3d_features, seed_indices = \
                self.extract_pts_feat(points)

            img_features, masks = self.fusion_layer(img, bboxes_2d, seeds_3d,
                                                    img_metas)

            inds = sample_valid_seeds(masks, self.num_sampled_seed)
            batch_size, img_feat_size = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.view(batch_size, 1,
                                 -1).expand(-1, img_feat_size, -1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.view(batch_size, 1,
                                        -1).expand(-1, pts_feat_size, -1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)

            img_features = self.img_mlp(img_features)
            fused_features = torch.cat([seed_3d_features, img_features], dim=1)

            feat_dict_joint = dict(
                seed_points=seeds_3d,
                seed_features=fused_features,
                seed_indices=seed_indices)
            feat_dict_pts = dict(
                seed_points=seeds_3d,
                seed_features=seed_3d_features,
                seed_indices=seed_indices)
            feat_dict_img = dict(
                seed_points=seeds_3d,
                seed_features=img_features,
                seed_indices=seed_indices)

            loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                           pts_semantic_mask, pts_instance_mask, img_metas)
            bbox_preds_joints = self.pts_bbox_head_joint(
                feat_dict_joint, self.train_cfg.pts.sample_mod)
            bbox_preds_pts = self.pts_bbox_head_pts(
                feat_dict_pts, self.train_cfg.pts.sample_mod)
            bbox_preds_img = self.pts_bbox_head_img(
                feat_dict_img, self.train_cfg.pts.sample_mod)
            losses_towers = []
            losses_joint = self.pts_bbox_head_joint.loss(
                bbox_preds_joints,
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses_pts = self.pts_bbox_head_pts.loss(
                bbox_preds_pts,
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses_img = self.pts_bbox_head_img.loss(
                bbox_preds_img,
                *loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore)
            losses_towers.append(losses_joint)
            losses_towers.append(losses_pts)
            losses_towers.append(losses_img)
            combined_losses = dict()
            for loss_term in losses_joint:
                if 'loss' in loss_term:
                    combined_losses[loss_term] = 0
                    for i in range(len(losses_towers)):
                        combined_losses[loss_term] += \
                            losses_towers[i][loss_term] * \
                            self.loss_weights[i]
                else:
                    # only save the metric of the joint head
                    # if it is not a loss
                    combined_losses[loss_term] = \
                        losses_towers[0][loss_term]

            return combined_losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None,
                     bboxes_2d=None,
                     **kwargs):
        """Forwarding of test for image branch pretrain or stage 2 train.

        Args:
            points (list[list[torch.Tensor]], optional): the outer
                list indicates test-time augmentations and the inner
                list contains all points in the batch, where each Tensor
                should have a shape NxC. Defaults to None.
            img_metas (list[list[dict]], optional): the outer list
                indicates test-time augs (multiscale, flip, etc.)
                and the inner list indicates images in a batch.
                Defaults to None.
            img (list[list[torch.Tensor]], optional): the outer
                list indicates test-time augmentations and inner Tensor
                should have a shape NxCxHxW, which contains all images
                in the batch. Defaults to None. Defaults to None.
            bboxes_2d (list[list[torch.Tensor]], optional):
                Provided 2d bboxes, not supported yet. Defaults to None.

        Returns:
            list[list[torch.Tensor]]|list[dict]: Predicted 2d or 3d boxes.
        """
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
                # 这一部分还没有实现
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
        r"""Test without augmentation, image network pretrain. May refer to
        `<https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/two_stage.py>`_.

        Args:
            img (torch.Tensor): Should have a shape NxCxHxW, which contains
                all images in the batch.
            img_metas (list[dict]):
            proposals (list[Tensor], optional): override rpn proposals
                with custom proposals. Defaults to None.
            rescale (bool, optional): Whether or not rescale bboxes to the
                original shape of input image. Defaults to False.

        Returns:
            list[list[torch.Tensor]]: Predicted 2d boxes and scores. The outer list indicates different img prediction in batch 
            and the inner list indicates different class and each tensor contains all predeiction in a img in this class. 
            Tensor shape: 1x5, [tl_x, tl_y, br_x, br_y, score]
        """  # noqa: E501
        feat = self.extract_img_feat(img)
        results_list = self.img_bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.img_bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        # for idx, _bbox2results in enumerate(bbox_results):
        #         bbox_results[idx] = self.filter(_bbox2results,thresh=0.09)
        return bbox_results

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img=None,
                    bboxes_2d=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation, stage 2.

        Args:
            points (list[torch.Tensor], optional): Elements in the list
                should have a shape NxC, the list indicates all point-clouds
                in the batch. Defaults to None.
            img_metas (list[dict], optional): List indicates
                images in a batch. Defaults to None.
            img (torch.Tensor, optional): Should have a shape NxCxHxW,
                which contains all images in the batch. Defaults to None.
            bboxes_2d (list[torch.Tensor], optional):
                Provided 2d bboxes, not supported yet. Defaults to None.
            rescale (bool, optional): Whether or not rescale bboxes.
                Defaults to False.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        bboxes_2d = self.extract_bboxes_2d(
            img, img_metas, train=False, bboxes_2d=bboxes_2d, **kwargs)

        points = torch.stack(points)
        seeds_3d, seed_3d_features, seed_indices = \
            self.extract_pts_feat(points)

        img_features, masks = self.fusion_layer(img, bboxes_2d, seeds_3d,
                                                img_metas)

        inds = sample_valid_seeds(masks, self.num_sampled_seed)
        batch_size, img_feat_size = img_features.shape[:2]
        pts_feat_size = seed_3d_features.shape[1]
        inds_img = inds.view(batch_size, 1, -1).expand(-1, img_feat_size, -1)
        img_features = img_features.gather(-1, inds_img)
        inds = inds % inds.shape[1]
        inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
        seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
        inds_seed_feats = inds.view(batch_size, 1,
                                    -1).expand(-1, pts_feat_size, -1)
        seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
        seed_indices = seed_indices.gather(1, inds)

        img_features = self.img_mlp(img_features)

        fused_features = torch.cat([seed_3d_features, img_features], dim=1)

        feat_dict = dict(
            seed_points=seeds_3d,
            seed_features=fused_features,
            seed_indices=seed_indices)
        bbox_preds = self.pts_bbox_head_joint(feat_dict,
                                              self.test_cfg.pts.sample_mod)
        bbox_list = self.pts_bbox_head_joint.get_bboxes(
            points, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_img_only(self, img, img_metas, rescale=False):
        r"""Test function with augmentation, image network pretrain. May refer
        to `<https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/two_stage.py>`_.

        Args:
            img (list[list[torch.Tensor]], optional): the outer
                list indicates test-time augmentations and inner Tensor
                should have a shape NxCxHxW, which contains all images
                in the batch. Defaults to None. Defaults to None.
            img_metas (list[list[dict]], optional): the outer list
                indicates test-time augs (multiscale, flip, etc.)
                and the inner list indicates images in a batch.
                Defaults to None.
            rescale (bool, optional): Whether or not rescale bboxes to the
                original shape of input image. If rescale is False, then
                returned bboxes and masks will fit the scale of imgs[0].
                Defaults to None.

        Returns:
            list[list[torch.Tensor]]: Predicted 2d boxes.
        """  # noqa: E501
        assert self.with_img_bbox, 'Img bbox head must be implemented.'
        assert self.with_img_backbone, 'Img backbone must be implemented.'
        assert self.with_img_rpn, 'Img rpn must be implemented.'
        assert self.with_img_roi_head, 'Img roi head must be implemented.'

        x = self.extract_img_feats(img)
        proposal_list = self.img_rpn_head.aug_test_rpn(x, img_metas)

        return self.img_roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self,
                 points=None,
                 img_metas=None,
                 imgs=None,
                 bboxes_2d=None,
                 rescale=False,
                 **kwargs):
        """Test function with augmentation, stage 2.

        Args:
            points (list[list[torch.Tensor]], optional): the outer
                list indicates test-time augmentations and the inner
                list contains all points in the batch, where each Tensor
                should have a shape NxC. Defaults to None.
            img_metas (list[list[dict]], optional): the outer list
                indicates test-time augs (multiscale, flip, etc.)
                and the inner list indicates images in a batch.
                Defaults to None.
            imgs (list[list[torch.Tensor]], optional): the outer
                list indicates test-time augmentations and inner Tensor
                should have a shape NxCxHxW, which contains all images
                in the batch. Defaults to None. Defaults to None.
            bboxes_2d (list[list[torch.Tensor]], optional):
                Provided 2d bboxes, not supported yet. Defaults to None.
            rescale (bool, optional): Whether or not rescale bboxes.
                Defaults to False.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_pts_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta, bbox_2d, img in zip(feats, points_cat,
                                                      img_metas, bboxes_2d,
                                                      imgs):

            bbox_2d = self.extract_bboxes_2d(
                img, img_metas, train=False, bboxes_2d=bbox_2d, **kwargs)

            seeds_3d, seed_3d_features, seed_indices = x

            img_features, masks = self.fusion_layer(img, bbox_2d, seeds_3d,
                                                    img_metas)

            inds = sample_valid_seeds(masks, self.num_sampled_seed)
            batch_size, img_feat_size = img_features.shape[:2]
            pts_feat_size = seed_3d_features.shape[1]
            inds_img = inds.view(batch_size, 1,
                                 -1).expand(-1, img_feat_size, -1)
            img_features = img_features.gather(-1, inds_img)
            inds = inds % inds.shape[1]
            inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            inds_seed_feats = inds.view(batch_size, 1,
                                        -1).expand(-1, pts_feat_size, -1)
            seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            seed_indices = seed_indices.gather(1, inds)

            img_features = self.img_mlp(img_features)

            fused_features = torch.cat([seed_3d_features, img_features], dim=1)

            feat_dict = dict(
                seed_points=seeds_3d,
                seed_features=fused_features,
                seed_indices=seed_indices)
            bbox_preds = self.pts_bbox_head_joint(feat_dict,
                                                  self.test_cfg.pts.sample_mod)
            bbox_list = self.pts_bbox_head_joint.get_bboxes(
                pts_cat, bbox_preds, img_metas, rescale=rescale)

            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
