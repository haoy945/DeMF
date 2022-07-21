import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models import VoteHead
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.models.builder import build_loss
from mmdet3d.models.model_utils import VoteModule
from mmdet3d.ops import build_sa_module, furthest_point_sample
from mmdet3d.models.dense_heads.base_conv_bbox_head import BaseConvBboxHead
from mmdet3d.models.fusion_layers import (apply_3d_transformation,
                                          coord_2d_transform)
from mmdet3d.core.bbox import points_cam2img
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS

__all__ = ["CAVoteHead", "DeMFVoteHead"]


@HEADS.register_module()
class CAVoteHead(VoteHead):
    r"""
    Class agnostic vote head
    """
    def _get_cls_out_channels(self):
        if hasattr(self, 'semantic_loss'):
            return self.num_classes + 2
        else:
            return 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        return 6 + self.num_dir_bins * 2

    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask, bbox_preds)

        (vote_targets, vote_target_masks, dir_class_targets,
        dir_res_targets, mask_targets, objectness_targets, objectness_weights,
        box_loss_weights, distance_targets, centerness_targets, dir_targets) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                              bbox_preds['vote_points'],
                                              bbox_preds['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate distance loss
        size_res_loss = self.size_res_loss(
            bbox_preds['distance'],
            distance_targets,
            weight=box_loss_weights.unsqueeze(-1).repeat(1, 1, 6)
        )

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        if hasattr(self, 'semantic_loss'):
            # calculate semantic loss
            semantic_loss = self.semantic_loss(
                bbox_preds['sem_scores'].transpose(2, 1),
                mask_targets,
                weight=box_loss_weights)

        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_res_loss)

        if hasattr(self, 'semantic_loss'):
            losses.update(semantic_loss=semantic_loss)

        if self.iou_loss:
            corners_pred = self.bbox_coder.decode_corners(
                bbox_preds['distance'], bbox_preds['ref_points'])
            corners_target = self.bbox_coder.decode_corners(
                distance_targets, bbox_preds['ref_points'])
            iou_loss = self.iou_loss(
                corners_pred, corners_target, weight=box_loss_weights)
            losses['iou_loss'] = iou_loss

        if ret_target:
            losses['targets'] = targets

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, size_res_targets,
         dir_class_targets, dir_res_targets, centerness_targets,
         mask_targets, objectness_targets, objectness_masks,
         distance_targets, centerness_targets, dir_targets) = multi_apply(
            self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask, pts_instance_mask, aggregated_points
        )

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        dir_targets = torch.stack(dir_targets)
        mask_targets = torch.stack(mask_targets)
        distance_targets = torch.stack(distance_targets)
        centerness_targets = torch.stack(centerness_targets)

        return (vote_targets, vote_target_masks, dir_class_targets,
                dir_res_targets, mask_targets, objectness_targets, objectness_weights,
                box_loss_weights, distance_targets, centerness_targets,
                dir_targets)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None):
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]
        if self.bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                        selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets, dir_targets) = self.bbox_coder.encode(
            gt_bboxes_3d, gt_labels_3d, ret_dir_target=True)

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        center_targets = center_targets[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        size_res_targets = size_targets[assignment]
        dir_targets = dir_targets[assignment]

        mask_targets = gt_labels_3d[assignment]

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        # print(canonical_xyz.shape)
        # print(gt_bboxes_3d.yaw[assignment].shape)
        if self.bbox_coder.with_rot:
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment], 2).squeeze(1)

        distance_front  = size_res_targets[:, 0] - canonical_xyz[:, 0]
        distance_left   = size_res_targets[:, 1] - canonical_xyz[:, 1]
        distance_top    = size_res_targets[:, 2] - canonical_xyz[:, 2]
        distance_back   = size_res_targets[:, 0] + canonical_xyz[:, 0]
        distance_right  = size_res_targets[:, 1] + canonical_xyz[:, 1]
        distance_bottom = size_res_targets[:, 2] + canonical_xyz[:, 2]

        distance_targets = torch.cat(
            (distance_front.unsqueeze(-1),
             distance_left.unsqueeze(-1),
             distance_top.unsqueeze(-1),
             distance_back.unsqueeze(-1),
             distance_right.unsqueeze(-1),
             distance_bottom.unsqueeze(-1)),
            dim=-1
        )
        inside_mask = (distance_targets >= 0.).all(dim=-1)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        pos_mask = (euclidean_distance1 < self.train_cfg['pos_distance_thr']) & inside_mask
        objectness_targets[pos_mask] = 1

        distance_targets.clamp_(min=0)
        deltas = torch.cat(
            (distance_targets[:, 0:3, None], distance_targets[:, 3:6, None]),
            dim=-1
        )
        nominators = deltas.min(dim=-1).values.prod(dim=-1)
        denominators = deltas.max(dim=-1).values.prod(dim=-1) + 1e-6
        centerness_targets = (nominators / denominators + 1e-6) ** (1/3)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        return (
            vote_targets, vote_target_masks, size_res_targets,
            dir_class_targets, dir_res_targets, centerness_targets, mask_targets.long(),
            objectness_targets, objectness_masks, distance_targets, centerness_targets,
            dir_targets
        )

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        if hasattr(self, 'semantic_loss'):
            return super(CAVoteHead, self).get_bboxes(
                points, bbox_preds, input_metas, rescale=rescale, use_nms=use_nms
            )
        else:
            # decode boxes
            bbox3d = self.bbox_coder.decode(bbox_preds, mode='rpn')
            assert not use_nms
            return bbox3d


@HEADS.register_module()
class DeMFVoteHead(CAVoteHead):
    def __init__(self,
                 num_classes,
                 bbox_coder,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 semantic_loss=None,
                 iou_loss=None,
                 decoder=None,
                 init_cfg=None):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        if size_class_loss is not None:
            self.size_class_loss = build_loss(size_class_loss)
        if semantic_loss is not None:
            self.semantic_loss = build_loss(semantic_loss)
        if iou_loss is not None:
            self.iou_loss = build_loss(iou_loss)
        else:
            self.iou_loss = None

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)
        self.fp16_enabled = False

        self.num_decoder_layers = decoder.num_layers
        self.num_fusion_layers = decoder.num_layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                build_transformer_layer(decoder))

        # Bbox classification and regression
        self.conv_pred_layers = pred_layer_cfg.pop('conv_pred_layers')
        assert self.conv_pred_layers == self.num_decoder_layers + 1
        self.conv_preds = []
        for i in range(self.conv_pred_layers):
            conv_pred = BaseConvBboxHead(
                **pred_layer_cfg,
                num_cls_out_channels=self._get_cls_out_channels(),
                num_reg_out_channels=self._get_reg_out_channels())
            self.add_module('conv_pred'+str(i), conv_pred)
            self.conv_preds.append(conv_pred)

    def forward(self, feat_dict, sample_mod, img_dict):
        assert sample_mod in ['vote', 'seed', 'random', 'spec']

        seed_points, seed_features, seed_indices = self._extract_input(
            feat_dict)
        img_features, img_metas = img_dict['img_features'], img_dict['img_metas']

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, seed_features)
        results = dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset)

        # 2. aggregate vote_points
        if sample_mod == 'vote':
            # use fps in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=vote_points, features=vote_features)
        elif sample_mod == 'seed':
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(seed_points,
                                                   self.num_proposal)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif sample_mod == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(
                torch.randint(0, num_seed, (batch_size, self.num_proposal)),
                dtype=torch.int32)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif sample_mod == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=seed_points,
                features=seed_features,
                target_xyz=vote_points)
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mod} is not supported!')

        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret

        results['aggregated_points'] = aggregated_points
        results['aggregated_indices'] = aggregated_indices

        decode_res_all = self.transformer_decoder(
            features, aggregated_points, img_features, img_metas
        )
        results['decode_res_all'] = decode_res_all

        return results

    def transformer_decoder(self, 
                            features,
                            aggregated_points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []

        # get proposals
        cls_predictions, reg_predictions = self.conv_preds[0](features)
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions,
                                                aggregated_points)
        decode_res_all.append(decode_res)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                aggregated_points, img_features, img_metas)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query_pos = torch.cat(
                [decode_res['center'], decode_res['size']], 
                dim=-1).detach().clone()
            query = self.decoder[i](
                query=query,  # [N_query, BS, C_query]
                key=None,
                value=feat_flatten,  # [N_value, BS, C_value]
                query_pos=query_pos,  # [N_query, BS, C_query]
                key_padding_mask=mask_flatten,  # [BS, N_value]
                reference_points=reference_points,  # [BS, N_query, 2]
                spatial_shapes=spatial_shapes,  # [N_lvl, 2]
                level_start_index=level_start_index,  # [N_lvl]
                valid_ratios=valid_ratios,  # [BS, N_lvl, 2]
            )

            cls_predictions, reg_predictions = self.conv_preds[i+1](
                                                query.permute(1, 2, 0))
            decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                    reg_predictions,
                                                    aggregated_points)
            decode_res_all.append(decode_res)            

        return decode_res_all

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, seeds_3d_batch, img_metas):
        uv_all = []
        for seeds_3d, img_meta in zip(seeds_3d_batch, img_metas):
            img_shape = img_meta['img_shape']
            
            # first reverse the data transformations
            xyz_depth = apply_3d_transformation(
                seeds_3d, 'DEPTH', img_meta, reverse=True)
            
            # project points from depth to image
            depth2img = xyz_depth.new_tensor(img_meta['depth2img'])
            uv_origin = points_cam2img(xyz_depth, depth2img, False)
            # uv_origin = uv_origin - 1
            # uv_origin = (uv_origin - 1).round()
            
            # transform and normalize 2d coordinates
            uv_transformed = coord_2d_transform(img_meta, uv_origin, True)
            uv_transformed[:, 0] = uv_transformed[:, 0] / (img_shape[1] - 1)
            uv_transformed[:, 1] = uv_transformed[:, 1] / (img_shape[0] - 1)
            uv_transformed = torch.clamp(uv_transformed, 0, 1)
            
            uv_all.append(uv_transformed)
        uv_all = torch.stack(uv_all, dim=0)
        return uv_all

    def prepare_decoder_inputs(self, 
                               seeds_3d,
                               mlvl_feats,
                               img_metas,):
        # get_reference_points
        reference_points = self.get_reference_points(seeds_3d, img_metas)
        
        # get masks
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        mlvl_masks = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
        
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(
                zip(mlvl_feats, mlvl_masks)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        feat_flatten = feat_flatten.permute(1, 0, 2)

        return feat_flatten, mask_flatten, reference_points,\
            spatial_shapes, level_start_index, valid_ratios

    def loss(self, bbox_preds, *args, **kwargs):
        seed_points = bbox_preds.pop('seed_points')
        seed_indices = bbox_preds.pop('seed_indices')
        aggregated_points = bbox_preds.pop('aggregated_points')
        vote_points = bbox_preds.pop('vote_points')
        decode_res_all = bbox_preds.pop('decode_res_all')
        losses_all = []

        for decode_res in decode_res_all:
            _bbox_preds = dict(
                seed_points=seed_points,
                seed_indices=seed_indices,
                aggregated_points=aggregated_points,
                vote_points=vote_points,
            )
            _bbox_preds.update(decode_res)
            losses_all.append(self._loss(_bbox_preds, *args, **kwargs))
        
        losses = dict()
        assert self.num_fusion_layers + 1 == len(losses_all)
        for k in losses_all[0]:
            losses[k] = 0
            for i in range(self.num_fusion_layers + 1):
                losses[k] += losses_all[i][k] / (self.num_fusion_layers + 1)
        return losses

    @force_fp32(apply_to=('bbox_preds', ))
    def _loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask, bbox_preds)

        (vote_targets, vote_target_masks, dir_class_targets, dir_res_targets, 
        mask_targets, objectness_targets, objectness_weights, box_loss_weights, 
        distance_targets, dir_targets, size_targets, center_targets) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                              bbox_preds['vote_points'],
                                              bbox_preds['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate size loss
        size_reg_loss = self.size_res_loss(
            bbox_preds['size'],
            size_targets,
            weight=box_loss_weights.unsqueeze(-1).repeat(1, 1, 3)
        )

        # calculate center loss
        center_loss = self.center_loss(
            bbox_preds['center'],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1).repeat(1, 1, 3)
        )

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        if hasattr(self, 'semantic_loss'):
            # calculate semantic loss
            semantic_loss = self.semantic_loss(
                bbox_preds['sem_scores'].transpose(2, 1),
                mask_targets,
                weight=box_loss_weights)

        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_reg_loss,
            center_loss=center_loss,)

        if hasattr(self, 'semantic_loss'):
            losses.update(semantic_loss=semantic_loss)

        if self.iou_loss:
            corners_pred = self.bbox_coder.decode_corners(
                bbox_preds['center'], bbox_preds['size'])
            corners_target = self.bbox_coder.decode_corners(
                center_targets, size_targets)
            iou_loss = self.iou_loss(
                corners_pred, corners_target, weight=box_loss_weights)
            losses['iou_loss'] = iou_loss

        if ret_target:
            losses['targets'] = targets

        return losses

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        decode_res_all = bbox_preds.pop('decode_res_all')
        
        obj_scores = list()
        sem_scores = list()
        bbox3d = list()
        for i in self.test_cfg.ensemble_layers:
            decode_res = decode_res_all[i]
            obj_score = F.softmax(decode_res['obj_scores'], dim=-1)[..., -1]
            sem_score = F.softmax(decode_res['sem_scores'], dim=-1)
            bbox = self.bbox_coder.decode(decode_res)
            obj_scores.append(obj_score)
            sem_scores.append(sem_score)
            bbox3d.append(bbox)
            
        obj_scores = torch.cat(obj_scores, dim=1)
        sem_scores = torch.cat(sem_scores, dim=1)
        bbox3d = torch.cat(bbox3d, dim=1)
            
        if use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
                bbox = input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=self.bbox_coder.with_rot)
                results.append((bbox, score_selected, labels))

            return results
        else:
            return bbox3d

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, size_targets, center_targets,
         dir_class_targets, dir_res_targets, mask_targets,
         objectness_targets, objectness_masks, distance_targets,
         dir_targets) = multi_apply(
            self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask, pts_instance_mask, aggregated_points
        )

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        dir_targets = torch.stack(dir_targets)
        mask_targets = torch.stack(mask_targets)
        distance_targets = torch.stack(distance_targets)
        size_targets = torch.stack(size_targets)
        center_targets = torch.stack(center_targets)

        return (vote_targets, vote_target_masks, dir_class_targets,
                dir_res_targets, mask_targets, objectness_targets, objectness_weights,
                box_loss_weights, distance_targets, dir_targets,
                size_targets, center_targets)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None):
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]
        if self.bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                        selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets, dir_targets) = self.bbox_coder.encode(
            gt_bboxes_3d, gt_labels_3d, ret_dir_target=True)

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        center_targets = center_targets[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        size_res_targets = size_targets[assignment]
        dir_targets = dir_targets[assignment]

        mask_targets = gt_labels_3d[assignment]

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        # print(canonical_xyz.shape)
        # print(gt_bboxes_3d.yaw[assignment].shape)
        if self.bbox_coder.with_rot:
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment], 2).squeeze(1)

        size_res_targets_half = size_res_targets / 2.0
        distance_front  = size_res_targets_half[:, 0] - canonical_xyz[:, 0]
        distance_left   = size_res_targets_half[:, 1] - canonical_xyz[:, 1]
        distance_top    = size_res_targets_half[:, 2] - canonical_xyz[:, 2]
        distance_back   = size_res_targets_half[:, 0] + canonical_xyz[:, 0]
        distance_right  = size_res_targets_half[:, 1] + canonical_xyz[:, 1]
        distance_bottom = size_res_targets_half[:, 2] + canonical_xyz[:, 2]

        distance_targets = torch.cat(
            (distance_front.unsqueeze(-1),
             distance_left.unsqueeze(-1),
             distance_top.unsqueeze(-1),
             distance_back.unsqueeze(-1),
             distance_right.unsqueeze(-1),
             distance_bottom.unsqueeze(-1)),
            dim=-1
        )
        inside_mask = (distance_targets >= 0.).all(dim=-1)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        pos_mask = (euclidean_distance1 < self.train_cfg['pos_distance_thr']) & inside_mask
        objectness_targets[pos_mask] = 1

        return (
            vote_targets, vote_target_masks, size_res_targets, center_targets,
            dir_class_targets, dir_res_targets, mask_targets.long(),
            objectness_targets, objectness_masks, distance_targets,
            dir_targets
        )
