import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from mmcv.cnn import constant_init
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmdet.core import reduce_mean, build_assigner
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import Scale, bias_init_with_prob

from mmdet3d.ops.pcdet_nms import pcdet_nms_gpu, pcdet_nms_normal_gpu
from mmdet3d.models import builder
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.models.fusion_layers import (apply_3d_transformation,
                                          coord_2d_transform)


@HEADS.register_module()
class CAHead(nn.Module):
    def __init__(self, 
                 fusion_layer=None,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),):
        super().__init__()
        self.centerness = nn.Linear(256, 1)
        self.reg = nn.Linear(256, 8)
        self.cls = nn.Linear(256, 10)
        self.upsample_layer = nn.Linear(128, 256)
        # self.scales = nn.ModuleList([Scale(1.) for _ in range(4)])
        if fusion_layer:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        else:
            self.fusion_layer = None
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.yaw_parametrization = yaw_parametrization

    def forward(self, select_points, img_dict=None):
        points, features, sort_inds = select_points
        if img_dict and self.fusion_layer:
            img_features, img_metas = img_dict['img_features'], img_dict['img_metas']
            # fusion layer, fused_features stage*batch*C*N
            features = self.upsample_layer(features) # 128->256
            fused_features = self.fusion_layer(
                img_features, points, features, img_metas)
            fused_features = fused_features.permute(0,1,3,2)[-1]
        else:
            features = self.upsample_layer(features) # 128->256
            fused_features = features
        centerness = self.centerness(fused_features)
        cls_scores = self.cls(fused_features)
        reg_final = self.reg(fused_features)
        # reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_distance = torch.exp(reg_final[:, :6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)
        
        return centerness, bbox_pred, cls_scores

    def loss(self, stage_preds, targets, points):
        loss_centerness, loss_bbox, loss_cls = [], [], []
        batch_size = points.shape[0]
        for i in range(batch_size):
            img_loss_centerness, img_loss_bbox, img_loss_cls = self._loss_single(
                centerness=stage_preds[0][i],
                bbox_preds=stage_preds[1][i],
                cls_scores=stage_preds[2][i],
                points=points[i],
                centerness_targets=targets[0][i],
                bbox_targets=targets[1][i],
                labels=targets[2][i],
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            stage2_loss_centerness=torch.mean(torch.stack(loss_centerness)),
            stage2_loss_bbox=torch.mean(torch.stack(loss_bbox)),
            stage2_loss_cls=torch.mean(torch.stack(loss_cls))
        )

    def _loss_single(self, centerness, bbox_preds, cls_scores, points,
                    centerness_targets, bbox_targets, labels):
        
        # skip background
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            )
        else:
            # loss_centerness = centerness.sum() * 0.0
            # loss_bbox = bbox_preds.sum() * 0.0
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        return loss_centerness, loss_bbox, loss_cls

    def get_bboxes(self, stage_preds, points, img_metas,
                   rescale=False):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=stage_preds[0][i],
                bbox_preds=stage_preds[1][i],
                cls_scores=stage_preds[2][i],
                points=points[i],
                img_meta=img_metas[i]
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta):
        scores = cls_scores.sigmoid() * centernesses.sigmoid()
        bboxes = self._bbox_pred_to_bbox(points, bbox_preds)
        return bboxes, scores
    
    def _bbox_pred_to_bbox(self, points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        if self.yaw_parametrization == 'naive':
            # ..., alpha
            return torch.cat((
                base_bbox,
                bbox_pred[:, 6:7]
            ), -1)
        elif self.yaw_parametrization == 'sin-cos':
            # ..., sin(a), cos(a)
            norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
            sin = bbox_pred[:, 6:7] / norm
            cos = bbox_pred[:, 7:8] / norm
            return torch.cat((
                base_bbox,
                torch.atan2(sin, cos)
            ), -1)
        else:  # self.yaw_parametrization == 'fcaf3d'
            # ..., sin(2a)ln(q), cos(2a)ln(q)
            scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
            q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
            alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
            return torch.stack((
                x_center,
                y_center,
                z_center,
                scale / (1 + q),
                scale / (1 + q) * q,
                bbox_pred[:, 5] + bbox_pred[:, 4],
                alpha
            ), dim=-1)


@HEADS.register_module()
class CAHeadIter(CAHead):
    def __init__(self, 
                 decoder=None,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),):
        super().__init__()
        self.upsample_layer = nn.Conv1d(128, 256, kernel_size=1)
        
        self.num_decoder_layers = decoder.num_layers
        self.num_fusion_layers = decoder.num_layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                build_transformer_layer(decoder))

        self.conv_preds = nn.ModuleList()
        for _ in range(self.num_decoder_layers + 1):
            self.conv_preds.append(
                nn.Conv1d(256, 19, kernel_size=1)
            )
        
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.yaw_parametrization = yaw_parametrization

    def forward(self, select_points, img_dict):
        points, features, _ = select_points
        # [B, C, N]
        features = self.upsample_layer(features.permute(0, 2, 1))
        img_features, img_metas = img_dict['img_features'], img_dict['img_metas']

        preds_all = self.transformer_decoder(
            features, points, img_features, img_metas
        )
        return preds_all

    def split_pred(self, preds, base_xyz):
        results = {}
        start, end = 0, 0

        preds = preds.transpose(2, 1)

        # decode centerness
        end += 1
        # (batch_size, 1, num_proposal)
        results['centerness'] = preds[..., start:end].contiguous()
        start = end

        # decode bbox
        end += 6
        distance = preds[..., start:end].contiguous()
        distance = torch.exp(distance)
        results['distance'] = distance
        start = end
        end += 2
        angle = preds[..., start:end].contiguous()
        results['bbox_pred'] = torch.cat((distance, angle), dim=-1)
        start = end

        # decode directions
        end += 10
        results['cls_scores'] = preds[..., start:end].contiguous()
        start = end

        results['ref_points'] = base_xyz  # (batch_size, num_proposal, 3)

        return results

    def transformer_decoder(self, 
                            features,
                            points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []

        # get proposals
        predictions = self.conv_preds[0](features)
        decode_res = self.split_pred(predictions, points)
        decode_res_all.append(decode_res)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query_pos = torch.cat(
                [decode_res['distance'], decode_res['ref_points']], 
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

            predictions = self.conv_preds[i+1](query.permute(1, 2, 0))
            decode_res = self.split_pred(predictions, points)
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

    def loss(self, preds_all, targets, points):
        losses_all = []
        for preds in preds_all:
            centerness = preds['centerness']
            bbox_pred = preds['bbox_pred']
            cls_scores = preds['cls_scores']
            bbox_preds = (centerness, bbox_pred, cls_scores)
            losses_all.append(self._loss(bbox_preds, targets, points))
        
        losses = dict()
        assert self.num_fusion_layers + 1 == len(losses_all)
        for k in losses_all[0]:
            losses[k] = 0
            for i in range(self.num_fusion_layers + 1):
                losses[k] += losses_all[i][k] / (self.num_fusion_layers + 1)
        return losses

    def _loss(self, stage_preds, targets, points):
        loss_centerness, loss_bbox, loss_cls = [], [], []
        batch_size = points.shape[0]
        for i in range(batch_size):
            img_loss_centerness, img_loss_bbox, img_loss_cls = self._loss_single(
                centerness=stage_preds[0][i],
                bbox_preds=stage_preds[1][i],
                cls_scores=stage_preds[2][i],
                points=points[i],
                centerness_targets=targets[0][i],
                bbox_targets=targets[1][i],
                labels=targets[2][i],
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            stage2_loss_centerness=torch.mean(torch.stack(loss_centerness)),
            stage2_loss_bbox=torch.mean(torch.stack(loss_bbox)),
            stage2_loss_cls=torch.mean(torch.stack(loss_cls))
        )

    def get_bboxes(self, preds_all, points, img_metas,
                   rescale=False):
        results_all = []
        for preds in preds_all:
            results = []
            for i in range(len(img_metas)):
                result = self._get_bboxes_single(
                    centernesses=preds['centerness'][i],
                    bbox_preds=preds['bbox_pred'][i],
                    cls_scores=preds['cls_scores'][i],
                    points=points[i],
                    img_meta=img_metas[i]
                )
                results.append(result)
            results_all.append(results)
        return results_all


@HEADS.register_module()
class CAHeadGridOffset(CAHeadIter):
    def transformer_decoder(self, 
                            features,
                            points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []

        # get proposals
        predictions = self.conv_preds[0](features)
        decode_res = self.split_pred(predictions, points)
        decode_res_all.append(decode_res)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query_pos = torch.cat(
                [decode_res['distance'], decode_res['ref_points']], 
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
                img_features=img_features,
            )

            predictions = self.conv_preds[i+1](query.permute(1, 2, 0))
            decode_res = self.split_pred(predictions, points)
            decode_res_all.append(decode_res)            

        return decode_res_all


@HEADS.register_module()
class CAHeadIterOffset(CAHeadIter):
    def transformer_decoder(self, 
                            features,
                            points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []

        # get proposals
        predictions = self.conv_preds[0](features)
        decode_res = self.split_pred(predictions, points)
        decode_res_all.append(decode_res)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)
        offset_features = self.get_offset_features(
            reference_points, img_features, valid_ratios)
        offset_features = offset_features.permute(0, 2, 1)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query_pos = torch.cat(
                [decode_res['distance'], decode_res['ref_points']], 
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
                offset_features=offset_features,  # [BS, N_query, C_query]
            )

            predictions = self.conv_preds[i+1](query.permute(1, 2, 0))
            decode_res = self.split_pred(predictions, points)
            decode_res_all.append(decode_res)            

        return decode_res_all

    def get_offset_features(self, reference_points, img_features, valid_ratios):
        used_features = img_features[0]
        # convert to [-1,1]
        reference_points_ = reference_points * valid_ratios[:, 0:1]
        unnormalized_points = 2 * reference_points_ - 1
        offset_features = F.grid_sample(
            used_features,
            unnormalized_points[:,:,None,:],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        offset_features = offset_features.squeeze(dim=3)
        return offset_features


@HEADS.register_module()
class CAHeadIterOffsetRefPoint(CAHeadIterOffset):
    def __init__(self, 
                 decoder=None,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),):
        nn.Module.__init__(self)
        self.upsample_layer = nn.Conv1d(128, 256, kernel_size=1)
        
        self.num_decoder_layers = decoder.num_layers
        self.num_fusion_layers = decoder.num_layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                build_transformer_layer(decoder))

        self.conv_preds = nn.ModuleList()
        for _ in range(self.num_decoder_layers + 1):
            self.conv_preds.append(
                nn.Conv1d(256, 19, kernel_size=1)
            )

        self.query_embedding = nn.Embedding(256, 256)
        self.reference_points = nn.Linear(256, 2)
        
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.yaw_parametrization = yaw_parametrization

    def get_reference_points(self, seeds_3d_batch, img_metas):
        bs, _, _ = seeds_3d_batch.shape
        query_embeds = self.query_embedding.weight
        query_embeds = query_embeds.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embeds).sigmoid()
        return reference_points


@HEADS.register_module()
class CAHeadIterOffsetRefPointRefine(CAHeadIterOffset):
    def __init__(self, 
                 decoder=None,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),):
        nn.Module.__init__(self)
        self.upsample_layer = nn.Conv1d(128, 256, kernel_size=1)
        
        self.num_decoder_layers = decoder.num_layers
        self.num_fusion_layers = decoder.num_layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                build_transformer_layer(decoder))

        self.conv_preds = nn.ModuleList()
        for _ in range(self.num_decoder_layers + 1):
            self.conv_preds.append(
                nn.Conv1d(256, 19, kernel_size=1)
            )

        self.reference_points_refine = nn.Linear(256, 2)
        constant_init(self.reference_points_refine, val=0., bias=0.)
        
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.yaw_parametrization = yaw_parametrization

    def get_reference_points(self, seeds_3d_batch, img_metas, img_features, valid_ratios, spatial_shapes):
        uv_all = []
        for seeds_3d, img_meta in zip(seeds_3d_batch, img_metas):
            img_shape = img_meta['img_shape']
            
            # first reverse the data transformations
            xyz_depth = apply_3d_transformation(
                seeds_3d, 'DEPTH', img_meta, reverse=True)
            
            # project points from depth to image
            depth2img = xyz_depth.new_tensor(img_meta['depth2img'])
            uv_origin = points_cam2img(xyz_depth, depth2img, False)
            
            # transform and normalize 2d coordinates
            uv_transformed = coord_2d_transform(img_meta, uv_origin, True)
            uv_transformed[:, 0] = uv_transformed[:, 0] / (img_shape[1] - 1)
            uv_transformed[:, 1] = uv_transformed[:, 1] / (img_shape[0] - 1)
            uv_transformed = torch.clamp(uv_transformed, 0, 1)
            
            uv_all.append(uv_transformed)
        uv_all = torch.stack(uv_all, dim=0)

        refine_features = self.get_offset_features(uv_all, img_features, valid_ratios)
        refine = self.reference_points_refine(refine_features.permute(0, 2, 1))

        offset_normalizer = torch.stack(
                [spatial_shapes[0:1, 1], spatial_shapes[0:1, 0]], -1)
        reference_points = uv_all + refine / offset_normalizer[None, ...]
        return reference_points

    def prepare_decoder_inputs(self, 
                               seeds_3d,
                               mlvl_feats,
                               img_metas,):
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

        # get_reference_points
        reference_points = self.get_reference_points(seeds_3d, img_metas, mlvl_feats, valid_ratios, spatial_shapes)

        return feat_flatten, mask_flatten, reference_points,\
            spatial_shapes, level_start_index, valid_ratios


@HEADS.register_module()
class CAHeadOffset(CAHeadIterOffset):
    def transformer_decoder(self, 
                            features,
                            points,
                            img_features,
                            img_metas,
                            ):
        decode_res_all = []

        # get proposals
        predictions = self.conv_preds[0](features)
        decode_res = self.split_pred(predictions, points)
        decode_res_all.append(decode_res)

        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)
        offset_features = self.get_offset_features(
            reference_points, img_features, valid_ratios)
        offset_features = offset_features.permute(0, 2, 1)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query_pos = None
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
                offset_features=offset_features,  # [BS, N_query, C_query]
            )

            predictions = self.conv_preds[i+1](query.permute(1, 2, 0))
            decode_res = self.split_pred(predictions, points)
            decode_res_all.append(decode_res)            

        return decode_res_all


@HEADS.register_module()
class CAHeadOffsetNoAug(CAHeadIter):
    def __init__(self, 
                 decoder=None,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),):
        nn.Module.__init__(self)
        self.upsample_layer = nn.Conv1d(128, 256, kernel_size=1)
        
        self.num_decoder_layers = decoder.num_layers
        self.num_fusion_layers = decoder.num_layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                build_transformer_layer(decoder))

        self.conv_pred = nn.Conv1d(256, 19, kernel_size=1)
        
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.yaw_parametrization = yaw_parametrization

    def transformer_decoder(self, 
                            features,
                            points,
                            img_features,
                            img_metas,
                            ):
        # get inputs
        feat_flatten, mask_flatten, reference_points, spatial_shapes,\
            level_start_index, valid_ratios = self.prepare_decoder_inputs(
                points, img_features, img_metas)
        offset_features = self.get_offset_features(
            reference_points, img_features, valid_ratios)
        offset_features = offset_features.permute(0, 2, 1)

        query = features.permute(2, 0, 1)
        for i in range(self.num_decoder_layers):
            query = self.decoder[i](
                query=query,  # [N_query, BS, C_query]
                key=None,
                value=feat_flatten,  # [N_value, BS, C_value]
                query_pos=None,  # [N_query, BS, C_query]
                key_padding_mask=mask_flatten,  # [BS, N_value]
                reference_points=reference_points,  # [BS, N_query, 2]
                spatial_shapes=spatial_shapes,  # [N_lvl, 2]
                level_start_index=level_start_index,  # [N_lvl]
                valid_ratios=valid_ratios,  # [BS, N_lvl, 2]
                offset_features=offset_features,  # [BS, N_query, C_query]
            )  

        predictions = self.conv_pred(query.permute(1, 2, 0))
        decode_res = self.split_pred(predictions, points)        

        return decode_res

    def get_offset_features(self, reference_points, img_features, valid_ratios):
        used_features = img_features[0]
        # convert to [-1,1]
        reference_points_ = reference_points * valid_ratios[:, 0:1]
        unnormalized_points = 2 * reference_points_ - 1
        offset_features = F.grid_sample(
            used_features,
            unnormalized_points[:,:,None,:],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        offset_features = offset_features.squeeze(dim=3)
        return offset_features

    def loss(self, preds, targets, points):
        centerness = preds['centerness']
        bbox_pred = preds['bbox_pred']
        cls_scores = preds['cls_scores']
        bbox_preds = (centerness, bbox_pred, cls_scores)
        losses = self._loss(bbox_preds, targets, points)
        return losses

    def get_bboxes(self, preds, points, img_metas,
                   rescale=False):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=preds['centerness'][i],
                bbox_preds=preds['bbox_pred'][i],
                cls_scores=preds['cls_scores'][i],
                points=points[i],
                img_meta=img_metas[i]
            )
            results.append(result)
        return [results]
    

@HEADS.register_module()
class Fcaf3DNeckWithHead_my(nn.Module):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 pts_threshold,
                 assigner,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(Fcaf3DNeckWithHead_my, self).__init__()
        self.voxel_size = voxel_size
        self.yaw_parametrization = yaw_parametrization
        self.assigner = build_assigner(assigner)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pts_threshold = pts_threshold
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)
        self.stage2_reg = nn.Linear(out_channels, n_reg_outs)
        self.stage2_cls = nn.Linear(out_channels, n_classes)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))

        # head layers
        self.centerness_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, dimension=3)
        self.reg_conv = ME.MinkowskiConvolution(out_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(len(in_channels))])

    def init_weights(self):
        nn.init.normal_(self.centerness_conv.kernel, std=.01)
        nn.init.normal_(self.reg_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    def forward(self, x):
        outs = []
        outs_features = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, scores)

            out = self.__getattr__(f'out_block_{i}')(x)
            out = self.forward_single(out, self.scales[i])
            scores = out[-1]
            outs_features.append(out[-2])
            outs.append(out[:-2])
        
        batch_size = len(outs[-1][0])
        # turned to scale 1 to 4, 1 is the max scale
        outs = outs[::-1]
        outs_features = outs_features[::-1]
        points = []
        features = []
        sort_inds = []
        for i in range(batch_size):
            _centernesses = [out[0][i] for out in outs]
            _cls_scores = [out[2][i] for out in outs]
            _points = [out[3][i] for out in outs]
            _features = [out[i] for out in outs_features]
            _centernesses = torch.cat(_centernesses, dim=0).squeeze()
            _cls_scores = torch.cat(_cls_scores, dim=0)
            _points = torch.cat(_points, dim=0)
            _features = torch.cat(_features, dim=0)
            # select topk
            select_scores = _cls_scores.sigmoid() * _centernesses.sigmoid().unsqueeze(dim=-1)
            max_scores, _ = select_scores.max(dim=1)
            max_k = max_scores.shape[0]
            top_k = min(max_k, 256)
            # top_k = min(max_k, 512)
            # if top_k<2000:
            #     print(top_k)
            inds = torch.topk(max_scores, top_k)[1]
            _sort_inds = torch.sort(inds)[0]
            features.append(_features[_sort_inds])
            points.append(_points[_sort_inds])
            sort_inds.append(_sort_inds)
        
        features = torch.stack(features, dim=0)
        sort_inds = torch.stack(sort_inds, dim=0)
        points = torch.stack(points, dim=0)
        
        # return zip(*outs[::-1])
        return zip(*outs), (points, features, sort_inds)

    def _prune(self, x, scores):
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros((len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             points,
             gt_bboxes,
             gt_labels,
             img_metas):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        loss_centerness, loss_bbox, loss_cls = [], [], []
        targets = []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls, img_targets = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
            targets.append(img_targets)
        return dict(
            loss_centerness=torch.mean(torch.stack(loss_centerness)),
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls))
        ), targets

    # per image
    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        with torch.no_grad():
            centerness_targets, bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)

        centerness = torch.cat(centernesses)
        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        # skip background
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        
        targets = (centerness_targets, bbox_targets, labels)
        return loss_centerness, loss_bbox, loss_cls, targets

    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta,):
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, point in zip(
            centernesses, bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid() * centerness.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            # print(bboxes.shape)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        # bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        # return bboxes, scores, labels
        return bboxes, scores

    # per scale
    def forward_single(self, x, scale):
        centerness = self.centerness_conv(x).features
        scores = self.cls_conv(x)
        cls_score = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.reg_conv(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        features = x.decomposed_features

        return centernesses, bbox_preds, cls_scores, points, features, prune_scores

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred.view(-1,7)

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        if self.yaw_parametrization == 'naive':
            # ..., alpha
            return torch.cat((
                base_bbox,
                bbox_pred[:, 6:7]
            ), -1)
        elif self.yaw_parametrization == 'sin-cos':
            # ..., sin(a), cos(a)
            norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
            sin = bbox_pred[:, 6:7] / norm
            cos = bbox_pred[:, 7:8] / norm
            return torch.cat((
                base_bbox,
                torch.atan2(sin, cos)
            ), -1)
        else:  # self.yaw_parametrization == 'fcaf3d'
            # ..., sin(2a)ln(q), cos(2a)ln(q)
            scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
            q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
            alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
            return torch.stack((
                x_center,
                y_center,
                z_center,
                scale / (1 + q),
                scale / (1 + q) * q,
                bbox_pred[:, 5] + bbox_pred[:, 4],
                alpha
            ), dim=-1)

    def _nms(self, bboxes, scores, img_meta):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = pcdet_nms_gpu
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = pcdet_nms_normal_gpu

            nms_ids, _ = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
