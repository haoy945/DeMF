import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmdet3d.models import FUSION_LAYERS
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.models.fusion_layers import (apply_3d_transformation,
                                          coord_2d_transform)

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


@HEADS.register_module()
class DeformableDetrEncoder(BaseModule):
    def __init__(self, 
                 encoder=None, 
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 num_feature_levels=4,
                 embed_dims=256,
                 init_cfg=None):
        super(DeformableDetrEncoder, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        
        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dims))
        
    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        
    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
        
    def forward(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        mlvl_feats_encoder = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    mlvl_positional_encodings,
            )
        return mlvl_feats_encoder
        
    def transformer(self,
                    mlvl_feats,
                    mlvl_masks,
                    mlvl_pos_embeds,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        memory = memory.permute(1, 2, 0)
        
        encoder_outputs = []
        start = end = 0
        for spatial_shape in spatial_shapes:
            h, w = spatial_shape
            end = end + h * w
            feat = memory[:, :, start:end]
            encoder_outputs.append(feat.view(bs, c, h, w))
            start = end

        return encoder_outputs


@FUSION_LAYERS.register_module()
class ImCALayer(nn.Module):
    '''
    将image的feature融入到point的feature当中，其中融合的方式为deformable cross attention，
    reference point为point对应的2d投射点。
    网络的输入包括point，point feature，image feature，以及将3d与2d相对应所必要的输入。
    '''
    def __init__(self, decoder=None):
        super().__init__()
        self.decoder = build_transformer_layer_sequence(decoder)
        
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
        '''
        每一个point都会有3d空间中对应的三维坐标，同时我们也可以知道从三维空间到二维图像的映射，
        因此我们可以将每个point的三维坐标投影到二维图像当中得到对应的二维坐标，然后将这个二维
        坐标作为reference_point。
        '''
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
        
    def forward(self, img_features, seeds_3d, seed_3d_features, img_metas):
        '''
        seeds_3d: BxNx3
        seed_3d_features: BxNxC
        '''
        # get masks
        batch_size = img_features[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = img_features[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        
        mlvl_masks = []
        for feat in img_features:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
        
        # get query and query pos
        query = seed_3d_features
        query_pos = None
        
        # get_reference_points
        reference_points = self.get_reference_points(seeds_3d, img_metas)
        
        # transformer
        feats_out = self.transformer(
            img_features, mlvl_masks, query, query_pos, reference_points)
        
        return feats_out
    
    def transformer(self,
                    mlvl_feats,
                    mlvl_masks,
                    query,
                    query_pos,
                    reference_points,
                    **kwargs):
        '''
        Args:
            mlvl_feats (list[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                [N, C, H, W].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [N, H, W].
            query (Tensor): The query for decoder, with shape
                [N_query, C].
            query_pos (Tensor): The query_pos for decoder, with 
                shape [N_query, C].
        
        Decoder的输入需要以下几个:
            query: seed_3d_features
            value: img_features
            query_pos: 为query的位置编码，这个需要视情况而定
            key_padding_mask: mask_flatten
            reference_points: 可以尝试的是直接利用3d的点云坐标映射到2d图像中得到
            spatial_shapes: 不同stage的feature的大小
            level_start_index: 每一个stage的feature在img_features中的起始位置
            valid_ratios: img_features中非padding部分的占比
        '''
        
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
        
        # decoder
        query = query.permute(1, 0, 2)
        feat_flatten = feat_flatten.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        inter_states, _ = self.decoder(
            query=query,  # [N_query, BS, C_query]
            key=None,
            value=feat_flatten,  # [N_value, BS, C_value]
            query_pos=query_pos,  # [N_query, BS, C_query]
            key_padding_mask=mask_flatten,  # [BS, N_value]
            reference_points=reference_points,  # [BS, N_query, 2]
            spatial_shapes=spatial_shapes,  # [N_lvl, 2]
            level_start_index=level_start_index,  # [N_lvl]
            valid_ratios=valid_ratios,  # [BS, N_lvl, 2]
            **kwargs)
        
        feats_out = inter_states[-1].permute(1, 2, 0)
        return feats_out


@FUSION_LAYERS.register_module()
class ImCALayerWithInit(ImCALayer):
    def init_weights(self):
        """Initialize the weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()


@FUSION_LAYERS.register_module()
class ImCALayerAux(ImCALayerWithInit):
    def transformer(self,
                    mlvl_feats,
                    mlvl_masks,
                    query,
                    query_pos,
                    reference_points,
                    **kwargs):
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
        
        # decoder
        query = query.permute(1, 0, 2)
        feat_flatten = feat_flatten.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        inter_states, _ = self.decoder(
            query=query,  # [N_query, BS, C_query]
            key=None,
            value=feat_flatten,  # [N_value, BS, C_value]
            query_pos=query_pos,  # [N_query, BS, C_query]
            key_padding_mask=mask_flatten,  # [BS, N_value]
            reference_points=reference_points,  # [BS, N_query, 2]
            spatial_shapes=spatial_shapes,  # [N_lvl, 2]
            level_start_index=level_start_index,  # [N_lvl]
            valid_ratios=valid_ratios,  # [BS, N_lvl, 2]
            **kwargs)

        feats_out = inter_states.permute(0, 2, 3, 1)
        return feats_out
