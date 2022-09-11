import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import deprecated_api_warning
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetV2(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # [num_heads, num_levels, num_points, 2]
        self.grid = self.set_grid()

        self.init_weights()

    def set_grid(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        return grid_init

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, val=0., bias=0.)
        # thetas = torch.arange(
        #     self.num_heads,
        #     dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init /
        #              grid_init.abs().max(-1, keepdim=True)[0]).view(
        #                  self.num_heads, 1, 1,
        #                  2).repeat(1, self.num_levels, self.num_points, 1)
        # for i in range(self.num_points):
        #     grid_init[:, :, i, :] *= i + 1

        # self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offset(self, reference_points, img_features, spatial_shapes):
        sampling_feat_list = []
        bs, n_query, _, _ = reference_points.shape

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        self.grid = self.grid.to(offset_normalizer.device)
        # [num_heads, num_levels, num_points, 2]
        grid = self.grid / offset_normalizer[None, :, None, :]
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_grid = reference_points[:, :, None, :, None, :] + grid[None, None, ...]
        sampling_grid = 2 * sampling_grid - 1
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        sampling_grid = sampling_grid.view(
            bs, n_query*self.num_heads, self.num_levels, self.num_points, 2
        )

        # 首先利用reference_points以及grid来采样feature
        for level, img_features_l in enumerate(img_features):
            # [bs, C, num_query * num_heads, num_points]
            sampling_feat_l_ = F.grid_sample(
                img_features_l,
                sampling_grid[:, :, level],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_feat_list.append(sampling_feat_l_)
        # [bs, C, num_query * num_heads, num_levels, num_points]
        sampling_feat = torch.stack(sampling_feat_list, dim=-2)
        # [bs, num_query * num_heads, num_levels, num_points, C]
        sampling_feat = sampling_feat.permute(0, 2, 3, 4, 1)
        
        # 然后用采样后的feature来计算offset
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        relative_offset = self.sampling_offsets(sampling_feat)
        relative_offset = relative_offset.view(
            bs, n_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # 最后将offset与grid想加得到最终的offset
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        offset = self.grid[None, None] + relative_offset
        return offset

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets = self.get_sampling_offset(reference_points, img_features, spatial_shapes)
        # sampling_offsets = self.sampling_offsets(query).view(
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetV2LearnGrid(MultiScaleDeformableAttentionOffsetV2):
    def set_grid(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        # [num_heads, num_levels, num_points, 2]
        grid = nn.Embedding(self.num_heads, self.num_levels * self.num_points * 2)
        grid.weight.data = grid_init.view(self.num_heads, self.num_levels * self.num_points * 2)
        return grid

    def get_sampling_offset(self, reference_points, img_features, spatial_shapes):
        sampling_feat_list = []
        bs, n_query, _, _ = reference_points.shape

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # [num_heads, num_levels, num_points, 2]
        grid_ = self.grid.weight.view(self.num_heads, self.num_levels, self.num_points, 2)
        grid = grid_ / offset_normalizer[None, :, None, :]
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_grid = reference_points[:, :, None, :, None, :] + grid[None, None, ...]
        sampling_grid = 2 * sampling_grid - 1
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        sampling_grid = sampling_grid.view(
            bs, n_query*self.num_heads, self.num_levels, self.num_points, 2
        )

        # 首先利用reference_points以及grid来采样feature
        for level, img_features_l in enumerate(img_features):
            # [bs, C, num_query * num_heads, num_points]
            sampling_feat_l_ = F.grid_sample(
                img_features_l,
                sampling_grid[:, :, level],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_feat_list.append(sampling_feat_l_)
        # [bs, C, num_query * num_heads, num_levels, num_points]
        sampling_feat = torch.stack(sampling_feat_list, dim=-2)
        # [bs, num_query * num_heads, num_levels, num_points, C]
        sampling_feat = sampling_feat.permute(0, 2, 3, 4, 1)
        
        # 然后用采样后的feature来计算offset
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        relative_offset = self.sampling_offsets(sampling_feat)
        relative_offset = relative_offset.view(
            bs, n_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # 最后将offset与grid想加得到最终的offset
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        offset = grid_[None, None] + relative_offset
        return offset


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetV4(BaseModule):
    '''
    采用learned grid，同时在预测offset时是一个head的feat拼接在一起同时预测。
    '''
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims * self.num_levels * self.num_points,
                                          self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # [num_heads, num_levels, num_points, 2]
        self.grid = self.set_grid()

        self.init_weights()

    def set_grid(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        # [num_heads, num_levels, num_points, 2]
        grid = nn.Embedding(self.num_heads, self.num_levels * self.num_points * 2)
        grid.weight.data = grid_init.view(self.num_heads, self.num_levels * self.num_points * 2)
        return grid

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offset(self, reference_points, img_features, spatial_shapes):
        sampling_feat_list = []
        bs, n_query, _, _ = reference_points.shape

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # [num_heads, num_levels, num_points, 2]
        grid_ = self.grid.weight.view(self.num_heads, self.num_levels, self.num_points, 2)
        grid = grid_ / offset_normalizer[None, :, None, :]
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_grid = reference_points[:, :, None, :, None, :] + grid[None, None, ...]
        sampling_grid = 2 * sampling_grid - 1
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        sampling_grid = sampling_grid.view(
            bs, n_query*self.num_heads, self.num_levels, self.num_points, 2
        )

        # 首先利用reference_points以及grid来采样feature
        for level, img_features_l in enumerate(img_features):
            # [bs, C, num_query * num_heads, num_points]
            sampling_feat_l_ = F.grid_sample(
                img_features_l,
                sampling_grid[:, :, level],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_feat_list.append(sampling_feat_l_)
        # [bs, C, num_query * num_heads, num_levels, num_points]
        sampling_feat = torch.stack(sampling_feat_list, dim=-2)
        # [bs, num_query * num_heads, num_levels, num_points, C]
        sampling_feat = sampling_feat.permute(0, 2, 3, 4, 1)
        # [bs, num_query * num_heads, num_levels * num_points * C]
        sampling_feat = sampling_feat.contiguous().view(
            bs, n_query * self.num_heads, self.num_levels * self.num_points * self.embed_dims
        )
        
        # 然后用采样后的feature来计算offset
        # [bs, num_query * num_heads, num_levels * num_points * 2]
        relative_offset = self.sampling_offsets(sampling_feat)
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        relative_offset = relative_offset.view(
            bs, n_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # 最后将offset与grid想加得到最终的offset
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        offset = grid_[None, None] + relative_offset
        return offset

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets = self.get_sampling_offset(reference_points, img_features, spatial_shapes)
        # sampling_offsets = self.sampling_offsets(query).view(
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetV5(BaseModule):
    '''
    采用learned grid，同时在预测offset时是一个head的feat拼接在一起同时预测，同时attention weight也是这么同时预测。
    '''
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims * self.num_levels * self.num_points,
                                          self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * self.num_levels * self.num_points,
                                           num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # [num_heads, num_levels, num_points, 2]
        self.grid = self.set_grid()

        self.init_weights()

    def set_grid(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        # [num_heads, num_levels, num_points, 2]
        grid = nn.Embedding(self.num_heads, self.num_levels * self.num_points * 2)
        grid.weight.data = grid_init.view(self.num_heads, self.num_levels * self.num_points * 2)
        return grid

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offset_and_weight(self, reference_points, img_features, spatial_shapes):
        sampling_feat_list = []
        bs, n_query, _, _ = reference_points.shape

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # [num_heads, num_levels, num_points, 2]
        grid_ = self.grid.weight.view(self.num_heads, self.num_levels, self.num_points, 2)
        grid = grid_ / offset_normalizer[None, :, None, :]
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_grid = reference_points[:, :, None, :, None, :] + grid[None, None, ...]
        sampling_grid = 2 * sampling_grid - 1
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        sampling_grid = sampling_grid.view(
            bs, n_query*self.num_heads, self.num_levels, self.num_points, 2
        )

        # 首先利用reference_points以及grid来采样feature
        for level, img_features_l in enumerate(img_features):
            # [bs, C, num_query * num_heads, num_points]
            sampling_feat_l_ = F.grid_sample(
                img_features_l,
                sampling_grid[:, :, level],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_feat_list.append(sampling_feat_l_)
        # [bs, C, num_query * num_heads, num_levels, num_points]
        sampling_feat = torch.stack(sampling_feat_list, dim=-2)
        # [bs, num_query * num_heads, num_levels, num_points, C]
        sampling_feat = sampling_feat.permute(0, 2, 3, 4, 1)
        # [bs, num_query * num_heads, num_levels * num_points * C]
        sampling_feat = sampling_feat.contiguous().view(
            bs, n_query * self.num_heads, self.num_levels * self.num_points * self.embed_dims
        )
        
        # 然后用采样后的feature来计算offset
        # [bs, num_query * num_heads, num_levels * num_points * 2]
        relative_offset = self.sampling_offsets(sampling_feat)
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        relative_offset = relative_offset.view(
            bs, n_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # [bs, num_query * num_heads, num_levels * num_points]
        attention_weight = self.attention_weights(sampling_feat)
        # [bs, num_query, num_heads, num_levels * num_points]
        attention_weight = attention_weight.view(
            bs, n_query, self.num_heads, self.num_levels * self.num_points
        )

        # 最后将offset与grid想加得到最终的offset
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        offset = grid_[None, None] + relative_offset
        return offset, attention_weight

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets, attention_weights = self.get_sampling_offset_and_weight(reference_points, img_features, spatial_shapes)
        # sampling_offsets = self.sampling_offsets(query).view(
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # attention_weights = self.attention_weights(query).view(
        #     bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetV3(BaseModule):
    '''
    这一版是指attention weight也采用img feat来进行预测。
    '''
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, 2)
        self.attention_weights = nn.Linear(embed_dims, 1)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # [num_heads, num_levels, num_points, 2]
        self.grid = self.set_grid()

        self.init_weights()

    def set_grid(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        return grid_init

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, val=0., bias=0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offset_and_weight(self, reference_points, img_features, spatial_shapes):
        sampling_feat_list = []
        bs, n_query, _, _ = reference_points.shape

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        self.grid = self.grid.to(offset_normalizer.device)
        # [num_heads, num_levels, num_points, 2]
        grid = self.grid / offset_normalizer[None, :, None, :]
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_grid = reference_points[:, :, None, :, None, :] + grid[None, None, ...]
        sampling_grid = 2 * sampling_grid - 1
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        sampling_grid = sampling_grid.view(
            bs, n_query*self.num_heads, self.num_levels, self.num_points, 2
        )

        # 首先利用reference_points以及grid来采样feature
        for level, img_features_l in enumerate(img_features):
            # [bs, C, num_query * num_heads, num_points]
            sampling_feat_l_ = F.grid_sample(
                img_features_l,
                sampling_grid[:, :, level],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_feat_list.append(sampling_feat_l_)
        # [bs, C, num_query * num_heads, num_levels, num_points]
        sampling_feat = torch.stack(sampling_feat_list, dim=-2)
        # [bs, num_query * num_heads, num_levels, num_points, C]
        sampling_feat = sampling_feat.permute(0, 2, 3, 4, 1)
        
        # 然后用采样后的feature来计算offset与weight
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        relative_offset = self.sampling_offsets(sampling_feat)
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        relative_offset = relative_offset.view(
            bs, n_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # [bs, num_query * num_heads, num_levels, num_points, 1]
        attention_weight = self.attention_weights(sampling_feat)
        # [bs, num_query, num_heads, num_levels*num_points]
        attention_weight = attention_weight.view(
            bs, n_query, self.num_heads, self.num_levels * self.num_points
        )

        # 最后将offset与grid想加得到最终的offset
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        offset = self.grid[None, None] + relative_offset
        return offset, attention_weight

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets, attention_weights =\
             self.get_sampling_offset_and_weight(reference_points, img_features, spatial_shapes)
        # sampling_offsets = self.sampling_offsets(query).view(
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # attention_weights = self.attention_weights(query).view(
        #     bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetV3LearnGrid(MultiScaleDeformableAttentionOffsetV3):
    def set_grid(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        # [num_heads, num_levels, num_points, 2]
        grid = nn.Embedding(self.num_heads, self.num_levels * self.num_points * 2)
        grid.weight.data = grid_init.view(self.num_heads, self.num_levels * self.num_points * 2)
        return grid

    def get_sampling_offset_and_weight(self, reference_points, img_features, spatial_shapes):
        sampling_feat_list = []
        bs, n_query, _, _ = reference_points.shape

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # [num_heads, num_levels, num_points, 2]
        grid_ = self.grid.weight.view(self.num_heads, self.num_levels, self.num_points, 2)
        grid = grid_ / offset_normalizer[None, :, None, :]
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_grid = reference_points[:, :, None, :, None, :] + grid[None, None, ...]
        sampling_grid = 2 * sampling_grid - 1
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        sampling_grid = sampling_grid.view(
            bs, n_query*self.num_heads, self.num_levels, self.num_points, 2
        )

        # 首先利用reference_points以及grid来采样feature
        for level, img_features_l in enumerate(img_features):
            # [bs, C, num_query * num_heads, num_points]
            sampling_feat_l_ = F.grid_sample(
                img_features_l,
                sampling_grid[:, :, level],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_feat_list.append(sampling_feat_l_)
        # [bs, C, num_query * num_heads, num_levels, num_points]
        sampling_feat = torch.stack(sampling_feat_list, dim=-2)
        # [bs, num_query * num_heads, num_levels, num_points, C]
        sampling_feat = sampling_feat.permute(0, 2, 3, 4, 1)
        
        # 然后用采样后的feature来计算offset与weight
        # [bs, num_query * num_heads, num_levels, num_points, 2]
        relative_offset = self.sampling_offsets(sampling_feat)
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        relative_offset = relative_offset.view(
            bs, n_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # [bs, num_query * num_heads, num_levels, num_points, 1]
        attention_weight = self.attention_weights(sampling_feat)
        # [bs, num_query, num_heads, num_levels*num_points]
        attention_weight = attention_weight.view(
            bs, n_query, self.num_heads, self.num_levels * self.num_points
        )

        # 最后将offset与grid想加得到最终的offset
        # [bs, num_query, num_heads, num_levels, num_points, 2]
        offset = grid_[None, None] + relative_offset
        return offset, attention_weight


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffset(BaseModule):
    """An attention module used in Deformable-Detr."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffset1(MultiScaleDeformableAttentionOffset):
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetAdd(MultiScaleDeformableAttentionOffset):
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        offset_features = offset_features + query
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetAddLinear(MultiScaleDeformableAttentionOffsetAdd):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.transform = nn.Linear(embed_dims, embed_dims)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        offset_features = self.transform(offset_features) + query
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetCat(MultiScaleDeformableAttentionOffset):
    """An attention module used in Deformable-Detr."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims * 2, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * 2,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        offset_features = torch.cat([offset_features, query], dim=-1)
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetCatLinear(MultiScaleDeformableAttentionOffsetCat):
    """An attention module used in Deformable-Detr."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.transform = nn.Linear(embed_dims, embed_dims)
        self.sampling_offsets = nn.Linear(
            embed_dims * 2, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * 2,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        offset_features = self.transform(offset_features)
        offset_features = torch.cat([offset_features, query], dim=-1)
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    
@ATTENTION.register_module()
class MultiScaleDeformableAttentionOffsetCat1(MultiScaleDeformableAttentionOffsetCat):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims * 2, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                offset_features=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # offset_features = kwargs['offset_features']
        if offset_features is None:
            offset_features = query
        offset_features = torch.cat([offset_features, query], dim=-1)
        sampling_offsets = self.sampling_offsets(offset_features).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
