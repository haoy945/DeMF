import warnings
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import build_transformer_layer

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, cfg):
        super().__init__()
        input_channel = cfg['input_channel']
        num_pos_feats = cfg['num_pos_feats']
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@TRANSFORMER_LAYER.register_module()
class TransformerDecoderLayerWithPos(nn.Module):
    def __init__(self, *args, transformerlayers=None, posembed=None, **kwargs):
        super().__init__()
        self.layer = build_transformer_layer(transformerlayers)
        self.posembed = PositionEmbeddingLearned(posembed)

    def init_weights(self):
        """Initialize the weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self,
                query,
                query_pos,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * \
                valid_ratios[:, None]

        query_pos_embed = self.posembed(query_pos)
        query_pos_embed = query_pos_embed.permute(2, 0, 1)

        output = self.layer(
            query,
            *args,
            query_pos=query_pos_embed,
            reference_points=reference_points_input,
            **kwargs)
        
        return output


@TRANSFORMER_LAYER.register_module()
class TransformerDecoderLayerWithoutPos(nn.Module):
    def __init__(self, *args, transformerlayers=None, **kwargs):
        super().__init__()
        self.layer = build_transformer_layer(transformerlayers)

    def init_weights(self):
        """Initialize the weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * \
                valid_ratios[:, None]

        output = self.layer(
            query,
            *args,
            reference_points=reference_points_input,
            **kwargs)
        
        return output
