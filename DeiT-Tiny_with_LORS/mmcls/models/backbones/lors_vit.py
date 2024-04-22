# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, build_activation_layer, Linear
from mmcv.cnn.bricks.transformer import PatchEmbed  # FFN
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS

from mmcls.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import resize_pos_embed, to_2tuple
from .base_backbone import BaseBackbone


class MultiheadAttention_LORS(BaseModule):
    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 kernel_sizes=None,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention_LORS, self).__init__(init_cfg=init_cfg)

        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=False)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=False)
        
        qkv_private = []
        proj_private = []
        bias_per_layer = []
        for n_layer in range(num_layers):
            
            k_size = self.kernel_sizes[n_layer]
            
            qkv_private.append(
                ModuleList(
                    [Linear(self.input_dims, k_size, bias=False),
                        Linear(k_size, self.embed_dims * 3, bias=False)] )
                )
            
            proj_private.append(
                ModuleList(
                    [Linear(self.embed_dims, k_size, bias=False),
                        Linear(k_size, self.embed_dims, bias=False)] )
                )
            
            bias_per_layer.append(nn.ParameterList([
                nn.Parameter(torch.zeros(embed_dims * 3)),
                nn.Parameter(torch.zeros(embed_dims))]))
            
        self.qkv_private = ModuleList(qkv_private)
        self.proj_private = ModuleList(proj_private)
        self.bias_per_layer = ModuleList(bias_per_layer)
        
    def init_weights(self):
        super(MultiheadAttention_LORS, self).init_weights()
                        
        for qkv_private, proj_private, bias_per_layer in zip(
            self.qkv_private, self.proj_private, self.bias_per_layer):
            
            A, B = [p.weight for p in qkv_private]
            nn.init.kaiming_uniform_(A)
            nn.init.zeros_(B)

            A, B = [p.weight for p in proj_private]
            nn.init.kaiming_uniform_(A)
            nn.init.zeros_(B)
            
            for bias in bias_per_layer:
                nn.init.zeros_(bias)

    def forward(self, x, n_layer):
        assert n_layer != None
        common_qkv = self.qkv
        common_proj = self.proj
        
        assert len(self.bias_per_layer[n_layer])==2, \
            'now only 1 bias per linear'

        private_qkv = self.qkv_private[n_layer]
        private_proj = self.proj_private[n_layer]

        B, N, _ = x.shape
        
        base_weight_qkv = common_qkv.weight
        _A_qkv, _B_qkv = [p.weight for p in private_qkv]
        bias_qkv = self.bias_per_layer[n_layer][0]

        qkv = F.linear(x, base_weight_qkv + _B_qkv @ _A_qkv, bias_qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        
        base_weight_proj = common_proj.weight
        _A_proj, _B_proj = [p.weight for p in private_proj]
        bias_proj = self.bias_per_layer[n_layer][1]        

        x = F.linear(x, base_weight_proj + _B_proj @ _A_proj, bias_proj)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
         

class FFN_LORS(BaseModule):
    """ dropout_layer (obj:`ConfigDict`): The dropout_layer used
       when adding the shortcut. """

    def __init__(self,
                 num_layers=None,
                 kernel_sizes=None,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        self.kernel_sizes = kernel_sizes

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Sequential(Linear(feedforward_channels, embed_dims, bias=False),
                                 nn.Dropout(ffn_drop) ))
        self.layers = Sequential(*layers)
        
        layers_private = []
        bias_per_layer = []
        for n_layer in range(num_layers):
            in_channels = embed_dims
            k_size = self.kernel_sizes[n_layer]
            layers = ModuleList()
            biases = nn.ParameterList()
            for _ in range(num_fcs - 1):
                layers.append(
                    ModuleList(
                        [Linear(in_channels, k_size, bias=False),
                         Linear(k_size, feedforward_channels, bias=False)] )
                    )
                biases.append(nn.Parameter(torch.zeros(feedforward_channels)))
                in_channels = feedforward_channels
            layers.append(
                ModuleList(
                    [Linear(feedforward_channels, k_size, bias=False),
                     Linear(k_size, embed_dims, bias=False)])
                )
            biases.append(nn.Parameter(torch.zeros(embed_dims)))

            layers_private.append(layers)
            bias_per_layer.append(biases)
            
        self.layers_private = ModuleList(layers_private)
        self.bias_per_layer = ModuleList(bias_per_layer)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def init_weights(self):
        super(FFN_LORS, self).init_weights()
        ###########################################################################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.bias, std=1e-6)
                
        for private_params, biases in zip(self.layers_private, self.bias_per_layer):
            for private_param, bias in zip(private_params, biases):
                A, B = [p.weight for p in private_param]
                nn.init.kaiming_uniform_(A)
                nn.init.zeros_(B)
                nn.init.zeros_(bias)

    def forward(self, x, identity=None, n_layer=None):
        assert n_layer != None
        
        common_params = self.layers
        private_params = self.layers_private[n_layer]
        biases = self.bias_per_layer[n_layer]

        out = x
        for common_param, private_param, bias in zip(common_params, private_params, biases):
            act_n_dp = common_param[1:]
            base_weight = common_param[0].weight
            _A, _B = [p.weight for p in private_param]
            
            out = F.linear(out, base_weight + _B @ _A, bias)
            out = act_n_dp(out)
                    
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class TransformerEncoderLayerLors(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims=None,
                 num_heads=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayerLors, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayerLors, self).init_weights()

    def forward(self, x, attn_lors, ffn_lors, n_layer):
        x = x + attn_lors(self.norm1(x), n_layer=n_layer)
        x = ffn_lors(self.norm2(x), identity=x, n_layer=n_layer)
        return x


@BACKBONES.register_module()
class VisionTransformerWlors(BaseBackbone):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    # Some structures have multiple extra tokens, like DeiT.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 attn_kernel_sizes=None,
                 attn_kernel_groups=None,
                 ffn_kernel_sizes=None,
                 ffn_kernel_groups=None,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(VisionTransformerWlors, self).__init__(init_cfg)

        self.attn_kernel_sizes = attn_kernel_sizes
        self.attn_kernel_groups = attn_kernel_groups

        self.ffn_kernel_sizes = ffn_kernel_sizes
        self.ffn_kernel_groups = ffn_kernel_groups
        
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        
        attn_kernel_sizes = [i*j for i,j in zip(self.attn_kernel_sizes,
                                           self.attn_kernel_groups)]      
        ffn_kernel_sizes = [i*j for i,j in zip(self.ffn_kernel_sizes,
                                           self.ffn_kernel_groups)]

        self.attn_lors = MultiheadAttention_LORS(
            num_layers=self.num_layers,
            kernel_sizes=attn_kernel_sizes,
            embed_dims=self.embed_dims,
            num_heads=self.arch_settings['num_heads'])

        self.ffn_lors = FFN_LORS(
            num_layers=self.num_layers,
            kernel_sizes=ffn_kernel_sizes, 
            embed_dims=self.embed_dims,
            feedforward_channels=self.
            arch_settings['feedforward_channels'],
            num_fcs=2,
            ffn_drop=0,
            dropout_layer=dict(type='DropPath', drop_prob=0),
            act_cfg=dict(type='GELU'))

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayerLors(
                **_layer_cfg))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(VisionTransformerWlors, self).init_weights()
        ####################################################################
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmcv.utils import print_log
            logger = get_root_logger()
            print_log(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.',
                logger=logger)

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, self.attn_lors, self.ffn_lors, i)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return tuple(outs)