from audioop import bias
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from nnunetv2.network_architecture.init import InitWeights
from einops import rearrange
from .swin_3D import *


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels,kernel_size = [3,3,3],padding=[1,1,1])
        self.conv_s = nn.Conv3d(output_channels, output_channels,kernel_size = [1,3,3],padding=[0,1,1])
        self.conv_h = nn.Conv3d(output_channels, output_channels,kernel_size = [3,1,3],padding=[1,0,1])
        self.conv_w = nn.Conv3d(output_channels, output_channels,kernel_size = [3,3,1],padding=[1,1,0])

        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs[
                'p'] > 0:
        
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm1 = norm_op(output_channels, **norm_op_kwargs)
        self.instnorm2 = norm_op(output_channels, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin_kwargs != None else nonlin() 
        
        self.rd_conv = nn.Conv3d(input_channels, output_channels,kernel_size = [1,1,1]) if input_channels != output_channels else None
        

    def forward(self, x):

        res = x
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.nonlin(self.instnorm1(x))
        x_s = self.conv_s(x)
        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        if self.dropout is not None:
            x_s = self.dropout(x_s)
            x_h = self.dropout(x_h)
            x_w = self.dropout(x_w)
        if self.rd_conv is not None:
            res = self.rd_conv(res)
        x = self.nonlin(self.instnorm2(x_s+x_h+x_w+res))
        # x = self.nonlin(self.instnorm2(x_s+x_h+x_w))

        return x


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        
        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs[
                'p'] > 0:
        
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu = nonlin(**nonlin_kwargs) if nonlin_kwargs != None else nonlin() 

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class DeepSupervision(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.proj = nn.Conv3d(
            dim, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self, num_stage, only_conv, num_only_conv_stage, num_pool, base_num_features, dim, input_resolution, depth, num_heads,
                 window_size, image_channels=1, num_conv_per_stage=2, conv_op=None,
                 norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, basic_block=ConvBlock, max_num_features=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, down_or_upsample=None, feat_map_mul_on_downscale=2, is_encoder=True,use_checkpoint=False):

        super().__init__()
        self.num_stage = num_stage
        self.only_conv = only_conv
        self.num_only_conv_stage = num_only_conv_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder
        self.image_channels = image_channels
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
       
        
        input_features = dim if is_encoder else 2*dim 
        
        # self.depth = depth
        conv_kwargs['kernel_size'] = conv_kernel_sizes[num_stage]
        conv_kwargs['padding'] = conv_pad_sizes[num_stage]

    
        input_du_channels = min(int(base_num_features * feat_map_mul_on_downscale ** (num_stage-1 if is_encoder else num_stage+1)),
                                      max_num_features)
        output_du_channels = dim
        if self.is_encoder and self.num_stage==0:
            self.frist_conv = conv_op(image_channels, dim, kernel_size=1, stride=1, bias=True)
        else: 
            self.frist_conv = None
        self.conv_blocks = nn.Sequential(
            *([basic_block(input_features, dim, conv_op,
                           conv_kwargs,
                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                           nonlin, nonlin_kwargs)] +
              [basic_block(dim, dim, conv_op,
                           conv_kwargs,
                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                           nonlin, nonlin_kwargs) for _ in range(num_conv_per_stage - 1)]))

        # build blocks
        if num_stage >= num_only_conv_stage and not only_conv:
            self.swin_blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=[0, 0, 0] if (i % 2 == 0) else [
                                         window_size[0] // 2, window_size[1] // 2, window_size[2] // 2],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(
                                         drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])
        
        # patch merging layer
        if down_or_upsample is not None:
            dowm_stage = num_stage-1 if is_encoder else num_stage
            self.down_or_upsample = nn.Sequential(down_or_upsample(input_du_channels, output_du_channels, pool_op_kernel_sizes[dowm_stage],
                                                                   pool_op_kernel_sizes[dowm_stage], bias=False),
                                                  norm_op(output_du_channels, **norm_op_kwargs)
                                                  )
        else:
            self.down_or_upsample = None

    
        
    def forward(self, x, skip):
        if self.frist_conv is not None:
            x = self.frist_conv(x) 
        if self.down_or_upsample is not None:
            x = self.down_or_upsample(x)
        s = x
        if not self.is_encoder and self.num_stage < self.num_pool:
            x = torch.cat((x, skip), dim=1)
            
        x = self.conv_blocks(x)
        if self.num_stage >= self.num_only_conv_stage and not self.only_conv:
            if not self.is_encoder and self.num_stage < self.num_pool:
                s = s + skip
            for tblk in self.swin_blocks:
                if self.use_checkpoint:
                    s = checkpoint.checkpoint(tblk, s)
                else:
                    s = tblk(s)
            x = x + s
        return x
class PHTrans(nn.Module):
    def __init__(self, img_size, base_num_features, num_classes, image_channels=1, only_conv=False, num_only_conv_stage=2, num_conv_per_stage=2,num_transformer_per_stage = 2,
                 feat_map_mul_on_downscale=2,  pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, deep_supervision=True, depths=None, num_heads=None,
                 window_size=None, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., dropout_p=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.conv_op = nn.Conv3d
        
        norm_op = nn.InstanceNorm3d 
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout3d 
        dropout_op_kwargs = {'p': dropout_p, 'inplace': True}
        nonlin = nn.GELU
        nonlin_kwargs = None
        # nonlin = nn.LeakyReLU 
        # nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        max_num_features = base_num_features*10
        self.deep_supervision = deep_supervision
        
        self.num_pool = len(pool_op_kernel_sizes)
        window_size = torch.tensor(img_size)
        for i in pool_op_kernel_sizes:
            window_size = window_size//torch.tensor(i)

        depths = np.ones(len(conv_kernel_sizes)-num_only_conv_stage,dtype = np.uint8)*num_transformer_per_stage
        num_heads = np.ones(len(conv_kernel_sizes)-num_only_conv_stage,dtype = np.uint8)

        for i,head in enumerate(num_heads):
            num_heads[i] = head*2*2**i
        conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]
        
        self.seg_outputs=[]
        for ds in range(self.num_pool):
            self.seg_outputs.append(DeepSupervision(min((base_num_features * feat_map_mul_on_downscale **ds), max_num_features), self.num_classes))
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
      
        # build layers
        self.down_layers = nn.ModuleList()
        for i_layer in range(self.num_pool+1):  # 0,1,2,3,4,5
            layer = BasicLayer(num_stage=i_layer, 
                               only_conv = only_conv,
                               num_only_conv_stage=num_only_conv_stage, 
                               num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               dim = min((base_num_features * feat_map_mul_on_downscale ** i_layer), max_num_features),
                               input_resolution=(
                                   img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),
                                #    if i_layer > 0 else img_size,  # 56,28,14,7
                               depth=depths[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer-num_only_conv_stage]):sum(depths[:i_layer-num_only_conv_stage + 1])] if (
                                   i_layer >= num_only_conv_stage) else None,  
                               norm_layer=norm_layer,
                               down_or_upsample=nn.Conv3d if i_layer > 0 else None,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint,
                               is_encoder=True)
            self.down_layers.append(layer)
        self.up_layers = nn.ModuleList()
        for i_layer in range(self.num_pool)[::-1]:
            layer = BasicLayer(num_stage=i_layer, 
                               only_conv = only_conv,
                               num_only_conv_stage=num_only_conv_stage, 
                               num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               dim = min((base_num_features * feat_map_mul_on_downscale ** i_layer), max_num_features),
                               input_resolution=(
                                   img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)), 
                               depth=depths[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer-num_only_conv_stage] if (
                                   i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer-num_only_conv_stage]):sum(depths[:i_layer-num_only_conv_stage + 1])] if (
                                   i_layer >= num_only_conv_stage) else None,  
                               norm_layer=norm_layer,
                               down_or_upsample=nn.ConvTranspose3d,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint,
                               is_encoder=False)
            self.up_layers.append(layer)
        self.apply(InitWeights)


    def forward(self, x):
      
        x_skip = list()
        for i, layer in enumerate(self.down_layers):
            x = layer(x, None)  
            if i < self.num_pool:
                x_skip.append(x)
        out = []
        for i, layer in enumerate(self.up_layers):
            x = layer(x, x_skip[-(i+1)]) 
            if self.deep_supervision:
                out.append(x)
        if self.deep_supervision:
            ds = []
            for i in range(len(out)):
                ds.append(self.seg_outputs[i](out[-(i+1)])) 
        else:
            ds = self.seg_outputs[0](x)
        

        return ds

       
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}  
