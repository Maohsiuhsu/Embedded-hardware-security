import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import trunc_normal_
from timm.models import register_model
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from facelivt import (
        Conv2d_BN, GroupNorm, BN_Linear, LoRaConv, Residual, FFN, Classfier,
        RepConv, StemLayer, PatchMerging, MHSA, LinearAttention, Block, Stage
    )
except ImportError:
    print("Warning: Cannot import base components from facelivt.")
    print("Please ensure facelivt.py is in Python path, or refer to full implementation.")
    raise ImportError("Base component definitions from facelivt.py are required")

class FingerVIT_14x14(nn.Module):  
    def __init__(self, in_chans=3, img_size=112,
                 num_classes=512,
                 dims=[96, 192, 384],
                 depths=[2, 4, 6],
                 type=["repmix", "mhsa", "mhsa"],
                 ks_pe=3,
                 patch_size=4, 
                 mlp_ratio=3, 
                 act_layer=nn.GELU, 
                 distillation=False,
                 final_feature_dim=None, 
                 drop_rate=0.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.final_feature_dim = final_feature_dim

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        
        num_stage = len(depths)
        self.num_stage = num_stage

        stages = []
        img_res = img_size//patch_size
        patch_embedds = []
        
        patch_embedds.append(StemLayer(in_chans, dims[0], ps=patch_size, act_layer=act_layer))

        for i_stage in range(num_stage):
            stage = Stage(
                    dim=dims[i_stage],
                    depth=depths[i_stage], 
                    type=type[i_stage],
                    resolution=img_res, 
                    mlp_ratio=mlp_ratio, 
                    act_layer=act_layer,
            )
            stages.append(stage)
            
            if i_stage < (num_stage-1):
                patch_embedd = PatchMerging(dims[i_stage], dims[i_stage+1], ks=ks_pe, act_layer=act_layer)
                patch_embedds.append(patch_embedd)
                if i_stage == 2:
                    pass
                else:
                    img_res = math.ceil(img_res/2)

        self.patch_embedds = nn.Sequential(*patch_embedds)
        self.stages = nn.Sequential(*stages)
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        if self.final_feature_dim is not None:
            if isinstance(self.final_feature_dim, (list, tuple)):
                self.pre_head = nn.Sequential(
                            Conv2d_BN(dims[-1], self.final_feature_dim[0]),
                            nn.AdaptiveAvgPool2d(1)
                            )
            else:
                self.pre_head = nn.AdaptiveAvgPool2d(1)
                self.final_feature_dim=[dims[-1], self.final_feature_dim]

            self.head = nn.Sequential(
                BN_Linear(self.final_feature_dim[0], self.final_feature_dim[1]),
                act_layer(),
                self.head_drop,
                Classfier(self.final_feature_dim[1], num_classes, distillation)
                )
        else:
            self.pre_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), self.head_drop)
            self.head = Classfier(dims[-1], num_classes, distillation)

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feature(self, x):
        for i in range(self.num_stage):
            x = self.patch_embedds[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.pre_head(x).flatten(1)
        x = self.head(x)
        return x

@register_model
def fingervit_14x14_s(num_classes=512, distillation=False, pretrained=True, **kwargs):
    model = FingerVIT_14x14(
        num_classes=num_classes,
        dims=[40, 80, 160],
        depths=[2, 4, 6],
        type=["repmix", "mhsa", "mhsa"],
        final_feature_dim=None, 
        distillation=False,
        **kwargs)
    return model

@register_model
def fingervit_14x14_m(num_classes=512, distillation=False, pretrained=True, **kwargs):
    model = FingerVIT_14x14(
        num_classes=num_classes,
        dims=[64, 128, 256],
        depths=[2, 4, 6],
        type=["repmix", "mhsa", "mhsa"],
        final_feature_dim=None, 
        distillation=False,
        **kwargs)
    return model

@register_model
def fingervit_14x14_l(num_classes=512, distillation=False, pretrained=True, **kwargs):
    model = FingerVIT_14x14(
        num_classes=num_classes,
        dims=[96, 192, 384],
        depths=[2, 4, 6],
        type=["repmix", "mhsa", "mhsa"],
        final_feature_dim=None, 
        distillation=False,
        **kwargs)
    return model

def calculate_14x14_resolution():
    print("FingerVIT 14x14 Resolution Calculation")
    print("="*50)
    
    img_size = 112
    patch_size = 4
    dims = [96, 192, 384]
    depths = [2, 4, 6]
    
    print(f"Input Image Size: {img_size}x{img_size}")
    print(f"Patch Size: {patch_size}")
    print(f"Stage Dimensions: {dims}")
    print(f"Stage Depths: {depths}")
    print("="*50)
    
    img_res = img_size // patch_size
    print(f"Initial Resolution: {img_res}x{img_res}")
    
    current_res = img_res
    print("\nResolution Change Process:")
    print("-" * 30)
    
    for i in range(len(depths)):
        print(f"Stage {i+1}:")
        print(f"  Input Resolution: {current_res}x{current_res}")
        print(f"  Feature Dimension: {dims[i]}")
        print(f"  Depth: {depths[i]}")
        
        if i < len(depths) - 1:
            current_res = math.ceil(current_res / 2)
            print(f"  After PatchMerging: {current_res}x{current_res}")
        else:
            print(f"  Final Resolution: {current_res}x{current_res}")
        print()
    
    print("="*50)
    print(f"Final Feature Map Resolution: {current_res}x{current_res}")
    print(f"Final Feature Dimension: {dims[-1]}")
    print(f"Total Features: {current_res}x{current_res}x{dims[-1]} = {current_res*current_res*dims[-1]}")
    print("="*50)

if __name__ == "__main__":
    calculate_14x14_resolution()
