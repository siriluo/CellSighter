# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Any, Optional, Union
from models import create_model
from gat_model import GraphAttentionLayer
from convnext_model import convnextv2_tiny


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'convnextv2_tiny': [convnextv2_tiny, 768],
}


##
# Resnets above
##


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, **kwargs):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(**kwargs)

        # elif head == 'mlp':
        #     self.head = nn.Sequential(
        #         nn.Linear(dim_in, dim_in),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(dim_in, feat_dim)
        #     )

    def forward(self, x):
        feat = self.encoder(x)
        # feat = F.normalize(self.head(feat), dim=1)
        return feat
    
    
class SupConViT(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_base_patch16_224', head='mlp', feat_dim=128, **kwargs):
        super(SupConViT, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(**kwargs)
        vit = timm.create_model(
                'vit_base_patch16_224.mae',
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
            )


    def forward(self, x):
        feat = self.encoder(x)

        return feat
  
  
class MaskBranch(nn.Module):
    # Encodes 2 masks: center-cell + neighbor-cell
    def __init__(self, in_ch: int = 2, out_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, center_mask, neighbor_mask):
        m = torch.cat([center_mask, neighbor_mask], dim=1)  # [B,2,H,W]
        x = self.encoder(m).flatten(1)
        x = self.fc(x)
        return x
    
    
class HEFusedContrastiveModel(nn.Module):
    """
    RGB backbone (DINOv2 ViT-B/14 or ResNet50) + mask branch fusion + SupCon projection head.
    """
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",  # "dinov2_vitb14" | "resnet50" | "resnet18"
        pretrained: bool = True,
        freeze_backbone: bool = False,
        mask_feat_dim: int = 128,
        fusion_dim: int = 512,
        # proj_dim: int = 128,
    ):
        super().__init__()

        # 1) RGB image encoder
        if backbone == "dinov2_vitb14":
            if timm is None:
                raise ImportError("Please install timm: pip install timm")
            # timm model name can vary by timm version; this is the common one:
            self.rgb_encoder = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                num_classes=0,  # return features
                img_size=64,
                dynamic_image_size=True,      
            )
            rgb_dim = self.rgb_encoder.num_features

        elif backbone == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = models.resnet50(weights=w)
            rgb_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.rgb_encoder = m

        elif backbone == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.resnet18(weights=w)
            rgb_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.rgb_encoder = m
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # 2) Mask encoder
        self.mask_branch = MaskBranch(in_ch=2, out_dim=mask_feat_dim)

        # 3) Fusion + projection
        self.fusion = nn.Sequential(
            nn.Linear(rgb_dim + mask_feat_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # self.projector = ProjectionHead(fusion_dim, hidden_dim=fusion_dim, out_dim=proj_dim)

        # Optional classifier head for joint training/eval
        # self.classifier = nn.Linear(fusion_dim, 1)

    def forward(self, x, return_features=False):
        """
        rgb:           [B, 3, H, W]
        center_mask:   [B, 1, H, W]
        neighbor_mask: [B, 1, H, W]
        """
        rgb = x[:, :3]
        neighbor_mask = x[:, 3:4]
        center_mask = x[:, 4:5]
        
        rgb_feat = self.rgb_encoder(rgb)                # [B, rgb_dim]
        mask_feat = self.mask_branch(center_mask, neighbor_mask)  # [B, mask_feat_dim]
        feat = self.fusion(torch.cat([rgb_feat, mask_feat], dim=1))  # [B, fusion_dim]
        # z = self.projector(feat)                        # [B, proj_dim], normalized

        # if return_features:
        #     return {"z": z, "feat": feat, "logit": self.classifier(feat)}
        return feat
    

class PretrainedSupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, num_classes=10, **kwargs):
        super(PretrainedSupConResNet, self).__init__()
        if name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        else:
            self.model = models.resnet18(weights='DEFAULT')
            # self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
        channels = kwargs.get('in_channel', 3)
        self.model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        ##Weights init
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.encoder = nn.Sequential(*list(self.model.children())[:-1])


    def forward(self, x):
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        
        return feat


class SupConGraphResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, **kwargs):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(**kwargs)

        self.gat_layer = GraphAttentionLayer(dim_in, dim_in, dropout=0.6, alpha=0.2)

    def forward(self,         
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,):
        feat = self.encoder(x)

        graph_feats = self.gat_layer(feat, edge_index)

        # feat = F.normalize(self.head(feat), dim=1)
        return feat


# Set up the contrastive learning code separately from the regular training code.
class ProjectionHeadSimp(nn.Module):
    # 2-layer MLP projection head (standard for SupCon)
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)  # normalized embeddings for contrastive loss
    
    
class ProjectionHead(nn.Module):    
    def __init__(self,
        feature_dims=(2048, 128),
        hidden_dim=None,
        activation='gelu',
        use_batch_norm=False,
        normalize_output=True,
        dropout=0.1,
        **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        hidden_dim = hidden_dim or feature_dims[0]
        self.normalize_output = normalize_output

        act = nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(feature_dims[0], hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.act = act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, feature_dims[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.normalize_output:
            x = F.normalize(x, dim=1)
        return x

class ClassificationHead2(nn.Module):
    def __init__(self, in_dim, n_classes, p=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, n_classes),
        )

    def forward(self, feat):
        x = self.classifier(feat)
        return x   # logits


class ClassificationHead(nn.Module):
    def __init__(self,
        # input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.5,
        name='resnet50',
        **kwargs):
        super(ClassificationHead, self).__init__(**kwargs)
        _, feat_dim = model_dict[name]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x


class ContrastiveModel(nn.Module):
    def __init__(self, 
                 base_model: str, 
                 encoder_kwargs=None, 
                 projection_head_kwargs=None, 
                 classification_head_kwargs=None,
                 norm_proj_head_input=False,
                 norm_class_head_input=False,
                 model_name='resnet18',
                 pretrained=False,):
        super(ContrastiveModel, self).__init__()

        self.norm_proj_head_input = norm_proj_head_input
        self.norm_class_head_input = norm_class_head_input

        assert encoder_kwargs is not None
        projection_head_kwargs = projection_head_kwargs or {}
        # classification_head_kwargs = classification_head_kwargs or {}

        if base_model == 'resnet':
            if not pretrained:
                self.encoder = SupConResNet(name=model_name, **encoder_kwargs) # resnet18 currently used resnet50, try out resnet18 next
            else:
                self.encoder = PretrainedSupConResNet(name=model_name, **encoder_kwargs)
        elif base_model == 'convnext':
            self.encoder = convnextv2_tiny(**encoder_kwargs) # the output features should have size (B, 768, 7, 7), so, just 768?
        elif base_model == 'new_fused':
            # pretrained: bool = True,
            # freeze_backbone: bool = False,
            # mask_feat_dim: int = 128,
            # fusion_dim: int = 512,
            # **encoder_kwargs, 
            self.encoder = HEFusedContrastiveModel(backbone='resnet50', mask_feat_dim=128, fusion_dim=512) 
        else:
            raise ValueError(f'encoder model: {base_model} not recognized.')
        self.projection_head = ProjectionHead(**projection_head_kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_embedding = self.encoder(x)
        normalized_embedding = F.normalize(feature_embedding, dim=1)

        projection_in = normalized_embedding if self.norm_proj_head_input else feature_embedding

        projection_out = self.projection_head(projection_in)

        return feature_embedding, normalized_embedding, projection_out #, class_out



