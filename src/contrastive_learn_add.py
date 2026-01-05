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
class ProjectionHead(nn.Module):    
    def __init__(self,
        feature_dims=(2048, 128),
        activation=nn.ReLU(),
        # kernel_initializer=tf.random_normal_initializer(stddev=.01),
        # bias_initializer=tf.zeros_initializer(),
        use_batch_norm=False,
        normalize_output=True,
    #    batch_norm_momentum=blocks.BATCH_NORM_MOMENTUM,
    #    use_batch_norm_beta=False,
    #    use_global_batch_norm=True,
        **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)

        self.fc1 = nn.Linear(feature_dims[0], feature_dims[0])
        self.fc2 = nn.Linear(feature_dims[0], feature_dims[1])
        self.activation = activation
        self.batch_norm = use_batch_norm
        self.dropout = nn.Dropout(p=0.1) if use_batch_norm else None

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(feature_dims[0])
            # self.bn2 = nn.BatchNorm1d(feature_dims[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        # if self.batch_norm:
        #     x = self.bn2(x)

        # Just use l2 normalization at the end.
        x = F.normalize(x, dim=1)

        return x


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


# class LinearClassifier(nn.Module):
#     """Linear classifier"""
#     def __init__(self, name='resnet50', num_classes=12):
#         super(LinearClassifier, self).__init__()
#         _, feat_dim = model_dict[name]
#         self.fc = nn.Linear(feat_dim, num_classes)

#     def forward(self, features):
#         return self.fc(features)


class ContrastiveModel(nn.Module):
    def __init__(self, 
                 base_model: str, 
                 encoder_kwargs=None, 
                 projection_head_kwargs=None, 
                 classification_head_kwargs=None,
                 norm_proj_head_input=False,
                 norm_class_head_input=False,
                 model_name='resnet18'):
        super(ContrastiveModel, self).__init__()

        self.norm_proj_head_input = norm_proj_head_input
        self.norm_class_head_input = norm_class_head_input

        assert encoder_kwargs is not None
        projection_head_kwargs = projection_head_kwargs or {}
        classification_head_kwargs = classification_head_kwargs or {}

        # self.encoder = create_model(**encoder_kwargs)
        if base_model == 'resnet':
            self.encoder = SupConResNet(name=model_name, **encoder_kwargs) # resnet18 currently used resnet50, try out resnet18 next
        elif base_model == 'convnext':
            self.encoder = convnextv2_tiny(**encoder_kwargs) # the output features should have size (B, 768, 7, 7), so, just 768?
        else:
            raise ValueError(f'encoder model: {base_model} not recognized.')
        self.projection_head = ProjectionHead(**projection_head_kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        feature_embedding = self.encoder(x)
        normalized_embedding = F.normalize(feature_embedding, dim=1)

        projection_in = normalized_embedding if self.norm_proj_head_input else feature_embedding

        projection_out = self.projection_head(projection_in)

        return feature_embedding, normalized_embedding, projection_out #, class_out



