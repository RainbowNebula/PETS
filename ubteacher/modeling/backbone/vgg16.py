from abc import ABC

import torch.nn as nn
import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False, weights_path=None):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, class_num)
        )
        if init_weights and weights_path is None:
            self._initialize_weights()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", weights_path=None):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), weights_path=weights_path)
    return model


class VGG_NEW(Backbone):
    def __init__(self, features):
        super(VGG_NEW, self).__init__()
        self.features = features
        self._out_feature_channels = {"p2": 512}
        self._out_feature_strides = {"p2": 16}
        self._out_features = ["p2"]

    def forward(self, x):
        result = self.features(x)
        return {"p2": result}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class VGG_two_feat(Backbone):
    def __init__(self, features):
        super(VGG_two_feat, self).__init__()
        self.features = features
        self._out_feature_channels = {"p1": 512, "p2": 512}
        self._out_feature_strides = {"p1": 16, "p2": 16}
        self._out_features = ["p2", "p1"]

    def forward(self, x):
        feat1 = self.features[:23](x)
        result = self.features[23:](feat1)
        return {"p1": feat1, "p2": result}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_vgg16_backbone(cfg, input_shape: ShapeSpec):
    vgg_feature = vgg(model_name="vgg16",
                      weights_path="/mnt/share/dataset_for_UDA_OD/pretrain-model/vgg16-397923af.pth").features
    vgg_feature = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除feature中最后的maxpool层
    backbone = VGG_two_feat(vgg_feature)
    return backbone
