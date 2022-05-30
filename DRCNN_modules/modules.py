#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : modules.py
@Date         : 2022/05/30 13:18:48
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : modules for DRCNN
'''
from torchvision.models import vgg16
import torch.nn as nn


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    model = vgg16()

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier






# if __name__ == '__main__':
#     extractor, classifier = decom_vgg16()
#     print(extractor)
#     print(classifier)