import timm

import torch
import torch.nn as nn

from tllib.alignment.dann import ImageClassifier


def get_model(config):
    """
    vitb16
    """

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.out_features = model.head.in_features
    model.head = nn.Identity()

    pool_layer = torch.nn.Identity()
    classifier = ImageClassifier(model, config['num_classes'], pool_layer=pool_layer, bottleneck_dim=config['bottleneck']).cuda()


    return classifier