from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from libs.bps.basic import _BaseWrapper

class BackPropagation(_BaseWrapper):

    def __init__(self, extractor, classifier):
        super(BackPropagation, self).__init__(extractor, classifier)

        self.grad_pool = {}
        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook
        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(save_grads(module[0])))

    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self, target_layer):
        
        if target_layer == "image":
            gradient = self.image.grad.clone()
        else:
            gradient = self._find(self.grad_pool, target_layer).clone()

        weights = F.adaptive_avg_pool2d(gradient, 1)
        gradient = torch.mul(gradient, weights).sum(dim=1, keepdim=True)
        gradient = F.interpolate(
            gradient, self.image_shape, mode="bilinear", align_corners=False
        )

        return gradient