import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from libs.bps.bp import BackPropagation

class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, extractor, classifier):
        super(GuidedBackPropagation, self).__init__(extractor, classifier)
        self.grad_pool = {}
        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

                if isinstance(module, nn.ReLU):
                    return (F.relu(grad_in[0]),)
            return backward_hook
        
        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(save_grads(module[0])))
    
