import torch
import torch.nn as nn
import numpy as np
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
from topo_image import precompute_topo_images

class TopoLoss(nn.Module):
    def __init__(self, reduction="mean", filtration=None):
        super().__init__()
        self.reduction = reduction
        self.filtration = filtration

    def forward(self, inputs, targets):
        inputs = precompute_topo_images(inputs, self.filtration)
        targets = precompute_topo_images(targets, self.filtration)

        self.losses = torch.empty((0), device=inputs[0].device)
        for input, target in zip(inputs, targets):
            loss = wasserstein_topological(input, target)
            self.losses = torch.hstack([self.losses, loss])

        if self.reduction == "mean":
            self.loss = torch.mean(self.losses)
        elif self.reduction == "sum":
            self.loss = torch.sum(self.losses)
        return self.loss
    
    def backward(self):
        return self.loss.backward()


def wasserstein_topological(topo_img1, topo_img2):
    pd0_img1, pd1_img1 = topo_img1.persistence_diagram()
    pd0_img2, pd1_img2 = topo_img2.persistence_diagram()

    dist = wasserstein_distance(pd0_img1, pd0_img2, order=2, internal_p=2, enable_autodiff=True, keep_essential_parts=False) ** 2 + \
           wasserstein_distance(pd1_img1, pd1_img2, order=2, internal_p=2, enable_autodiff=True, keep_essential_parts=False) ** 2

    return dist
