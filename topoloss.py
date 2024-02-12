import torch
import torch.nn as nn
import numpy as np
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance

class TopoLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(TopoLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.losses = torch.empty((0))
        for input_img, target_img in zip(inputs, targets):
            loss = wasserstein_topological(torch.mean(input_img, dim=0), torch.mean(target_img, dim=0))
            self.losses = torch.hstack([self.losses, loss])
        if self.reduction == "mean":
            self.loss = torch.mean(self.losses)
        elif self.reduction == "sum":
            self.loss = torch.sum(self.losses)
        return self.loss
    
    def backward(self):
        return self.loss.backward()
    

def persistence_diagram(image):

    h, w = image.shape
    img_flat = image.flatten()

    ccomplex = gd.CubicalComplex(
        dimensions = (h, w), 
        top_dimensional_cells=img_flat
    )
    
    # get pairs of critical simplices
    ccomplex.compute_persistence()
    critical_pairs = ccomplex.cofaces_of_persistence_pairs()

    # get essential critical pixels (never vanish)
    essential_features = critical_pairs[1][0]

    # 0-homology image critical pixels
    try:
        critical_pairs_0 = critical_pairs[0][0]
    except:
        critical_pairs_0 = np.empty((0, 2))
    critical_0_ver_ind = critical_pairs_0 // w
    critical_0_hor_ind = critical_pairs_0 % w
    critical_pixels_0 = np.stack([critical_0_ver_ind, critical_0_hor_ind], axis=2)

    # 0-homology essential pixels (ends with last added pixel)
    last_pixel = torch.argmax(image).item()
    essential_pixels_0 = np.array([[essential_features[0] // w, essential_features[0] % w], [last_pixel // w, last_pixel % 4]])[np.newaxis, ...]
    critical_pixels_0 = np.vstack([critical_pixels_0, essential_pixels_0])

    # 0-homology persistance diagram
    pd0 = image[critical_pixels_0[:, :, 0].flatten(), critical_pixels_0[:, :, 1].flatten()].reshape((critical_pixels_0.shape[0], 2))

    # 1-homology image critical pixels
    try:
        critical_pairs_1 = critical_pairs[0][1]
    except:
        critical_pairs_1 = np.empty((0, 2))
    critical_1_ver_ind = critical_pairs_1 // w
    critical_1_hor_ind = critical_pairs_1 % w
    critical_pixels_1 = np.stack([critical_1_ver_ind, critical_1_hor_ind], axis=2)

    # 1-homology persistance diagram
    pd1 = image[critical_pixels_1[:, :, 0].flatten(), critical_pixels_1[:, :, 1].flatten()].reshape((critical_pixels_1.shape[0], 2))

    return pd0, pd1


def wasserstein_topological(img1, img2):
    pd0_img1, pd1_img1 = persistence_diagram(img1)
    pd0_img2, pd1_img2 = persistence_diagram(img2)

    dist = wasserstein_distance(pd0_img1, pd0_img2, order=2, internal_p=2, enable_autodiff=True, keep_essential_parts=False) ** 2 + \
           wasserstein_distance(pd1_img1, pd1_img2, order=2, internal_p=2, enable_autodiff=True, keep_essential_parts=False) ** 2

    return dist
