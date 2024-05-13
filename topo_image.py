import torch
import numpy as np
import gudhi as gd

class TopoImage(object):
    def __init__(self, img):
        self.image = np.squeeze(img)
        self.shape = self.image.shape
        self.device = self.image.device

        self.ccomplex = gd.CubicalComplex(
            dimensions = (self.shape[0], self.shape[1]), 
            top_dimensional_cells=self.image.flatten()
        )
        self.ccomplex.compute_persistence()

    def betti_numbers(self):
        return self.ccomplex.persistent_betti_numbers(from_value=1, to_value=0)
    
    def critical_pairs(self):
        return self.ccomplex.cofaces_of_persistence_pairs()
    
    def persistence_diagram(self):
        h, w = self.shape
        
        # get pairs of critical simplices
        critical_pairs = self.critical_pairs()

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
        last_pixel = torch.argmax(self.image).item()
        essential_pixels_0 = np.array([[essential_features[0] // w, essential_features[0] % w], [last_pixel // w, last_pixel % 4]])[np.newaxis, ...]
        critical_pixels_0 = np.vstack([critical_pixels_0, essential_pixels_0])

        # 0-homology persistance diagram
        pd0 = self.image[critical_pixels_0[:, :, 0].flatten(), critical_pixels_0[:, :, 1].flatten()].reshape((critical_pixels_0.shape[0], 2))

        # 1-homology image critical pixels
        try:
            critical_pairs_1 = critical_pairs[0][1]
        except:
            critical_pairs_1 = np.empty((0, 2))
        critical_1_ver_ind = critical_pairs_1 // w
        critical_1_hor_ind = critical_pairs_1 % w
        critical_pixels_1 = np.stack([critical_1_ver_ind, critical_1_hor_ind], axis=2)

        # 1-homology persistance diagram
        pd1 = self.image[critical_pixels_1[:, :, 0].flatten(), critical_pixels_1[:, :, 1].flatten()].reshape((critical_pixels_1.shape[0], 2))

        return pd0, pd1
    

class HeightFiltration(object):
    def __init__(self, img_shape, direction):

        direction = direction * 1.0 / np.linalg.norm(direction, 1)

        indicies_dim_0 = np.tile(np.arange(img_shape[0]), (img_shape[1], 1)).T
        indicies_dim_1 = np.tile(np.arange(img_shape[1]), (img_shape[0], 1))
        indicies = np.stack([indicies_dim_0, indicies_dim_1]).transpose(1, 2, 0)
        
        self.filtration_matrix = torch.tensor(np.dot(indicies, direction))

        if self.filtration_matrix.min() <= 0:
            self.filtration_matrix -= self.filtration_matrix.min()

        self.filtration_matrix /= self.filtration_matrix.max()

    def __call__(self, img_batch):
        return (img_batch + self.filtration_matrix[np.newaxis, np.newaxis, ...]) / 2
    

def precompute_topo_images(img_batch, filtration=None):
    img_batch = 1 - img_batch
    if filtration:
        img_batch = filtration(img_batch)
    topo_images = []
    for img in img_batch:
        topo_images.append(TopoImage(img))
    return topo_images