import os
import random
import numpy as np
from torchvision import transforms
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image
    

class ImagesDataset(Dataset):
    def __init__(self, images_path=None, images_names=None, noise_transformations=None, transformations=None, patch_mode="resize", 
                 img_size=None, patch_size=None, mode="gt", seed=None):
        self.images_path = images_path
        self.transformations = transformations
        self.noise_transformations = noise_transformations
        self.mode = mode

        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.img_size = img_size

        self.seed = seed

        if images_names is None:
            self.images_dataset = os.listdir(images_path)
        else:
            self.images_dataset = list(images_names)


        if self.patch_mode == "split":
            self.patches_row = self.img_size[1] // self.patch_size[1]
            self.patches_col = self.img_size[0] // self.patch_size[0]
            num_patches = self.patches_row * self.patches_col
            self.images_dataset = [(img_name, i) for img_name in self.images_dataset for i in range(num_patches)]
        elif self.patch_mode == "resize":
            self.images_dataset = [(img_name, 0) for img_name in self.images_dataset]


    def __getitem__(self, index):
        (img_name, patch_num) = self.images_dataset[index]

        img = np.array(Image.open(os.path.join(self.images_path, img_name)).convert("RGB")).astype(np.float32) / 255

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)


        if self.patch_mode == "split":
            patch_row = patch_num // self.patches_row
            patch_col = patch_num % self.patches_row
            img_patch = img[patch_col * self.patch_size[1] : (patch_col + 1) * self.patch_size[1], 
                            patch_row * self.patch_size[0] : (patch_row + 1) * self.patch_size[0]]
        elif self.patch_mode == "resize":
            img_patch = A.Resize(self.patch_size[0], self.patch_size[1])(image=img)["image"]

        if self.mode == "gt":
            noised_patch = self.noise_transformations(image=img_patch)["image"]
        elif self.mode == "noised":
            noised_patch = img_patch.copy()
        
        img_patch = A.pytorch.transforms.ToTensorV2()(image=img_patch)["image"]

        if self.transformations is not None:
            noised_patch = self.transformations(image=noised_patch)["image"]
        
        return noised_patch, img_patch, img_name, patch_num

    def __len__(self):
        return len(self.images_dataset)


def get_learn_data(images_path, noise_transformations=None, transform_train=None, patch_mode="resize", patch_size=None, img_size=None, mode="gt",
             batch_size=8, val_size=0.25, test_size=0.25, num_workers=0, seed=0):
             
    transform_test = A.Compose([
        A.pytorch.transforms.ToTensorV2(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if transform_train is None:
        transform_train = transform_test
    
    # Разбиваем train/val/test
    trainval = ImagesDataset(images_path, noise_transformations=noise_transformations, patch_mode=patch_mode, patch_size=patch_size, img_size=img_size)
    img_names = [el[0] for el in trainval.images_dataset]
    split_val = int(len(trainval) * (1 - val_size - test_size))
    split_test = int(len(trainval) * (1 - test_size))
    train_names, val_names, test_names = set(img_names[:split_val]), set(img_names[split_val: split_test]), set(img_names[split_test:])
    
    # Загружаем данные
    trainset = ImagesDataset(images_path, train_names, noise_transformations, transform_train, patch_mode, img_size, patch_size, mode)
    valset = ImagesDataset(images_path, val_names, noise_transformations, transform_test, patch_mode, img_size, patch_size, mode, seed)
    testset = ImagesDataset(images_path, test_names, noise_transformations, transform_test, patch_mode, img_size, patch_size, mode, seed)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader