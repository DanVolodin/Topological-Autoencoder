import os
import random
import numpy as np
import albumentations as A
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
    

class ImagesDataset(Dataset):
    def __init__(self, images_path, gt_prefix=None, noised_prefix=None, images_names=None, noise_transformations=None, transformations=None, patch_mode="resize", 
                 img_size=None, patch_size=None, max_size=None, filter_empty=False, shuffle_truncated=True):
        
        self.gt_path = images_path if gt_prefix is None else os.path.join(images_path, gt_prefix)
        self.noised_path = None if noised_prefix is None else os.path.join(images_path, noised_prefix)
        self.transformations = transformations
        self.noise_transformations = noise_transformations

        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.img_size = img_size

        if images_names is None:
            self.images_dataset = os.listdir(self.gt_path)
        else:
            self.images_dataset = list(images_names)
        self.images_dataset.sort()

        if self.patch_mode == "split":
            self.patches_row = self.img_size[1] // self.patch_size[1]
            self.patches_col = self.img_size[0] // self.patch_size[0]
            num_patches = self.patches_row * self.patches_col
            self.images_dataset = [(img_name, i) for img_name in self.images_dataset for i in range(num_patches)]
            if filter_empty:
                self.images_dataset = list(filter(lambda x: (self.get_patch(self.get_img(self.gt_path, x[0]), x[1]) > 0).any(), self.images_dataset))
                
        elif self.patch_mode in ["resize", "keep"]:
            self.images_dataset = [(img_name, 0) for img_name in self.images_dataset]
            if filter_empty:
                self.images_dataset = list(filter(lambda x: (self.get_img(self.gt_path, x[0]) > 0).any(), self.images_dataset))

        if max_size is not None and len(self.images_dataset) > max_size:
                if shuffle_truncated:
                    random.shuffle(self.images_dataset)
                self.images_dataset = self.images_dataset[:max_size]

    def get_img(self, path, img_name):
        img_path = os.path.join(path, img_name)
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255
        return img

    def get_patch(self, img, patch_num):
            patch_row = patch_num // self.patches_row
            patch_col = patch_num % self.patches_row
            img_patch = img[patch_col * self.patch_size[1] : (patch_col + 1) * self.patch_size[1], 
                            patch_row * self.patch_size[0] : (patch_row + 1) * self.patch_size[0]]
            return img_patch

    def __getitem__(self, index):
        (img_name, patch_num) = self.images_dataset[index]

        img = self.get_img(self.gt_path, img_name)

        if self.patch_mode == "split":
            img_patch = self.get_patch(img, patch_num)
        elif self.patch_mode == "resize":
            img_patch = A.Resize(self.patch_size[0], self.patch_size[1])(image=img)["image"]
        elif self.patch_mode == "keep":
            img_patch = img.copy()
        else:
            raise ValueError("Unknown patch_mode")
        
        img_patch = (img_patch > 0).astype(np.float32)

        if self.noised_path is None:
            noised_patch = self.noise_transformations(image=img_patch)["image"]
        else:
            noised_patch = self.get_img(self.noised_path, img_name)
            if self.patch_mode == "resize":
                noised_patch = A.Resize(self.patch_size[0], self.patch_size[1])(image=noised_patch)["image"]
        # noised_patch = (noised_patch > 0).astype(np.float32)
        
        img_patch = A.pytorch.transforms.ToTensorV2()(image=img_patch)["image"]

        if self.transformations is not None:
            noised_patch = self.transformations(image=noised_patch)["image"]
        
        return noised_patch[0].unsqueeze(0), img_patch[0].unsqueeze(0), img_name, patch_num

    def __len__(self):
        return len(self.images_dataset)


def get_learn_data(images_path, train_gt_prefix=None, val_gt_prefix=None, test_gt_prefix=None,
                   train_noised_prefix=None, val_noised_prefix=None, test_noised_prefix=None,
                   noise_transformations=None, transform_train=None, patch_mode="resize", patch_size=None, img_size=None,
                   batch_size=32, max_size=None, filter_empty=False, val_size=0.25, test_size=0.25, shuffle_splits=True, 
                   num_workers=0, pin_memory=False):
             
    transform_test = A.Compose([
        ToTensorV2(),
    ])

    if transform_train is None:
        transform_train = transform_test
    
    if train_gt_prefix and val_gt_prefix and test_gt_prefix:
        # Загружаем данные
        trainset = ImagesDataset(images_path, train_gt_prefix, train_noised_prefix, None,
                                 noise_transformations, transform_train, patch_mode, img_size, patch_size, max_size, filter_empty)
        valset = ImagesDataset(images_path, val_gt_prefix, val_noised_prefix, None,
                               noise_transformations, transform_test, patch_mode, img_size, patch_size, max_size, filter_empty)
        testset = ImagesDataset(images_path, test_gt_prefix, test_noised_prefix, None,
                                noise_transformations, transform_test, patch_mode, img_size, patch_size, max_size, filter_empty)
    else:
        # Разбиваем train/val/test
        trainval = ImagesDataset(images_path, patch_mode="keep")
        img_names = [el[0] for el in trainval.images_dataset]
        if shuffle_splits:
            random.shuffle(img_names)
        split_val = int(len(trainval) * (1 - val_size - test_size))
        split_test = int(len(trainval) * (1 - test_size))
        train_names, val_names, test_names = set(img_names[:split_val]), set(img_names[split_val: split_test]), set(img_names[split_test:])
        
        # Загружаем данные
        trainset = ImagesDataset(images_path, train_gt_prefix, train_noised_prefix, train_names, noise_transformations, 
                                 transform_train, patch_mode, img_size, patch_size, max_size, filter_empty)
        valset = ImagesDataset(images_path, val_gt_prefix, val_noised_prefix, val_names, noise_transformations, 
                               transform_test, patch_mode, img_size, patch_size, max_size, filter_empty)
        testset = ImagesDataset(images_path, test_gt_prefix, test_noised_prefix, test_names, noise_transformations, 
                                transform_test, patch_mode, img_size, patch_size, max_size, filter_empty)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


def save_datasets(save_path, loaders, split_names, filter_empty=False):
    for loader, split_name in zip(loaders, split_names):

        noised_path = os.path.join(save_path, split_name)
        gt_path = os.path.join(save_path, f"{split_name}_gt")

        os.makedirs(noised_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        for noised_images, gt_images, img_names, patch_nums in tqdm(loader):
            for noised_image, gt_image, img_name, patch_num in zip(noised_images, gt_images, img_names, patch_nums):
                if (noised_image != 0).any():
                    img_name = img_name.split('.')[0]
                    save_image(noised_image, os.path.join(noised_path, f"{img_name}_{patch_num}.png"))
                    save_image(gt_image, os.path.join(gt_path, f"{img_name}_{patch_num}.png"))