from tqdm import tqdm
import numpy as np

def make_prediction(model, loader, device, mode="image", threshold=0.5):
    model.to(device)
    model.eval()

    results = {}
    for noised_images, gt_images, img_names, patch_nums in tqdm(loader):
        noised_images = noised_images.to(device)
        output = model(noised_images)
        if mode == "image":
            output = (output >= threshold).float()
        predictions = output.cpu().detach().numpy()
        for pred, name, patch in zip(predictions, img_names, patch_nums):
            results[f"{name}_{patch}"] = pred

    return results