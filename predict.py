from tqdm import tqdm
import torch

def make_prediction(model, loader, device):
    model.to(device)
    model.eval()

    results = {}
    for noised_images, gt_images, img_names, patch_nums in tqdm(loader):
        noised_images = noised_images.to(device)
        output = model(noised_images)
        predictions = output.cpu().detach().numpy()
        for pred, key in zip(predictions, img_names):
            results[key] = pred

    return results