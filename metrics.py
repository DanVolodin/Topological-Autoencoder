import torch
import numpy as np
from matplotlib import pyplot as plt
from topo_image import precompute_topo_images
from tqdm import tqdm
from predict import make_prediction

def pixel_wise_accuracy(inputs, targets, threshold=0.5, reduction="mean"):
    inputs = (inputs >= threshold).float()
    accuracy = torch.mean((inputs == targets).float(), dim=(1, 2, 3))
    if reduction == "mean":
        return accuracy.mean()
    else:
        return accuracy.sum()
    

def betti_number_error(inputs, targets, threshold=0.5, reduction="mean"):
    inputs = (inputs >= threshold).float()
    inputs = precompute_topo_images(inputs)
    targets = precompute_topo_images(targets)

    betti_number_error_0 = 0
    betti_number_error_1 = 0

    for input, target in zip(inputs, targets):
        betti_number_error_0 += abs(target.betti_numbers()[0] - input.betti_numbers()[0])
        betti_number_error_1 += abs(target.betti_numbers()[1] - input.betti_numbers()[1])

    if reduction == "mean":
        betti_number_error_0 /= len(inputs)
        betti_number_error_1 /= len(inputs)

    betti_total = betti_number_error_0 + betti_number_error_1
    return betti_total, betti_number_error_0, betti_number_error_1


def calc_test_metrics(loader, predictions_dict):
    accuracy = 0
    betti_number_error_total = 0
    betti_number_error_0 = 0
    betti_number_error_1 = 0
    for noised_images, gt_images, img_names, patch_nums in tqdm(loader):
        predicted_images = torch.Tensor(np.array([predictions_dict[f"{img_name}_{patch_num}"] for img_name, patch_num in zip(img_names, patch_nums)]))

        accuracy += pixel_wise_accuracy(predicted_images, gt_images, reduction="sum").item()

        betti_total, betti_0, betti_1 = betti_number_error(predicted_images, gt_images, reduction="sum")
        betti_number_error_total += betti_total
        betti_number_error_0 += betti_0
        betti_number_error_1 += betti_1
    
    n_elements = len(loader.dataset)
    
    return {"accuracy" : accuracy / n_elements,
            "betti_number_error" : betti_number_error_total / n_elements,
            "betti_number_error_0" : betti_number_error_0 / n_elements,
            "betti_number_error_1" : betti_number_error_1 / n_elements
            }


def evaluate_model(net, loader, device, threshold=0.5, start_ind=0):

    predictions_proba = make_prediction(net, loader, device, mode="proba")
    predictions = dict()
    for key, prediciton in predictions_proba.items():
        predictions[key] = (prediciton >= threshold).astype(float)

    print(calc_test_metrics(loader, predictions))

    fig = plt.figure(figsize=(10, 32))
    for idx in np.arange(8):
        noised_image, gt_image, img_name, patch_num = loader.dataset[idx + start_ind]

        # noised input
        ax1 = fig.add_subplot(16, 4, 4 * idx + 1, xticks=[], yticks=[])
        ax1.imshow(np.squeeze(noised_image), cmap='grey')

        # model output
        ax2 = fig.add_subplot(16, 4, 4 * (idx) + 2, xticks=[], yticks=[])
        ax2.imshow(np.squeeze(predictions[f"{img_name}_{patch_num}"]), cmap='grey')

        # model output proba
        ax3 = fig.add_subplot(16, 4, 4 * (idx) + 3, xticks=[], yticks=[])
        ax3.imshow(np.squeeze(predictions_proba[f"{img_name}_{patch_num}"]), cmap='grey')

        # gt image
        ax4 = fig.add_subplot(16, 4, 4 * idx + 4, xticks=[], yticks=[])
        ax4.imshow(np.squeeze(gt_image), cmap='grey')

        fig.tight_layout()
    plt.show()
    
    return predictions, predictions_proba