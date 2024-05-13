import torch
import wandb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from tqdm import tqdm
from torch.nn import BCELoss
from topoloss import TopoLoss
from IPython.display import clear_output
from metrics import pixel_wise_accuracy, betti_number_error
from utils import set_random_seed

def plot_stats(
    train_log: list[float],
    valid_log: list[float],
    title: str,
    metric_name: str
):
    sns.set_theme(style='darkgrid')

    plt.figure(figsize=(16, 8))

    plt.title(title + ' ' + metric_name)

    plt.plot(train_log, label=f'Train {metric_name}')
    plt.plot(valid_log, label=f'Valid {metric_name}')
    plt.legend()

    plt.show()

    plt.show()


def epoch_plot(epoch, train_loss_log, val_loss_log, train_metrics_log, val_metrics_log, title):

    for loss in train_loss_log[0].keys():
        train_log = [epoch[loss] for epoch in train_loss_log]
        val_log = [epoch[loss] for epoch in val_loss_log]
        plot_stats(train_log, val_log, title, loss)

    for metric in train_metrics_log[0].keys():
        train_log = [epoch[metric] for epoch in train_metrics_log]
        val_log = [epoch[metric] for epoch in val_metrics_log]
        plot_stats(train_log, val_log, title, metric)


    print(f"Epoch {epoch}")
    print(f" train loss: {train_loss_log[-1]}, train metrics: {train_metrics_log[-1]}")
    print(f" val loss: {val_loss_log[-1]}, val metrics: {val_metrics_log[-1]}\n")


def wandb_log(train_loss, val_loss, train_metrics, val_metrics, lr):
    wandb.log({"train": train_loss | train_metrics, 
               "val": val_loss | val_metrics, 
               "lr": lr})


def plot_scheduler(scheduler, optimizer, steps, title='Scheduler'):
    lrs = []
    for _ in range(steps):
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.title(title + 'learning rate')
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.plot(lrs)
    plt.show()


def eval_epoch(model, loader, lambda_ce, lambda_topo, device, filtration):

    topo_loss = 0
    ce_loss = 0
    total_loss = 0
    
    accuracy = 0
    betti_number_error_0 = 0
    betti_number_error_1 = 0
    betti_number_error_total = 0

    model.eval()

    for data, target, _, _ in tqdm(loader):
        data, target = data.to(device), target.to(device)

        predictions = model(data)

        batch_ce_loss = BCELoss(reduction="sum")(predictions, target) / (data.shape[2] * data.shape[3])
        batch_topo_loss = TopoLoss(reduction="sum", filtration=filtration)(predictions, target)

        loss = lambda_ce * batch_ce_loss + lambda_topo * batch_topo_loss
        
        total_loss += loss.item()
        ce_loss += batch_ce_loss.item()
        topo_loss += batch_topo_loss.item()

        accuracy += pixel_wise_accuracy(predictions, target, reduction="sum").item()

        betti_total, betti_0, betti_1 = betti_number_error(predictions, target, reduction="sum")
        betti_number_error_total += betti_total
        betti_number_error_0 += betti_0
        betti_number_error_1 += betti_1

    n_elements = len(loader.dataset)

    loss = {"total" : total_loss / n_elements,
            "cross_entropy" : ce_loss / n_elements,
            "topoloss" : topo_loss / n_elements
            }
    
    metrics = {"accuracy" : accuracy / n_elements,
               "betti_number_error" : betti_number_error_total / n_elements,
               "betti_number_error_0" : betti_number_error_0 / n_elements,
               "betti_number_error_1" : betti_number_error_1 / n_elements
               }

    return loss, metrics


def train_epoch(model, optimizer, train_loader, lambda_ce, lambda_topo, device, calc_betti, filtration):

    topo_loss = 0
    ce_loss = 0
    total_loss = 0

    accuracy = 0
    betti_number_error_0 = 0
    betti_number_error_1 = 0
    betti_number_error_total = 0

    model.train()

    for data, target, _, _ in tqdm(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        predictions = model(data)

        batch_ce_loss = BCELoss(reduction="sum")(predictions, target) / (data.shape[2] * data.shape[3])
        batch_topo_loss = TopoLoss(reduction="sum", filtration=filtration)(predictions, target)

        loss = lambda_ce * batch_ce_loss + lambda_topo * batch_topo_loss
        total_loss += loss.item()
        loss /= len(data)

        loss.backward()
        optimizer.step()
        
        ce_loss += batch_ce_loss.item()
        topo_loss += batch_topo_loss.item()
        
        accuracy += pixel_wise_accuracy(predictions, target, reduction="sum").item()

        if calc_betti:
            betti_total, betti_0, betti_1 = betti_number_error(predictions, target, reduction="sum")
            betti_number_error_total += betti_total
            betti_number_error_0 += betti_0
            betti_number_error_1 += betti_1
    
    n_elements = len(train_loader.dataset)

    loss = {"total" : total_loss / n_elements,
            "cross_entropy" : ce_loss / n_elements,
            "topoloss" : topo_loss / n_elements
            }
    
    if calc_betti:
        metrics = {"accuracy" : accuracy / n_elements,
                "betti_number_error" : betti_number_error_total / n_elements,
                "betti_number_error_0" : betti_number_error_0 / n_elements,
                "betti_number_error_1" : betti_number_error_1 / n_elements
                }
    else:
        metrics = {"accuracy" : accuracy / n_elements}

    return loss, metrics


def fit(model, optimizer, n_epochs, train_loader, val_loader, lambda_ce, lambda_topo, device, scheduler=None,
        title="Model", calc_train_betti=True, save_checkpoints=False, save_path=None, log_wandb=False, save_frequency=1,
        filtration=None):

    if log_wandb:
        wandb.watch(model, log="all", log_freq=200)

    model.to(device)

    train_loss_log, train_metrics_log, val_loss_log, val_metrics_log = [], [], [], []

    for epoch in range(n_epochs):

        train_loss, train_metrics = train_epoch(model, optimizer, train_loader, lambda_ce, lambda_topo, device, calc_train_betti, filtration)
        val_loss, val_metrics = eval_epoch(model, val_loader, lambda_ce, lambda_topo, device, filtration)

        train_loss_log.append(train_loss)
        train_metrics_log.append(train_metrics)

        val_loss_log.append(val_loss)
        val_metrics_log.append(val_metrics)

        if log_wandb:
            wandb_log(train_loss, val_loss, train_metrics, val_metrics, optimizer.param_groups[0]["lr"])
        clear_output()
        epoch_plot(epoch, train_loss_log, val_loss_log, train_metrics_log, val_metrics_log, title)

        if scheduler is not None:
            scheduler.step()

        if save_checkpoints and epoch % save_frequency == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch}_{val_metrics}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_path, f"optimizer_{epoch}_{val_metrics}.pt"))
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(save_path, f"scheduler_{epoch}_{val_metrics}.pt"))

    return train_loss_log, train_metrics_log, val_loss_log, val_metrics_log