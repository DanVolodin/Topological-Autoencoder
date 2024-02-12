import torch
import wandb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from tqdm import tqdm
from piq import ssim, psnr
from torch.nn import MSELoss, BCELoss
from topoloss import TopoLoss

sns.set(style='darkgrid')

def plot_stats(
    train_log: list[float],
    valid_log: list[float],
    title: str,
    metric_name: str
):
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


def epoch_log(epoch, train_loss, val_loss, train_metrics, val_metrics, lr):
    wandb.log({"train": {"loss": train_loss["total"], "psnr": train_metrics["psnr"]}, "val": {"loss": val_loss["total"], "psnr": val_metrics["psnr"]}, "lr": lr})

    print(f"Epoch {epoch}")
    print(f" train loss: {train_loss['total']}, train psnr: {train_metrics['psnr']}")
    print(f" val loss: {val_loss['total']}, val psnr: {val_metrics['psnr']}\n")


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


def eval_epoch(model, loader, criterion, device):

    total_mse = 0
    total_topo = 0
    total_ce = 0
    total_loss = 0
    
    total_psnr = 0
    total_ssim = 0

    model.eval()

    for data, target, _, _ in tqdm(loader):
        data, target = data.to(device), target.to(device)

        predictions = model(data)

        loss = criterion(predictions, target)

        total_loss += loss.item()
        total_mse += MSELoss(reduction="sum")(predictions, target).item()
        total_ce += BCELoss(reduction="sum")(predictions, target).item()
        total_topo += TopoLoss(reduction="sum")(predictions, target).item()

        total_psnr += psnr(predictions, target, reduction="sum").item()
        total_ssim += ssim(predictions, target, reduction="sum").item()

    n_elements = len(loader.dataset)

    loss = {"total" : total_loss / n_elements,
            "mse" : total_mse / n_elements,
            "cross_entropy" : total_ce / n_elements,
            "topological_loss" : total_topo / n_elements}
    
    metrics = {"psnr" : total_psnr / n_elements,
               "ssim" : total_ssim / n_elements}

    return loss, metrics


def train_epoch(model, optimizer, train_loader, criterion, device, batch_aug=None):

    total_mse = 0
    total_topo = 0
    total_ce = 0
    total_loss = 0

    total_psnr = 0
    total_ssim = 0

    model.train()

    for data, target, _, _ in tqdm(train_loader):

        if batch_aug is not None:
            data, target = batch_aug(data, target)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        predictions = model(data)

        loss = criterion(predictions, target) / len(data)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += MSELoss(reduction="sum")(predictions, target).item()
        total_ce += BCELoss(reduction="sum")(predictions, target).item()
        total_topo += TopoLoss(reduction="sum")(predictions, target).item()

        total_psnr += psnr(predictions, target, reduction="sum").item()
        total_ssim += ssim(predictions, target, reduction="sum").item()
    
    n_elements = len(train_loader.dataset)

    loss = {"total" : total_loss / n_elements,
            "mse" : total_mse / n_elements,
            "cross_entropy" : total_ce / n_elements,
            "topological_loss" : total_topo / n_elements}
    
    metrics = {"psnr" : total_psnr / n_elements,
               "ssim" : total_ssim / n_elements}

    return loss, metrics


def fit(model, optimizer, n_epochs, train_loader, val_loader, criterion, device, scheduler=None,
        title="Model", save_checkpoints=False, path=None, log_wandb=False, batch_aug=None,
        save_frequency=1):

    if log_wandb:
        wandb.watch(model, criterion, log="all", log_freq=200)

    model.to(device)

    train_loss_log, train_metrics_log, val_loss_log, val_metrics_log = [], [], [], []

    for epoch in range(n_epochs):

        train_loss, train_metrics = train_epoch(model, optimizer, train_loader, criterion, device, batch_aug)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        train_loss_log.append(train_loss)
        train_metrics_log.append(train_metrics)

        val_loss_log.append(val_loss)
        val_metrics_log.append(val_metrics)

        if log_wandb:
            epoch_log(epoch, train_loss, val_loss, train_metrics, val_metrics, optimizer.param_groups[0]["lr"])
        else:
            clear_output()
            epoch_plot(epoch, train_loss_log, val_loss_log, train_metrics_log, val_metrics_log, title)

        if scheduler is not None:
            scheduler.step()

        if save_checkpoints and epoch % save_frequency == 0:
            torch.save(model.state_dict(), os.path.join(path, f"model_{epoch}_{val_metrics}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(path, f"optimizer_{epoch}_{val_metrics}.pt"))
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(path, f"scheduler_{epoch}_{val_metrics}.pt"))

    return train_loss_log, train_metrics_log, val_loss_log, val_metrics_log