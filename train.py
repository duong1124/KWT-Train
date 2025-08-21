import torch
import json
import yaml
import os
import time

from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from kwt import kwt_from_name, KWT
from train_utils import get_config, count_params, calc_step, seed_everything, log, save_model
from dataset import PrecomputedSpeechDataset, GoogleSpeechDataset
from train_items import LabelSmoothingLoss, WarmUpLR


def train_single_batch(net: nn.Module, data: torch.Tensor, targets: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable, device: torch.device) -> Tuple[float, int]:
    """Performs a single training step."""

    data, targets = data.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = net(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    correct = outputs.argmax(1).eq(targets).sum()
    return loss.item(), correct.item()


@torch.no_grad()
def evaluate(net: nn.Module, criterion: Callable, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:

    net.eval()
    correct = 0
    running_loss = 0.0

    for data, targets in tqdm(dataloader):
        data, targets = data.to(device), targets.to(device)
        out = net(data)
        correct += out.argmax(1).eq(targets).sum().item()
        loss = criterion(out, targets)
        running_loss += loss.item()

    net.train()
    accuracy = correct / len(dataloader.dataset)
    return accuracy, running_loss / len(dataloader)


def train(net: nn.Module, optimizer: optim.Optimizer, criterion: Callable, trainloader: DataLoader, schedulers: dict, config: dict) -> None:

    step = 0
    best_acc = 0.0
    n_batches = len(trainloader)
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    ############################
    # start training
    ############################
    net.train()

    for epoch in range(config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        correct = 0

        for batch_index, (data, targets) in enumerate(trainloader):

            if schedulers["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()

            elif schedulers["scheduler"] is not None:
                schedulers["scheduler"].step()

            ####################
            # optimization step
            ####################

            loss, corr = train_single_batch(net, data, targets, optimizer, criterion, device)
            running_loss += loss
            correct += corr

            if not step % config["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                log(log_dict, step, config)

            step += 1

        #######################
        # epoch complete
        #######################
        train_acc = correct / len(trainloader.dataset)
        log_dict = {
            "epoch": epoch,
            "time_per_epoch": time.time() - t0,
            "train_acc": train_acc,
            "avg_loss_per_ep": running_loss / len(trainloader)
        }
        log(log_dict, step, config)

        # save best model based on train acc
        if train_acc > best_acc:
            best_acc = train_acc
            save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
            save_model(epoch, train_acc, save_path, net, optimizer, log_file)

    ###########################
    # training complete
    ###########################

    # save final ckpt
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, train_acc, save_path, net, optimizer, log_file)


def training_pipeline(config):
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """
    start_func_time = time.time()
    # Get label map
    with open(config["label_map"], "r") as f:
      label_map = json.load(f)

    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)

    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)

    #####################################
    # initialize training items
    #####################################

    # data
    train_dataset = PrecomputedSpeechDataset(
        pkl_path=config["pkl_train"],
        aug_settings = config["hparams"]["augment"],
        label_map = label_map,
        train=True # for data_augment
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=True
    )

    # model
    if config["hparams"]["model"]["name"] is not None:
        model =  kwt_from_name(config["hparams"]["model"]["name"])
    else:
        model = KWT(**config["hparams"]["model"])

    model = model.to(config["hparams"]["device"])
    print(f"Created model with {count_params(model)} parameters.")

    # loss
    if config["hparams"]["l_smooth"]:
        criterion = LabelSmoothingLoss(num_classes=config["hparams"]["model"]["num_classes"], smoothing=config["hparams"]["l_smooth"])
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        **config["hparams"]["optimizer"]["opt_kwargs"]
    )

    # lr scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }

    if config["hparams"]["scheduler"]["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(optimizer, total_iters=len(trainloader) * config["hparams"]["scheduler"]["n_warmup"])

    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = len(trainloader) * max(1, (config["hparams"]["scheduler"]["max_epochs"] - config["hparams"]["scheduler"]["n_warmup"]))
        schedulers["scheduler"] = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iters,
            eta_min=1e-8
        )

    #####################################
    # Training Run
    #####################################
    end_func_time = time.time()
    print("Initiating training.")
    print(f"Takes {end_func_time - start_func_time} seconds to start training.")

    train(model, optimizer, criterion, trainloader, schedulers, config)

    #####################################
    # Final Test
    #####################################

    test_dataset = PrecomputedSpeechDataset(
        pkl_path=config["pkl_test"],
        aug_settings = config["hparams"]["augment"],
        label_map = label_map,
        train=False
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=False
    )
    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1)

    # evaluating the final state (last.pth)
    print("Evaluating last ckpt")
    test_acc, test_loss = evaluate(model, criterion, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_acc_last": test_acc
    }
    log(log_dict, final_step, config)

    # evaluating the best state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded - by train_acc.")

    test_acc, test_loss = evaluate(model, criterion, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_acc_best": test_acc
    }
    log(log_dict, final_step, config)

def main(conf_file):
    config = get_config(conf_file)
    seed_everything(config["hparams"]["seed"])
    training_pipeline(config)

if __name__ == '__main__':
    conf_file = "base_config.yaml"
    main(conf_file)