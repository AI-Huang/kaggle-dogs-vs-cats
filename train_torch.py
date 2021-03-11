#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-03-21 16:49
# @Update  : Mar-11-21 20:23
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""train_torch.py
Train models using PyTorch.

Model list:
- Vision Transformer
"""

from __future__ import print_function

import argparse
import glob
import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def training_args():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--arch', type=str, dest='arch',
                        action='store', default="resnet50", help='arch, model architecture.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=64, help='batch_size, e.g. 32.')
    parser.add_argument('--epochs', type=int, dest='epochs',
                        action='store', default=20, help='training epochs, e.g. 100.')  # origin exper 100, ETA about 2 hours
    parser.add_argument('--lr', type=float, dest='lr',
                        action='store', default=3e-5, help='')
    parser.add_argument('--gamma', type=float, dest='gamma',
                        action='store', default=0.7, help='')

    # Device
    parser.add_argument('--device', type=str, dest='device',
                        action='store', default="cuda", help=""""cpu" or "cuda".""")

    args = parser.parse_args()

    model_list = ["resnet50", "ViT"]
    if args.arch not in model_list:
        raise ValueError(
            f"args.arch {args.arch} not in model_list {model_list}")

    return args


def train(model, epochs, criterion, optimizer, train_loader, valid_loader, scheduler=None, **kwargs):
    device = kwargs["device"] if "device" in kwargs else "cpu"
    verbose = kwargs["verbose"] if "verbose" in kwargs else False
    save_weights = kwargs["save_weights"] if "save_weights" in kwargs else False
    if save_weights:
        assert "save_dir" in kwargs
        save_dir = kwargs["save_dir"]

    history = {
        "epoch": [],
        "step": [],
        "step_loss": [],
        "step_acc": [],
        "epoch_loss": [],
        "epoch_accuracy": [],
        "epoch_val_loss": [],
        "epoch_val_accuracy": []
    }

    step = 0
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_accuracy = 0

        # On train set
        history["epoch"].append(epoch)
        for data, label in tqdm(train_loader):
            history["step"].append(step)

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                tqdm.write(f"Batch loss: {loss:.4f}.")

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(data)

            history["step_loss"].append(loss)
            history["step_acc"].append(acc)
            step += 1

        history["epoch_accuracy"].append(epoch_accuracy)
        history["epoch_loss"].append(epoch_loss)

        if verbose:
            tqdm.write(f"Epoch average loss: {epoch_loss:.4f}.")
        if save_weights:
            weights_name = f"epoch-{epoch+1:03d}-loss-{epoch_loss:.4f}-acc-{epoch_accuracy:.4f}.pt"
            weights_path = os.path.join(save_dir, weights_name)
            torch.save(model.state_dict(), weights_path)
            tqdm.write(f"Save weights to: {weights_path}.")

        # On val set
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        if verbose:
            tqdm.write(f"Epoch average val_loss: {epoch_val_loss:.4f}.")

        history["epoch_val_accuracy"].append(epoch_val_accuracy)
        history["epoch_val_loss"].append(epoch_val_loss)

        tqdm.write(
            f"Epoch: {epoch+1} - loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

        if scheduler:
            scheduler.step()

    return history


def main():
    # Training settings
    args = training_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    gamma = args.gamma
    seed = 42
    device = args.device

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed)

    # Data path
    competition_name = "dogs-vs-cats-redux-kernels-edition"
    data_dir = os.path.expanduser(f"~/.kaggle/competitions/{competition_name}")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")
    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    # Split
    train_list, valid_list = train_test_split(
        train_list, test_size=0.2, stratify=labels, random_state=seed)

    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")

    # Image Augumentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # Dataloader
    from dataset.cats_dogs import CatsDogsDataset
    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, transform=val_transforms)
    test_data = CatsDogsDataset(test_list, transform=test_transforms)

    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        dataset=valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True)

    print(
        f"train samples: {len(train_data)},train batches: {len(train_loader)}.")
    print(
        f"val samples: {len(valid_data)},val batches: {len(valid_loader)}.")

    # Select and prepare model
    if args.arch == "ViT":
        from vit_pytorch.efficient import ViT
        # Effecient Attention
        # Linformer
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )
        # Visual Transformer
        model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            transformer=efficient_transformer,
            channels=3,
        ).to(device)
        # Training configs for ViT
        criterion = nn.CrossEntropyLoss()  # loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer
        scheduler = StepLR(optimizer, step_size=1,
                           gamma=gamma)  # scheduler TODO 没用上
    elif args.arch == "resnet50":
        from torchvision.models.resnet import resnet50
        from torch.optim.lr_scheduler import MultiStepLR
        model = resnet50().to(device)
        # Training configs for resnet50
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)  # optimizer
        # ResNet learning schedule
        scheduler = MultiStepLR(optimizer, milestones=[
                                80, 120, 160], gamma=0.1)

    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", competition_name)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    subfix = os.path.join(args.arch, current_time)

    save_dir = os.path.expanduser(os.path.join(prefix, subfix, "ckpts"))
    log_dir = os.path.expanduser(os.path.join(prefix, subfix, "logs"))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    history = train(model, epochs, criterion, optimizer,
                    train_loader, valid_loader, scheduler=scheduler, device=device, verbose=True, save_weights=True, save_dir=save_dir)

    path = os.path.join(log_dir, "history.pickle")
    with open(path, "wb") as f:
        pickle.dump(history, f)
        print(f"Saved history to: {path}")


if __name__ == "__main__":
    main()
