"""This file contains functions for training and testing the model for time series regression."""

import time
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler


def lr_update(step: int, total_epoch: int, train_size: int, initial_lr: float) -> float:
    progress = step / (total_epoch * train_size)
    lr = initial_lr * np.cos(progress * (7 * np.pi) / (2 * 8))
    lr *= np.clip(progress * 100, 0, 1)
    return lr


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
) -> torch.nn.Module:
    device = configs.get("device", "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_update(
            step * 256, epochs, len(train_loader) * 256, 0.1
        ),
    )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct_predictions = 0  # TODO 时序模型的准确率暂时就用MSELOSS来表达

        model.train()

        for data, target in train_loader:
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).float(),
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output.unsqueeze(-1), target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct_predictions += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)   # TODO 时序模型的准确率暂时就用MSELOSS来表达,这里的len先这样

        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

        if test_loader:
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    model.to("cpu")
    return model


def dp_train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
) -> Tuple[torch.nn.Module, float]:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    device = configs.get("device", "cpu")
    model = model.to(device)
    model = ModuleValidator.fix(model)

    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_update(
            step * 256, epochs, len(train_loader) * 256, 0.1
        ),
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss, correct_predictions = 0, 0

        model.train()

        for data, target in train_loader:
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).float(),
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct_predictions += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)

        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)

        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | DP guarantee: (ε = {epsilon:.2f}, δ = {1e-5})"
        )

        if test_loader:
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    model.to("cpu")
    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    return model, epsilon


def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple[float, float]:
    model.eval().to(device)
    loss_fn = nn.MSELoss()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).float()

            output = model(data)
            loss = loss_fn(output.unsqueeze(-1), target)

            total_loss += loss.item()
            correct_predictions += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return avg_loss, accuracy


def get_optimizer(model: torch.nn.Module, configs: Dict) -> torch.optim.Optimizer:
    optimizer_name = configs.get("optimizer", "SGD")
    learning_rate = configs.get("learning_rate", 0.001)
    weight_decay = configs.get("weight_decay", 0.0)
    momentum = configs.get("momentum", 0.0)

    print(
        f"Using optimizer: {optimizer_name} | Learning Rate: {learning_rate} | Weight Decay: {weight_decay}"
    )

    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{optimizer_name}' is not implemented. Choose 'SGD', 'Adam', or 'AdamW'."
        )
