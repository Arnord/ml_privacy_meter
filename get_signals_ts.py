import os.path
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, AutoTokenizer

from dataset.utils import load_dataset_subsets


def get_softmax(
    model: Union[PreTrainedModel, torch.nn.Module],
    samples: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
    temp: float = 1.0,
    pad_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Get the model's softmax probabilities for the given inputs and expected outputs.

    Args:
        model (PreTrainedModel or torch.nn.Module): Model instance.
        samples (torch.Tensor): Model input.
        labels (torch.Tensor): Model expected output.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for computing signals.
        temp (float): Temperature used in softmax computation.
        pad_token_id (Optional[int]): Padding token ID to ignore in aggregation.

    Returns:
        all_softmax_list (np.array): softmax value of all samples
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        conf_list = []
        batched_samples = torch.split(samples, batch_size)
        batched_labels = torch.split(labels, batch_size)

        for x, y in tqdm(
            zip(batched_samples, batched_labels),
            total=len(batched_samples),
            desc="Computing softmax",
        ):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            # For regression models, use negative MSE as a confidence-like signal
            # Use MAE instead of MSE to reduce sensitivity to outliers
            use_mae = False  # Set to False if you want to revert to MSE
            if use_mae:
                error = torch.abs(pred - y.squeeze(-1))
            else:
                error = F.mse_loss(pred, y.squeeze(-1), reduction="none")
            if error.dim() > 1:
                error = error.mean(dim=tuple(range(1, error.dim())), keepdim=False)

            # TODO : 攻击模型的置信度，可以尝试直接用-mse，即log(exp(-mse))应该会更符合高斯分布
            # TODO : 归一化，但如果用TFB,在数据处理阶段应该就已经归一化了，但是对MSE做归一化可能会更接近softmax
            error_mean = error.mean(dim=1, keepdim=True)
            error_std = error.std(dim=1, keepdim=True)
            if error_std > 0:
                error = (error - error_mean) / error_std

            conf_like = torch.exp(-error)  # higher = more confident
            conf_list.append(conf_like.view(-1, 1).to("cpu"))
        all_conf_list = np.concatenate(conf_list)

    model.to("cpu")
    return all_conf_list


def get_model_signals(models_list, dataset, configs, logger, is_population=False):
    """Function to get models' signals (softmax, loss, logits) on a given dataset.

    Args:
        models_list (list): List of models for computing (softmax, loss, logits) signals from them.
        dataset (torchvision.datasets): The whole dataset.
        configs (dict): Configurations of the tool.
        logger (logging.Logger): Logger object for the current run.
        is_population (bool): Whether the signals are computed on population data.

    Returns:
        signals (np.array): Signal value for all samples in all models
    """
    # Check if signals are available on disk
    signal_file_name = (
        f"{configs['audit']['algorithm'].lower()}_ramia_signals"
        if configs.get("ramia", None)
        else f"{configs['audit']['algorithm'].lower()}_signals"
    )
    signal_file_name += "_pop.npy" if is_population else ".npy"
    if os.path.exists(
        f"{configs['run']['log_dir']}/signals/{signal_file_name}",
    ):
        signals = np.load(
            f"{configs['run']['log_dir']}/signals/{signal_file_name}",
        )
        if configs.get("ramia", None) is None:
            expected_size = len(dataset)
            signal_source = "training data size"
        else:
            expected_size = len(dataset) * configs["ramia"]["sample_size"]
            signal_source = f"training data size multiplied by ramia sample size ({configs['ramia']['sample_size']})"

        if signals.shape[0] == expected_size:
            logger.info("Signals loaded from disk successfully.")
            return signals
        else:
            logger.warning(
                f"Signals shape ({signals.shape[0]}) does not match the expected size ({expected_size}). "
                f"This mismatch is likely due to a change in the {signal_source}."
            )
            logger.info("Ignoring the signals on disk and recomputing.")

    batch_size = configs["audit"]["batch_size"]  # Batch size used for inferring signals
    model_name = configs["train"]["model_name"]  # Algorithm used for training models
    device = configs["audit"]["device"]  # GPU device used for inferring signals
    if "tokenizer" in configs["data"].keys():
        tokenizer = AutoTokenizer.from_pretrained(
            configs["data"]["tokenizer"], clean_up_tokenization_spaces=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = None

    dataset_samples = np.arange(len(dataset))
    data, targets = load_dataset_subsets(
        dataset, dataset_samples, model_name, batch_size, device
    )

    signals = []
    logger.info("Computing signals for all models.")
    if configs.get("ramia", None) and not is_population:
        if len(data.shape) != 2:
            data = data.view(-1, *data.shape[2:])
            targets = targets.view(data.shape[0], -1)
    for model in models_list:
        signals.append(
            get_softmax(
                model, data, targets, batch_size, device, pad_token_id=pad_token_id
            )
        )

    signals = np.concatenate(signals, axis=1)
    os.makedirs(f"{configs['run']['log_dir']}/signals", exist_ok=True)
    np.save(
        f"{configs['run']['log_dir']}/signals/{signal_file_name}",
        signals,
    )
    logger.info("Signals saved to disk.")
    return signals
