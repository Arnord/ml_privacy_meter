"""This file is the main entry point for running the privacy auditing tool."""

import argparse
import math
import pdb
import time
import os
import re
import datetime
import pprint

import numpy as np
import torch
import yaml
from jupyter_server.transutils import base_dir
from torch.utils.data import Subset

from audit import get_average_audit_results, audit_models, sample_auditing_dataset
from get_signals import get_model_signals
from models.utils import load_models, train_models, split_dataset_for_training
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("Privacy Meter Tool!")
    print(20 * "-")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run privacy auditing tool.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/cifar10.yaml",      # 后期最好改成动态覆盖，方便跑实验
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration file
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Validate configurations
    check_configs(configs)

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories + send dr to result/
    log_dir = configs["run"]["log_dir"]
    if re.search(r"\d{8}_\d{6}", log_dir):
        log_dir = log_dir  # load model
    else:
        time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if configs["run"]["log_dir"] == "test":
            log_dir = os.path.join("result", configs["audit"]["algorithm"], "test", time_now)
        else:
            log_dir = os.path.join(base_dir, configs["audit"]["algorithm"], time_now)

    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )
    logger.info("Log directory: %s", log_dir)

    # Log the configuration
    logger.info("===== Loaded Configuration =====")
    logger.info("\n" + pprint.pformat(configs, width=120))
    logger.info("================================")


    start_time = time.time()

    # Load the dataset
    baseline_time = time.time()
    dataset, population = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    # models_list = None   # for test remember to del
    if models_list is None:
        # Split dataset for training two models per pair
        data_splits, memberships = split_dataset_for_training(
            len(dataset), num_model_pairs
        )
        models_list = train_models(
            log_dir, dataset, data_splits, memberships, configs, logger
        )
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    auditing_dataset, auditing_membership = sample_auditing_dataset(
        configs, dataset, logger, memberships
    )

    population = Subset(
        population,
        np.random.choice(
            len(population),
            configs["audit"].get("population_size", len(population)),
            replace=False,
        ),
    )

    # Generate signals (softmax outputs) for all models
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger)
    population_signals = get_model_signals(
        models_list, population, configs, logger, is_population=True
    )
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Perform the privacy audit
    baseline_time = time.time()
    target_model_indices = list(range(num_experiments))
    mia_score_list, membership_list = audit_models(
        f"{directories['report_dir']}/exp",
        target_model_indices,
        signals,
        population_signals,
        auditing_membership,
        num_reference_models,
        logger,
        configs,
    )

    if len(target_model_indices) > 1:
        logger.info(
            "Auditing privacy risk took %0.1f seconds", time.time() - baseline_time
        )

    # Get average audit results across all experiments
    if len(target_model_indices) > 1:
        get_average_audit_results(
            directories["report_dir"], mia_score_list, membership_list, logger
        )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
