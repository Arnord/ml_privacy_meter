from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import QuantileRegressor
from typing import Tuple, Optional

def get_rmia_out_signals(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> np.ndarray:
    """
    Get average prediction probability of samples over offline reference models (excluding the target model).

    Args:
        all_signals (np.ndarray): Softmax value of all samples in every model.
        all_memberships (np.ndarray): Membership matrix for all models (if a sample is used for training a model).
        target_model_idx (int): Target model index.
        num_reference_models (int): Number of reference models used for the attack.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    # Add non-target and non-paired model indices
    columns = [
        i
        for i in range(all_signals.shape[1])
        if i != target_model_idx and i != paired_model_idx
    ][: 2 * num_reference_models]
    selected_signals = all_signals[:, columns]
    non_members = ~all_memberships[:, columns]
    out_signals = selected_signals * non_members
    # Sort the signals such that only the non-zero signals (out signals) are kept
    out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    return out_signals


def tune_offline_a(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    logger: Any,
) -> (float, np.ndarray, np.ndarray):
    """
    Fine-tune coefficient offline_a used in RMIA.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in two models (target and reference).
        population_signals (np.ndarray): Population signals.
        all_memberships (np.ndarray): Membership matrix for all models.
        logger (Any): Logger object for the current run.

    Returns:
        float: Optimized offline_a obtained by attacking a paired model with the help of the reference models.
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    logger.info(f"Fine-tuning offline_a using paired model {paired_model_idx}")
    paired_memberships = all_memberships[:, paired_model_idx]
    offline_a = 0.0
    max_auc = 0
    for test_a in np.arange(0, 1.1, 0.1):
        mia_scores = run_rmia(
            paired_model_idx,
            all_signals,
            population_signals,
            all_memberships,
            1,
            test_a,
        )
        fpr_list, tpr_list, _ = roc_curve(
            paired_memberships.ravel(), mia_scores.ravel()
        )
        roc_auc = auc(fpr_list, tpr_list)
        if roc_auc > max_auc:
            max_auc = roc_auc
            offline_a = test_a
            mia_scores_array = mia_scores.ravel().copy()
            membership_array = paired_memberships.ravel().copy()
        logger.info(f"offline_a={test_a:.2f}: AUC {roc_auc:.4f}")
    return offline_a, mia_scores_array, membership_array


def compute_prob_ratio(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Compute the probability ratio for RMIA attack.
    The probability ratio is defined as:
        p(x) / p(z) = P_out(x) / P_out(z)
    where P_out(x) is the average prediction probability of x over OUT reference models.
    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in every model.
        population_signals (np.ndarray): Softmax value of all population samples in the target model.
        all_memberships (np.ndarray): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for the attack.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.
    Returns:
        np.ndarray: Probability ratio for all samples.
        np.ndarray: Probability ratio for target model samples.
        np.ndarray: Probability ratio for population samples.
    """
    target_signals = all_signals[:, target_model_idx]
    out_signals = get_rmia_out_signals(
        all_signals, all_memberships, target_model_idx, num_reference_models
    )
    mean_out_x = np.mean(out_signals, axis=1)
    mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2
    prob_ratio_x = target_signals.ravel() / mean_x

    z_signals = population_signals[:, target_model_idx]
    population_memberships = np.zeros_like(population_signals).astype(
        bool
    )  # All population data are OUT for all models
    z_out_signals = get_rmia_out_signals(
        population_signals,
        population_memberships,
        target_model_idx,
        num_reference_models,
    )
    mean_out_z = np.mean(z_out_signals, axis=1)
    mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2
    prob_ratio_z = z_signals.ravel() / mean_z

    log_x = np.log(prob_ratio_x + 1e-8)
    log_z = np.log(prob_ratio_z + 1e-8)
    ratios = log_x[:, np.newaxis] - log_z

    # ratios = prob_ratio_x[:, np.newaxis] / (prob_ratio_z + 1e-8)

    # Debugging information
    # print("prob_ratio_x:", np.percentile(prob_ratio_x, [0, 25, 50, 75, 100]))
    # print("prob_ratio_z:", np.percentile(prob_ratio_z, [0, 25, 50, 75, 100]))


    # return ratios, prob_ratio_x, prob_ratio_z
    return ratios, log_x, log_z


def run_rmia(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float,
    use_qrmia: bool = False,
    threshold_predictor: Any = None,
) -> np.ndarray:
    """
    Attack a target model using the RMIA attack with the help of offline reference models.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in the target model.
        population_signals (np.ndarray): Softmax value of all population samples in the target model.
        all_memberships (np.ndarray): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for the attack.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.
        use_qrmia (bool): Whether to use QRMIA or RMIA.
        threshold_predictor (Any): Trained quantile regressor for QRMIA.
    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    ratios, _, _ = compute_prob_ratio(
        target_model_idx,
        all_signals,
        population_signals,
        all_memberships,
        num_reference_models,
        offline_a,
    )

    # Debugging information
    # print("target_signals (sample):", target_signals[:5])
    # print("mean_x (sample):", mean_x[:5])
    # print("prob_ratio_x (sample):", prob_ratio_x[:5])
    # print("prob_ratio_z (sample):", prob_ratio_z[:5])
    # print("ratios (sample):", ratios[:5, :5])


    if use_qrmia and threshold_predictor is not None:
        target_signals = all_signals[:, target_model_idx].reshape(-1, 1)  # shape (N,1)
        lambda_x = threshold_predictor.predict(np.log(target_signals + 1e-8)) # shape (N,)
        lambda_x = lambda_x[:, np.newaxis]  # shape (N,1) for broadcasting
        counts = np.average(ratios > lambda_x, axis=1)
    else:
        counts = np.average(ratios > 1, axis=1)     # lamda  = 1

    return counts


def run_loss(target_signals: np.ndarray) -> np.ndarray:
    """
    Attack a target model using the LOSS attack.

    Args:
        target_signals (np.ndarray): Softmax value of all samples in the target model.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    mia_scores = -target_signals
    return mia_scores


def train_qrmia_regressor(
    population_signals: np.ndarray,
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
    offline_a: float,
    beta: float = 0.05,
    configs: Optional[dict] = None,
):
    """
    QRMIA-style training:
    - Use softmax-derived features as input.
    - Use RMIA scores under λ=1 as training target.
    - Fit a quantile regressor (minimize pinball loss).
    """

    # —— Step 1: Use population as non-members to compute RMIA scores ——
    # split population into two halves
    population_signals_x, population_signals_z = split_population(population_signals, train_ratio=0.8, seed=configs["run"]["random_seed"])

    S_x, _, _ = compute_prob_ratio(
        target_model_idx=target_model_idx,
        all_signals=population_signals_x,
        population_signals=population_signals_z,
        all_memberships=np.zeros_like(population_signals_x, dtype=bool),
        num_reference_models=num_reference_models,
        offline_a=offline_a,
    )

    # —— Step 2: Extract features φ(x) from target model's softmax outputs ——
    # 相当于用x的特征来预测对应RMIA S(x)的非成员阈值
    target_singles = population_signals_x[:, target_model_idx]  # shape (N,)

    X = np.log(target_singles.reshape(-1,1) + 1e-8)  # shape (N, 1)     # 可尝试扩展更多特征，即多分类的softmax，理论上效果会更好
    y = np.quantile(S_x, q=1-beta, axis=1) # shape (N,)
    # y = np.mean(S_x, axis=1)  # shape (N,)   # 直接用均值

    # —— Step 3: Quantile Regression using pinball loss ——
    qr = QuantileRegressor(
        quantile=1 - beta,
        alpha=0.0,          # no L2 penalty
        solver="highs"      # stable and accurate
    ).fit(X, y)

    return qr


def split_population(
    population_signals: np.ndarray,
    train_ratio: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 population_signals 随机拆分为 train / eval 两部分。
    Use for QRMIA中的分位数回归器训练
    Args:
        population_signals: shape (N, M, C)
        train_ratio:     用于训练回归器的比例（剩下用于 eval）
        seed:            随机种子，便于复现

    Returns:
        pop_train, pop_eval
    """
    N = population_signals.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    split = int(train_ratio * N)
    return population_signals[idx[:split]], population_signals[idx[split:]]