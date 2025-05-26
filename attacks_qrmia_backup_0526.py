from typing import Any

import numpy as np

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.fft import ifftshift
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import PowerTransformer
from typing import Tuple, Optional

from torch.ao.nn.quantized.functional import threshold
from tornado.gen import multi, multi_future


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
    method: str = " ",
) -> (float, np.ndarray, np.ndarray):
    """
    Fine-tune coefficient offline_a used in RMIA.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in two models (target and reference).
        population_signals (np.ndarray): Population signals.
        all_memberships (np.ndarray): Membership matrix for all models.
        method (str): Method used for QRMIA attack (e.g., "sklearn", "mlp").
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
            method=method,
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
    y_transform: str = "none",
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
        y_transform (str): Transformation applied to the output signals. Options are "none", "log", "exp".
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

    # use_log = False

    def safe_logit(p, eps=1e-8):
        return np.log(p + eps) - np.log(1 - p + eps)

    y_transform = "log"

    if y_transform == "log":
        prob_ratio_x = np.log(prob_ratio_x + 1e-8)
        prob_ratio_z = np.log(prob_ratio_z + 1e-8)
        ratios = prob_ratio_x[:, np.newaxis] - prob_ratio_z   # TODO 获取不应该直接对比值作log运算，应该对softmax信号作log处理

    elif y_transform == "log1p":
        prob_ratio_x = np.log1p(prob_ratio_x)
        prob_ratio_z = np.log1p(prob_ratio_z)
        ratios = prob_ratio_x[:, np.newaxis] - prob_ratio_z

    elif y_transform == "logit":
        prob_ratio_x = safe_logit(prob_ratio_x)
        prob_ratio_z = safe_logit(prob_ratio_z)
        ratios = prob_ratio_x[:, np.newaxis] - prob_ratio_z

    elif y_transform == "standard":
        mean = np.mean(prob_ratio_x)
        std = np.std(prob_ratio_x) + 1e-8
        prob_ratio_x = (prob_ratio_x - mean) / std
        prob_ratio_z = (prob_ratio_z - mean) / std
        ratios = prob_ratio_x[:, np.newaxis] - prob_ratio_z

    elif y_transform == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson")
        prob_ratio_x = pt.fit_transform(prob_ratio_x.reshape(-1, 1)).flatten()
        prob_ratio_z = pt.transform(prob_ratio_z.reshape(-1, 1)).flatten()
        ratios = prob_ratio_x[:, np.newaxis] - prob_ratio_z

    elif y_transform == "none":
        ratios = prob_ratio_x[:, np.newaxis] / (prob_ratio_z + 1e-8)

    else:
        raise ValueError(f"Unsupported y_transform: {y_transform}")

    # if use_log:
    #     # log变换以使得softmax比值趋紧高斯分布
    #     prob_ratio_x = np.log(prob_ratio_x + 1e-8)
    #     prob_ratio_z = np.log(prob_ratio_z + 1e-8)
    #     ratios = prob_ratio_x[:, np.newaxis] - prob_ratio_z
    # else:
    #     ratios = prob_ratio_x[:, np.newaxis] / (prob_ratio_z + 1e-8)

    # Debugging information
    # print("prob_ratio_x:", np.percentile(prob_ratio_x, [0, 25, 50, 75, 100]))
    # print("prob_ratio_z:", np.percentile(prob_ratio_z, [0, 25, 50, 75, 100]))


    # return ratios, prob_ratio_x, prob_ratio_z
    return ratios, prob_ratio_x, prob_ratio_z


def run_rmia(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float,
    method: str = None,
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
        method (str): Method used for QRMIA attack (e.g., "sklearn", "mlp").
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
        # lambda_x = threshold_predictor.predict(np.log(target_signals + 1e-8)) # shape (N,)
        X = build_X_features(target_signals)
        lambda_x = predict_qrmia_lambda(model=threshold_predictor, X=X, method=method)
        lambda_x = lambda_x[:, np.newaxis]  # shape (N,1) for broadcasting

        plt.hist(np.mean(ratios, axis=1), bins=100)
        plt.title("Ratios distribution")
        plt.show()

        # plt审计
        plot_ratios_vs_lambda(ratios, lambda_x)

        counts = np.average(ratios > lambda_x, axis=1)   # 求的是x在z中大于阈值的比例
    else:
        counts = np.average(ratios > 1.0, axis=1)     # lamda  = 1

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
    auxiliary_signals: np.ndarray,
    population_signals: np.ndarray,
    target_model_idx: int,
    offline_a: float,
    num_reference_models: int,
    beta: float,
    method: str,      # "sklearn" or "mlp"
):
    """
    QRMIA-style training:
    - Use softmax-derived features as input.
    - Use RMIA scores under λ=1 as training target.
    - Fit a quantile regressor (minimize pinball loss).
    """

    # —— Step 1: Use population as non-members to compute RMIA scores ——
    # split population into two halves
    # population_signals_x, population_signals_z = split_population(population_signals, train_ratio=0.8, seed=configs["run"]["random_seed"])

    S_x, _, _ = compute_prob_ratio(
        target_model_idx=target_model_idx,
        all_signals=auxiliary_signals,
        population_signals=population_signals,
        all_memberships=np.zeros_like(auxiliary_signals, dtype=bool),
        num_reference_models=num_reference_models,
        offline_a=offline_a,
    )

    # —— Step 2: Extract features φ(x) from target model's softmax outputs ——
    # 相当于用x的特征来预测对应RMIA S(x)的非成员阈值
    target_singles = auxiliary_signals[:, target_model_idx]
    # z_signals = population_signals[:, target_model_idx]# shape (N,)
    X = build_X_features(target_singles)
    # X = np.log(target_singles.reshape(-1,1) + 1e-8)  # shape (N, 1)     # 可尝试扩展更多特征，即多分类的softmax，理论上效果会更好
    y = np.quantile(S_x, q=0.6, axis=1)  # 用中位数近似0.5， 0.6， 0.7 # shape (N,)
    # y = np.mean(S_x, axis=1)  # shape (N, )  TODO 均值可能会受到极端值影响
    # 验证y的分布
    plt.hist(y, bins=100)
    plt.title("Target λ(x) distribution_1")
    plt.show()

    # Step 1: 用 1% - 99% 分位点 clip 掉极端值
    lower = np.quantile(y, 0.01)
    upper = np.quantile(y, 0.99)
    y = np.clip(y, lower, upper)

    # # Step 2: Robust 标准化
    # from sklearn.preprocessing import RobustScaler
    # scaler = RobustScaler(quantile_range=(5, 95))
    # y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    plt.hist(y, bins=100)
    plt.title("Target λ(x) distribution_2")
    plt.show()

    # lambda_x 为目标变量，X 是你的特征
    for i in range(X.shape[1]):
        plt.scatter(X[:, i], y, alpha=0.2)
        plt.title(f"Feature {i} vs λₓ")
        plt.xlabel(f"Feature {i}")
        plt.ylabel("λₓ")
        plt.show()


    # 验证y的分布
    plt.hist(y, bins=100)
    plt.title("Target λ(x) distribution")
    plt.show()

    print("Training target y stats:", np.min(y), np.mean(y), np.max(y))
    # y = np.mean(S_x, axis=1)  # shape (N,)   # 直接用均值
    # y = (y - np.mean(y)) / (np.std(y) + 1e-8) # 标准化
    # plt.hist(y, bins=100)
    # plt.title("y = RMIA quantile target S(x)")
    # plt.show()

    # —— Step 3: Quantile Regression using pinball loss ——
    if method == "sklearn":
        # Use sklearn's QuantileRegressor
        qr = QuantileRegressor(
            quantile=1 - beta,  # 1 - beta
            alpha=0.0,          # no L2 penalty
            solver="highs"      # stable and accurate
        ).fit(X, y)
    elif method == "mlp":
        qr = train_qrmia_mlp_regressor(
            X, y,
            beta=beta,
            hidden_dims=(128, 64, 32),
            epochs=200,
            lr=1e-3,
            batch_size=64,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

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


def predict_qrmia_lambda(
        model, X: np.ndarray,
        method: str = "sklearn",
        device: str = "mps"
) -> np.ndarray:
    """
    Unified prediction interface for both sklearn and PyTorch quantile models.

    Args:
        model: trained quantile model (sklearn QuantileRegressor or PyTorch MLP)
        X (np.ndarray): input feature matrix, shape [N, D]
        method (str): "sklearn" or "mlp"
        device (str): device used for torch model, e.g. "cpu" or "cuda" or "mps"

    Returns:
        np.ndarray: predicted lambda_x, shape [N,]
    """
    if method == "sklearn":
        return model.predict(X)

    elif method == "mlp":
        use_gaussian = False
        beta = 0.1

        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            lambda_pred = model(X_tensor).squeeze()

            if use_gaussian:
                mu = lambda_pred[:, 0]
                log_std = lambda_pred[:, 1]
                std = torch.exp(log_std)
                quantile_val = math.sqrt(2) * torch.special.erfinv(torch.tensor(2 * (1 - beta) - 1.0).to(device))
                lambda_pred = mu + std * quantile_val

            lambda_pred = lambda_pred.cpu().numpy()
            # lambda_pred = np.clip(lambda_pred, -10, 10)

        return lambda_pred

    else:
        raise ValueError(f"Unsupported method: {method}")


def build_X_features(target_signals: np.ndarray) -> np.ndarray:
    """
    Build features for QRMIA training.
    Args:
        target_signals (np.ndarray): Softmax value of all samples in the target model.
    Returns:
        np.ndarray: Feature matrix for QRMIA training.
    """
    target_softmax = target_signals  # shape (N, 1)
    use_multi_features = False
    if use_multi_features:
        X = np.concatenate([
            np.log(target_softmax + 1e-8).reshape(-1, 1),  # log(softmax)
            target_softmax.reshape(-1, 1),  # softmax
            -np.log(target_softmax + 1e-8).reshape(-1, 1),  # -log(softmax)
        ], axis=1)
    else:
        # 只用 log(softmax)  # 会在后续的ratios计算过程进行log运算
        # X = np.log(target_softmax + 1e-8).reshape(-1, 1)
        X = target_softmax.reshape(-1, 1)  # softmax
    return X


class QRMIARegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 64, 32), scale: float = 5.0):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))  # 防止collapse
        layers.append(nn.Linear(dims[-1], output_dim))
        # layers.append(nn.Hardtanh(min_val=0, max_val=scale))  # 限幅 [-5, 5] 但这样会导致模型塌陷
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_qrmia_mlp_regressor(
    X: np.ndarray,
    y: np.ndarray,
    beta: float = 0.1,
    hidden_dims: Tuple[int, int, int] = (128, 64, 32),
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    scale: float = 5.0,
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
) -> nn.Module:
    """
    Train QRMIA MLP-based quantile regressor using pinball loss.
    """
    # ---- Setting ----
    use_gaussian = False

    # ---- DataLoader ----
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---- Model ----
    input_dim = X.shape[1]
    output_dim = 2 if use_gaussian  else 1
    model = QRMIARegressor(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, scale=scale).to(device)

    # ---- Pinball loss ----
    def pinball_loss_fn(preds, targets, q=1 - beta):
        diff = targets - preds.squeeze()
        return torch.mean(torch.maximum(q * diff, (q - 1) * diff))

    # ---- Gaussian loss ----
    def gaussian_loss_fn(preds, targets):
        mu = preds[:, 0]
        log_std = preds[:, 1]
        return torch.mean(log_std + 0.5 * torch.exp(-2 * log_std) * (targets - mu) ** 2)

    # ---- Define loss function ----
    if use_gaussian:
        loss_fn = gaussian_loss_fn
    else:
        loss_fn = pinball_loss_fn


    # ---- Optimizer ----
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---- Training loop ----
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Epoch {epoch+1:02d}] Loss = {total_loss:.4f}")

    # ---- Evaluate ----
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device))
        loss = loss_fn(preds, y_tensor.to(device))
        print(f"Final evaluation loss: {loss.item():.4f}")

    plot_y_vs_pred(y_tensor, preds)

    return model


def plot_y_vs_pred(y: torch.Tensor, preds: torch.Tensor):
    """
    Plot the true vs predicted values for QRMIA.

    Args:
        y (np.ndarray): True values.
        preds (np.ndarray): Predicted values.
    """
    # 从预测输出中提取 μ（即预测 λ_pred）
    use_gaussian = False

    if use_gaussian:
        preds = preds[:, 0].detach().cpu().numpy()
    else:
        preds = preds.detach().cpu().numpy()

    y_true = y.cpu().numpy()

    from sklearn.metrics import r2_score
    print("R² score:", r2_score(y_true, preds))

    # 分布对比图：直方图
    plt.figure(figsize=(10, 5))
    plt.hist(y_true, bins=100, alpha=0.6, label="True λ(x)", density=True)
    plt.hist(preds, bins=100, alpha=0.6, label="Predicted μ(x)", density=True)
    plt.title("Distribution of true λ(x) and predicted μ(x)")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 回归拟合散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, preds, alpha=0.3, s=10)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal fit")
    plt.xlabel("True λ(x)")
    plt.ylabel("Predicted μ(x)")
    plt.title("Predicted vs True λ(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_ratios_vs_lambda(ratios: np.ndarray, lambda_x: np.ndarray, bins: int = 100):
    """
    Plot the distribution of log-ratios and predicted lambda values for QRMIA.

    Args:
        ratios (np.ndarray): shape (N, K), where K is number of z per x
        lambda_x (np.ndarray): shape (N,) or (N, 1)
        bins (int): number of bins for histogram
    """
    # Flatten ratios to get the empirical S(x) distribution over all (x,z)
    # ratios_flat = ratios.flatten()
    ratios = np.mean(ratios, axis=1)


    # Squeeze lambda_x to 1D if needed
    lambda_x = lambda_x.squeeze()

    # Plot 1
    plt.figure(figsize=(10, 5))
    plt.hist(lambda_x, bins=100)
    plt.title("λ(x) predicted by MLP")
    plt.show()

    # Plot 2
    plt.figure(figsize=(10, 5))
    diff = ratios - lambda_x.squeeze()
    plt.hist(diff, bins=100)
    plt.title("S(x) - λ(x) 's distribution")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.show()

    # Plot 3
    plt.figure(figsize=(10, 5))
    plt.hist(ratios, bins=bins, alpha=0.6, label="log-ratio S(x)", density=True, color="C0")
    plt.hist(lambda_x, bins=bins, alpha=0.6, label="Predicted λ(x)", density=True, color="C1")
    plt.axvline(x=0.0, linestyle='--', color='gray', label='λ = log(1) = 0')
    plt.xlabel("Score / Threshold Value")
    plt.ylabel("Density")
    plt.title("Distribution of log-ratios (S(x)) and predicted λ(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_full_sx_distributions(S_x_non: np.ndarray, S_x_all: np.ndarray, bins: int = 100):
    """
    S_x_non: [N_x_non, N_z] matrix from population-only pairs
    S_x_all: [N_x_all, N_z] matrix including members and non-members

    Plots the flattened value distributions of both S_x
    """
    plt.figure(figsize=(10, 5))
    plt.hist(S_x_non.flatten(), bins=bins, alpha=0.6, label="S(x)_non", density=True)
    plt.hist(S_x_all.flatten(), bins=bins, alpha=0.6, label="S(x)_all", density=True)
    plt.xlabel("RMIA score")
    plt.ylabel("Density")
    plt.title("Flattened RMIA S(x) distribution over (x, z) pairs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()