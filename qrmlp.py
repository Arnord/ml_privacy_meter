# Author: zzx
# Quantile Regression with MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

def pinball_loss(pred, target, quantile):
    error = target - pred
    loss = torch.max(quantile * error, (quantile - 1) * error)
    return torch.mean(loss)


def train_qrmia_regressor_mlp(X_np, y_np, quantile=0.95, num_epochs=100, batch_size=64, lr=1e-3, device="mps"):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.float32).to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = QuantileRegressorMLP(input_dim=X.shape[1], quantile=quantile).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in dataloader:
            pred = model(xb)
            loss = pinball_loss(pred, yb, quantile)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch}: loss {total_loss:.4f}")
    return model


class QuantileRegressorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64], quantile=0.95):
        super().__init__()
        self.quantile = quantile
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 输出 shape: (batch,)