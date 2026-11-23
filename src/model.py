from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def apply_vectorized_batch_augmentation(
    x: torch.Tensor,
    price_feature_count: int = 16,
    swap_prob: float = 0.15,
    jitter_std: float = 0.04,
) -> torch.Tensor:
    """
    On-the-fly vectorized GPU batch augmentation during training only:
    1. Protected Adjacent Candle Swaps:
       Randomly swap adjacent pairs (t-i, t-i-1) with p = 0.15,
       STRICTLY EXCLUDING the 3 most recent bars (t, t-1, t-2) -> indices [L-3, L-2, L-1].
    2. ATR-Scaled Jitter:
       Add Gaussian noise N(0, 0.04) to normalized price features.
    """
    B, L, C = x.shape
    x_aug = x.clone()

    # 1. Protected Adjacent Candle Swaps on indices [0 ... L-4]
    safe_len = L - 3  # Strictly exclude L-3, L-2, L-1 (the 3 most recent bars)
    if safe_len >= 2:
        offset = int(torch.randint(0, 2, (1,)).item())
        idx_a = torch.arange(offset, safe_len - 1, 2, device=x.device)
        idx_b = idx_a + 1
        num_pairs = len(idx_a)
        if num_pairs > 0:
            swap_mask = (
                torch.rand((B, num_pairs, 1), device=x.device) < swap_prob
            )
            candles_a = x_aug[:, idx_a, :]
            candles_b = x_aug[:, idx_b, :]
            x_aug[:, idx_a, :] = torch.where(swap_mask, candles_b, candles_a)
            x_aug[:, idx_b, :] = torch.where(swap_mask, candles_a, candles_b)

    # 2. ATR-Scaled Jitter N(0, 0.04) on normalized price features
    if jitter_std > 0.0:
        noise = torch.randn(
            (B, L, price_feature_count), device=x.device, dtype=x.dtype
        ) * jitter_std
        x_aug[:, :, :price_feature_count] = (
            x_aug[:, :, :price_feature_count] + noise
        )

    return x_aug


class ResBlock1D(nn.Module):
    """Compact residual 1D-CNN block with LayerNorm and Dropout(0.3)."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(channels)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        h = self.drop1(F.gelu(self.norm1(h)))
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop2(self.norm2(h))
        return F.gelu(residual + h)


class TripleBarrierProbClassifier(nn.Module):
    """
    Compact, heavily regularized 1D-CNN ResNet + BiGRU sequence classifier
    with LayerNorm and Dropout(0.3) predicting [logit_long, logit_short].
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 48,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block = ResBlock1D(hidden_dim, kernel_size=3, dropout=dropout)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_score = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_block(h)
        gru_out, _ = self.bigru(h)
        h = self.post_norm(h + self.dropout(gru_out))

        last_step = h[:, -1, :]
        recent_3 = h[:, -3:, :].mean(dim=1)
        attn_w = torch.softmax(self.attn_score(h), dim=1)
        context = torch.sum(h * attn_w, dim=1)

        combined = torch.cat([last_step, recent_3, context], dim=-1)
        logits = self.head(combined)
        return logits


class PlattTemperatureCalibrator(nn.Module):
    """
    Temperature Scaling (Platt Calibration) fitted strictly on the Validation set:
      cal_logit_k = (logit_k - mu_k) / T_k + mu_k + b_k
    where T_k > 0 scales logit dispersion around validation mean mu_k and b_k aligns
    mean validation probability with empirical validation win rate.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("logit_mean", torch.zeros(2))
        self.log_temp = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(min=0.4, max=1.5)
        centered = logits - self.logit_mean.unsqueeze(0)
        return (
            centered / temp.unsqueeze(0)
            + self.logit_mean.unsqueeze(0)
            + self.bias.unsqueeze(0)
        )

    def fit(self, val_logits: np.ndarray, val_targets: np.ndarray) -> Dict[str, List[float]]:
        device = torch.device("cpu")
        self.to(device)
        logits_t = torch.from_numpy(val_logits).float()
        targets_t = torch.from_numpy(val_targets).float()

        with torch.no_grad():
            self.logit_mean.copy_(logits_t.mean(dim=0))

        criterion = nn.BCEWithLogitsLoss()
        pre_loss = float(criterion(logits_t, targets_t).item())

        optimizer = torch.optim.LBFGS([self.log_temp, self.bias], lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            cal_logits = self.forward(logits_t)
            loss = criterion(cal_logits, targets_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            post_loss = float(criterion(self.forward(logits_t), targets_t).item())
            if post_loss > pre_loss:
                self.log_temp.zero_()
                self.bias.zero_()
                post_loss = pre_loss

            temps = torch.exp(self.log_temp).clamp(min=0.4, max=1.5).tolist()
            biases = self.bias.tolist()

        return {
            "val_logloss_pre_cal": pre_loss,
            "val_logloss_post_cal": post_loss,
            "temperatures": temps,
            "biases": biases,
        }

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_t = torch.from_numpy(logits).float()
            cal_logits = self.forward(logits_t)
            probs = torch.sigmoid(cal_logits).numpy()
        return probs


def predict_logits(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            logits = model(xb).cpu().numpy()
            preds.append(logits)
    return np.vstack(preds)


def train_and_calibrate_model(
    data_dict: Dict,
    timeframe: str,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    patience: int = 7,
    seed: int = 42,
) -> Dict:
    """
    Train the PyTorch sequence classifier with on-the-fly vectorized augmentation,
    CosineAnnealingLR, early stopping strictly on Validation Log-Loss, and
    Temperature Scaling (Platt Calibration) on the Validation set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    price_feat_count = len(data_dict["price_feature_indices"])

    # Stride non-overlapping/semi-overlapping windows per epoch (random phase shift each epoch)
    # so the model sees decorrelated trajectories per epoch instead of memorizing 48x overlapping bars in 1 pass
    if timeframe == "5m":
        stride = 4
    elif timeframe == "15m":
        stride = 2
    else:
        stride = 1

    model = TripleBarrierProbClassifier(
        num_features=X_train.shape[2], hidden_dim=48, dropout=0.3
    ).to(device)

    train_pos_rate = np.clip(y_train.mean(axis=0), 1e-3, 1.0 - 1e-3)
    init_bias = np.log(train_pos_rate / (1.0 - train_pos_rate))
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.from_numpy(init_bias).float().to(device))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0
    history = []

    print(
        f"\n[{timeframe}] Training PyTorch TripleBarrierProbClassifier on {device} "
        f"(Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,})"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        offset = (epoch - 1) % stride
        epoch_indices = np.arange(offset, len(X_train), stride)
        train_loader = DataLoader(
            SequenceDataset(X_train[epoch_indices], y_train[epoch_indices]),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        running_loss = 0.0
        n_obs = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb_aug = apply_vectorized_batch_augmentation(
                xb,
                price_feature_count=price_feat_count,
                swap_prob=0.15,
                jitter_std=0.04,
            )
            optimizer.zero_grad()
            logits = model(xb_aug)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(xb)
            n_obs += len(xb)

        scheduler.step()
        train_loss = running_loss / max(1, n_obs)

        val_logits = predict_logits(model, X_val, device)
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits).float(),
                torch.from_numpy(y_val).float(),
            ).item()
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            marker = "*"
        else:
            no_improve += 1
            marker = ""

        if epoch == 1 or epoch % 3 == 0 or marker == "*" or no_improve >= patience:
            print(
                f"  [{timeframe}] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train LogLoss: {train_loss:.5f} | Val LogLoss: {val_loss:.5f} {marker}"
            )

        if no_improve >= patience:
            print(
                f"  [{timeframe}] Early stopping triggered at epoch {epoch} "
                f"(best epoch {best_epoch}, Val LogLoss={best_val_loss:.5f})"
            )
            break

    model.load_state_dict(best_state)

    # Fit Centered Temperature Scaling (Platt Calibration) on Validation set only
    val_logits_best = predict_logits(model, X_val, device)
    calibrator = PlattTemperatureCalibrator()
    cal_info = calibrator.fit(val_logits_best, y_val)
    print(
        f"  [{timeframe}] Validation Calibration: LogLoss {cal_info['val_logloss_pre_cal']:.5f} -> "
        f"{cal_info['val_logloss_post_cal']:.5f} | "
        f"Temps={[round(t, 3) for t in cal_info['temperatures']]}, "
        f"Biases={[round(b, 4) for b in cal_info['biases']]}"
    )

    test_logits = predict_logits(model, X_test, device)
    test_probs_cal = calibrator.predict_proba(test_logits)
    test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))
    val_probs_cal = calibrator.predict_proba(val_logits_best)

    return {
        "model": model,
        "calibrator": calibrator,
        "calibration_info": cal_info,
        "history": history,
        "best_epoch": best_epoch,
        "val_probs": val_probs_cal,
        "test_probs": test_probs_cal,
        "test_probs_raw": test_probs_raw,
        "test_logits": test_logits,
    }
