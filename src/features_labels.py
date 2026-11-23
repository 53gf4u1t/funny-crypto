from typing import Dict, Tuple
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def compute_causal_atr_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Strictly causal Wilder's ATR(period) computation using only past and current bars.
    """
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    atr = np.empty(n, dtype=np.float64)

    tr[0] = high[0] - low[0]
    atr[0] = tr[0]

    alpha = 1.0 / float(period)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
        if i < period:
            s = 0.0
            for j in range(i + 1):
                s += tr[j]
            atr[i] = s / float(i + 1)
        elif i == period:
            s = 0.0
            for j in range(1, period + 1):
                s += tr[j]
            atr[i] = s / float(period)
        else:
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


@njit(cache=True)
def triple_barrier_labels_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    tp_mult: float = 3.0,
    sl_mult: float = 1.5,
    max_hold: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated (@njit) first-passage Triple Barrier labeling for Long & Short.

    Rules:
    - Long Label (y_long = 1): Future High touches Close_t + 3.0 * ATR_t BEFORE
      Future Low touches Close_t - 1.5 * ATR_t within max_hold bars.
    - Short Label (y_short = 1): Future Low touches Close_t - 3.0 * ATR_t BEFORE
      Future High touches Close_t + 1.5 * ATR_t within max_hold bars.
    - Intra-bar conservatism: If both SL and TP are breached inside the exact same
      future candle, always count it as a loss (0) and record SL exit (-1).
    """
    n = len(close)
    y_long = np.zeros(n, dtype=np.int64)
    y_short = np.zeros(n, dtype=np.int64)
    exit_idx_long = np.zeros(n, dtype=np.int64)
    exit_idx_short = np.zeros(n, dtype=np.int64)
    outcome_long = np.zeros(n, dtype=np.int64)
    outcome_short = np.zeros(n, dtype=np.int64)

    valid_n = n - max_hold
    for t in range(valid_n):
        c_t = close[t]
        a_t = atr[t]
        if a_t <= 0.0 or np.isnan(a_t):
            exit_idx_long[t] = t + max_hold
            exit_idx_short[t] = t + max_hold
            continue

        long_tp = c_t + tp_mult * a_t
        long_sl = c_t - sl_mult * a_t

        short_tp = c_t - tp_mult * a_t
        short_sl = c_t + sl_mult * a_t

        # Scan forward for Long
        l_done = False
        for k in range(1, max_hold + 1):
            idx = t + k
            hit_sl = low[idx] <= long_sl
            hit_tp = high[idx] >= long_tp
            if hit_sl and hit_tp:
                y_long[t] = 0
                outcome_long[t] = -1
                exit_idx_long[t] = idx
                l_done = True
                break
            elif hit_sl:
                y_long[t] = 0
                outcome_long[t] = -1
                exit_idx_long[t] = idx
                l_done = True
                break
            elif hit_tp:
                y_long[t] = 1
                outcome_long[t] = 1
                exit_idx_long[t] = idx
                l_done = True
                break
        if not l_done:
            y_long[t] = 0
            outcome_long[t] = 0
            exit_idx_long[t] = t + max_hold

        # Scan forward for Short
        s_done = False
        for k in range(1, max_hold + 1):
            idx = t + k
            hit_sl = high[idx] >= short_sl
            hit_tp = low[idx] <= short_tp
            if hit_sl and hit_tp:
                y_short[t] = 0
                outcome_short[t] = -1
                exit_idx_short[t] = idx
                s_done = True
                break
            elif hit_sl:
                y_short[t] = 0
                outcome_short[t] = -1
                exit_idx_short[t] = idx
                s_done = True
                break
            elif hit_tp:
                y_short[t] = 1
                outcome_short[t] = 1
                exit_idx_short[t] = idx
                s_done = True
                break
        if not s_done:
            y_short[t] = 0
            outcome_short[t] = 0
            exit_idx_short[t] = t + max_hold

    return y_long, y_short, exit_idx_long, exit_idx_short, outcome_long, outcome_short


@njit(cache=True)
def compute_rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Strictly causal Wilder's RSI(period), scaled to [-1, 1]."""
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    avg_gain = 0.0
    avg_loss = 0.0
    alpha = 1.0 / float(period)

    for i in range(1, n):
        diff = close[i] - close[i - 1]
        gain = diff if diff > 0.0 else 0.0
        loss = -diff if diff < 0.0 else 0.0

        if i <= period:
            avg_gain += gain / float(period)
            avg_loss += loss / float(period)
            if i == period:
                if avg_loss == 0.0:
                    raw_rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    raw_rsi = 100.0 - (100.0 / (1.0 + rs))
                rsi[i] = (raw_rsi - 50.0) / 50.0
        else:
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
            if avg_loss <= 1e-12:
                raw_rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                raw_rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi[i] = (raw_rsi - 50.0) / 50.0
    return rsi


@njit(cache=True)
def build_sliding_windows_numba(
    features: np.ndarray, valid_indices: np.ndarray, lookback: int
) -> np.ndarray:
    """
    Assemble 3D tensor (N_valid, lookback, num_features) where sample k ending at
    bar t = valid_indices[k] contains bars [t - lookback + 1 ... t].
    """
    n_samples = len(valid_indices)
    n_feat = features.shape[1]
    out = np.empty((n_samples, lookback, n_feat), dtype=np.float32)
    for idx_i in range(n_samples):
        t = valid_indices[idx_i]
        start = t - lookback + 1
        for l in range(lookback):
            for f in range(n_feat):
                out[idx_i, l, f] = features[start + l, f]
    return out


def prepare_dataset_and_purged_splits(
    df: pd.DataFrame,
    lookback_window: int = 64,
    max_hold: int = 48,
    tp_mult: float = 3.0,
    sl_mult: float = 1.5,
) -> Dict:
    """
    Construct stationary ATR-normalized features, run Numba triple barrier labeling,
    and perform a strict chronological 70/15/15 split with an explicit embargo gap
    of `max_hold + lookback_window` bars between Train/Val and Val/Test.
    """
    open_p = df["open"].to_numpy(dtype=np.float64)
    high_p = df["high"].to_numpy(dtype=np.float64)
    low_p = df["low"].to_numpy(dtype=np.float64)
    close_p = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)

    # 1. Strictly causal ATR(14) and ATR(100)
    atr_14 = compute_causal_atr_numba(high_p, low_p, close_p, period=14)
    atr_100 = compute_causal_atr_numba(high_p, low_p, close_p, period=100)
    safe_atr = np.maximum(atr_14, close_p * 1e-6)
    atr_pct = safe_atr / close_p

    # 2. Vectorized Triple Barrier Labels (Numba @njit)
    (
        y_long,
        y_short,
        exit_idx_long,
        exit_idx_short,
        outcome_long,
        outcome_short,
    ) = triple_barrier_labels_numba(
        high_p, low_p, close_p, safe_atr, tp_mult=tp_mult, sl_mult=sl_mult, max_hold=max_hold
    )

    # 3. Stationary ATR-normalized features
    body = (close_p - open_p) / safe_atr
    upper_wick = (high_p - np.maximum(open_p, close_p)) / safe_atr
    lower_wick = (np.minimum(open_p, close_p) - low_p) / safe_atr
    candle_range = (high_p - low_p) / safe_atr

    # Multi-lag log returns normalized by ATR_pct
    close_s = pd.Series(close_p)
    log_c = np.log(np.maximum(close_p, 1e-8))
    ret_1 = (log_c - np.roll(log_c, 1)) / atr_pct
    ret_2 = (log_c - np.roll(log_c, 2)) / (atr_pct * np.sqrt(2.0))
    ret_4 = (log_c - np.roll(log_c, 4)) / (atr_pct * 2.0)
    ret_8 = (log_c - np.roll(log_c, 8)) / (atr_pct * np.sqrt(8.0))
    ret_16 = (log_c - np.roll(log_c, 16)) / (atr_pct * 4.0)
    ret_32 = (log_c - np.roll(log_c, 32)) / (atr_pct * np.sqrt(32.0))

    # ATR-normalized distances to EMAs (9, 21, 50, 200)
    ema_9 = close_s.ewm(span=9, adjust=False).mean().to_numpy()
    ema_21 = close_s.ewm(span=21, adjust=False).mean().to_numpy()
    ema_50 = close_s.ewm(span=50, adjust=False).mean().to_numpy()
    ema_200 = close_s.ewm(span=200, adjust=False).mean().to_numpy()

    dist_ema_9 = (close_p - ema_9) / safe_atr
    dist_ema_21 = (close_p - ema_21) / safe_atr
    dist_ema_50 = (close_p - ema_50) / safe_atr
    dist_ema_200 = (close_p - ema_200) / safe_atr

    # Bollinger Band (20, 2.0) width (ATR-normalized) and position
    bb_mid = close_s.rolling(20, min_periods=1).mean().to_numpy()
    bb_std = close_s.rolling(20, min_periods=1).std().fillna(0.0).to_numpy()
    bb_width = (4.0 * bb_std) / safe_atr
    bb_pos = (close_p - bb_mid) / np.maximum(2.0 * bb_std, safe_atr * 0.1)

    # RSI(14) scaled to [-1, 1]
    rsi_14 = compute_rsi_numba(close_p, period=14)

    # Normalized volume z-score (causal rolling 100-bar z-score of log1p(volume))
    log_vol = pd.Series(np.log1p(np.maximum(vol, 0.0)))
    vol_mean = log_vol.rolling(100, min_periods=10).mean().bfill().to_numpy()
    vol_std = log_vol.rolling(100, min_periods=10).std().fillna(1.0).to_numpy()
    vol_z = (log_vol.to_numpy() - vol_mean) / np.maximum(vol_std, 1e-4)

    # Short-to-long ATR volatility regime ratio (ATR_14 / ATR_100) and rolling ATR z-score
    atr_ratio = (safe_atr / np.maximum(atr_100, close_p * 1e-6)) - 1.0
    log_atr_pct = pd.Series(np.log(np.maximum(atr_pct, 1e-6)))
    atr_z = (
        (log_atr_pct - log_atr_pct.rolling(200, min_periods=20).mean().bfill())
        / np.maximum(log_atr_pct.rolling(200, min_periods=20).std().fillna(1.0), 1e-4)
    ).to_numpy()

    # Order-flow pressure (signed body * volume z-score)
    signed_vol_flow = body * np.clip(vol_z, -3.0, 3.0)

    feature_names = [
        "body_atr",
        "upper_wick_atr",
        "lower_wick_atr",
        "range_atr",
        "ret_1_atr",
        "ret_2_atr",
        "ret_4_atr",
        "ret_8_atr",
        "ret_16_atr",
        "ret_32_atr",
        "dist_ema9_atr",
        "dist_ema21_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "bb_width_atr",
        "bb_pos",
        "rsi_14",
        "vol_zscore",
        "atr_regime_ratio",
        "atr_pct_zscore",
        "signed_vol_flow",
    ]
    price_feature_indices = list(range(16))

    raw_features = np.column_stack(
        [
            body,
            upper_wick,
            lower_wick,
            candle_range,
            ret_1,
            ret_2,
            ret_4,
            ret_8,
            ret_16,
            ret_32,
            dist_ema_9,
            dist_ema_21,
            dist_ema_50,
            dist_ema_200,
            bb_width,
            bb_pos,
            rsi_14,
            vol_z,
            atr_ratio,
            atr_z,
            signed_vol_flow,
        ]
    )
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
    raw_features = np.clip(raw_features, -8.0, 8.0).astype(np.float32)

    warmup = max(200, lookback_window)
    last_valid_bar = len(df) - max_hold - 1
    all_valid_bars = np.arange(warmup, last_valid_bar + 1, dtype=np.int64)

    n_total = len(all_valid_bars)
    embargo = max_hold + lookback_window

    # Strict Purged Split: 70% Train / 15% Val / 15% Test with embargo gap = max_hold + lookback_window
    train_end = int(n_total * 0.70)
    val_start = train_end + embargo
    val_end = val_start + int(n_total * 0.15)
    test_start = val_end + embargo

    train_bars = all_valid_bars[:train_end]
    val_bars = all_valid_bars[val_start:val_end]
    test_bars = all_valid_bars[test_start:]

    train_feat_slice = raw_features[train_bars[0] - lookback_window + 1 : train_bars[-1] + 1]
    feat_mean = train_feat_slice.mean(axis=0, keepdims=True)
    feat_std = np.maximum(train_feat_slice.std(axis=0, keepdims=True), 1e-5)
    norm_features = ((raw_features - feat_mean) / feat_std).astype(np.float32)

    X_train = build_sliding_windows_numba(norm_features, train_bars, lookback_window)
    X_val = build_sliding_windows_numba(norm_features, val_bars, lookback_window)
    X_test = build_sliding_windows_numba(norm_features, test_bars, lookback_window)

    y_all = np.column_stack([y_long, y_short]).astype(np.float32)
    y_train = y_all[train_bars]
    y_val = y_all[val_bars]
    y_test = y_all[test_bars]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_bars": train_bars,
        "val_bars": val_bars,
        "test_bars": test_bars,
        "embargo_bars": embargo,
        "feature_names": feature_names,
        "price_feature_indices": price_feature_indices,
        "close": close_p,
        "atr": safe_atr,
        "atr_pct": atr_pct,
        "timestamps": df["timestamp"].to_numpy(),
        "exit_idx_long": exit_idx_long,
        "exit_idx_short": exit_idx_short,
        "outcome_long": outcome_long,
        "outcome_short": outcome_short,
    }
