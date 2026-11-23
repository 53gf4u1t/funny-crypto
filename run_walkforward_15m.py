import copy
import json
import os
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import fetch_perp_ohlcv
from src.features_labels import (
    build_sliding_windows_numba,
    compute_causal_atr_numba,
    compute_rsi_numba,
    triple_barrier_labels_numba,
)
from src.model import (
    PlattTemperatureCalibrator,
    SequenceDataset,
    TripleBarrierProbClassifier,
    apply_vectorized_batch_augmentation,
    predict_logits,
)
from src.backtest import compute_calibration_diagnostics, run_futures_backtest


def build_raw_features_and_labels(
    df: pd.DataFrame,
    lookback_window: int = 64,
    max_hold: int = 48,
    tp_mult: float = 3.0,
    sl_mult: float = 1.5,
) -> Dict:
    """
    Compute strictly causal features and Numba triple barrier labels over the 15m series.
    Standardization is performed causally inside each walk-forward fold.
    """
    open_p = df["open"].to_numpy(dtype=np.float64)
    high_p = df["high"].to_numpy(dtype=np.float64)
    low_p = df["low"].to_numpy(dtype=np.float64)
    close_p = df["close"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)

    atr_14 = compute_causal_atr_numba(high_p, low_p, close_p, period=14)
    atr_100 = compute_causal_atr_numba(high_p, low_p, close_p, period=100)
    safe_atr = np.maximum(atr_14, close_p * 1e-6)
    atr_pct = safe_atr / close_p

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

    body = (close_p - open_p) / safe_atr
    upper_wick = (high_p - np.maximum(open_p, close_p)) / safe_atr
    lower_wick = (np.minimum(open_p, close_p) - low_p) / safe_atr
    candle_range = (high_p - low_p) / safe_atr

    close_s = pd.Series(close_p)
    log_c = np.log(np.maximum(close_p, 1e-8))
    ret_1 = (log_c - np.roll(log_c, 1)) / atr_pct
    ret_2 = (log_c - np.roll(log_c, 2)) / (atr_pct * np.sqrt(2.0))
    ret_4 = (log_c - np.roll(log_c, 4)) / (atr_pct * 2.0)
    ret_8 = (log_c - np.roll(log_c, 8)) / (atr_pct * np.sqrt(8.0))
    ret_16 = (log_c - np.roll(log_c, 16)) / (atr_pct * 4.0)
    ret_32 = (log_c - np.roll(log_c, 32)) / (atr_pct * np.sqrt(32.0))

    ema_9 = close_s.ewm(span=9, adjust=False).mean().to_numpy()
    ema_21 = close_s.ewm(span=21, adjust=False).mean().to_numpy()
    ema_50 = close_s.ewm(span=50, adjust=False).mean().to_numpy()
    ema_200 = close_s.ewm(span=200, adjust=False).mean().to_numpy()

    dist_ema_9 = (close_p - ema_9) / safe_atr
    dist_ema_21 = (close_p - ema_21) / safe_atr
    dist_ema_50 = (close_p - ema_50) / safe_atr
    dist_ema_200 = (close_p - ema_200) / safe_atr

    bb_mid = close_s.rolling(20, min_periods=1).mean().to_numpy()
    bb_std = close_s.rolling(20, min_periods=1).std().fillna(0.0).to_numpy()
    bb_width = (4.0 * bb_std) / safe_atr
    bb_pos = (close_p - bb_mid) / np.maximum(2.0 * bb_std, safe_atr * 0.1)

    rsi_14 = compute_rsi_numba(close_p, period=14)

    log_vol = pd.Series(np.log1p(np.maximum(vol, 0.0)))
    vol_mean = log_vol.rolling(100, min_periods=10).mean().bfill().to_numpy()
    vol_std = log_vol.rolling(100, min_periods=10).std().fillna(1.0).to_numpy()
    vol_z = (log_vol.to_numpy() - vol_mean) / np.maximum(vol_std, 1e-4)

    atr_ratio = (safe_atr / np.maximum(atr_100, close_p * 1e-6)) - 1.0
    log_atr_pct = pd.Series(np.log(np.maximum(atr_pct, 1e-6)))
    atr_z = (
        (log_atr_pct - log_atr_pct.rolling(200, min_periods=20).mean().bfill())
        / np.maximum(log_atr_pct.rolling(200, min_periods=20).std().fillna(1.0), 1e-4)
    ).to_numpy()

    signed_vol_flow = body * np.clip(vol_z, -3.0, 3.0)

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

    y_all = np.column_stack([y_long, y_short]).astype(np.float32)

    return {
        "raw_features": raw_features,
        "y_all": y_all,
        "price_feature_count": 16,
        "close": close_p,
        "atr": safe_atr,
        "atr_pct": atr_pct,
        "timestamps": df["timestamp"].to_numpy(),
        "exit_idx_long": exit_idx_long,
        "exit_idx_short": exit_idx_short,
        "outcome_long": outcome_long,
        "outcome_short": outcome_short,
        "lookback_window": lookback_window,
        "max_hold": max_hold,
    }


def sample_experience_replay_indices(
    train_bars: np.ndarray,
    recent_window_bars: int = 2880,
    max_samples_per_epoch: int = 4096,
    recent_weight: float = 0.50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Experience Replay Sampler:
    - Month 1: use all available training bars.
    - Months 2..23: sample 50% from the most recent month (fast adaptation) and
      50% uniformly from the older replay buffer (Months 1..k-1, preventing forgetting).
    """
    n_avail = len(train_bars)
    if n_avail <= recent_window_bars:
        return np.arange(n_avail, dtype=np.int64)

    split_pt = n_avail - recent_window_bars
    old_pool = np.arange(0, split_pt, dtype=np.int64)
    recent_pool = np.arange(split_pt, n_avail, dtype=np.int64)

    n_total = min(max_samples_per_epoch, n_avail)
    n_recent = min(len(recent_pool), int(n_total * recent_weight))
    n_old = min(len(old_pool), n_total - n_recent)

    chosen_recent = rng.choice(recent_pool, size=n_recent, replace=False)
    chosen_old = rng.choice(old_pool, size=n_old, replace=False)
    return np.concatenate([chosen_old, chosen_recent])


def run_monthly_walkforward_15m(
    asset: str,
    base_data: Dict,
    n_months: int = 24,
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    raw_features = base_data["raw_features"]
    y_all = base_data["y_all"]
    timestamps = base_data["timestamps"]
    lookback = base_data["lookback_window"]
    max_hold = base_data["max_hold"]
    embargo = lookback + max_hold  # 112 bars (28 hours on 15m)
    price_feat_count = base_data["price_feature_count"]

    warmup = max(200, lookback)
    last_valid_bar = len(raw_features) - max_hold - 1
    all_valid_bars = np.arange(warmup, last_valid_bar + 1, dtype=np.int64)

    month_blocks = np.array_split(all_valid_bars, n_months)

    model = TripleBarrierProbClassifier(
        num_features=raw_features.shape[1], hidden_dim=48, dropout=0.3
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    oos_bars_list = []
    oos_probs_list = []
    oos_naive_probs_list = []
    monthly_logs = []

    t0 = time.time()
    print(
        f"\n[{asset} 15m Walk-Forward] Starting 24-Month Continual Replay Walk-Forward on {device} "
        f"(Embargo={embargo} bars = 28h between Train/Val and Val/Test)"
    )

    for m_idx in range(1, n_months):
        test_month_bars = month_blocks[m_idx]
        test_start_bar = int(test_month_bars[0])
        test_end_bar = int(test_month_bars[-1])

        history_cutoff = test_start_bar - embargo
        avail_history_bars = all_valid_bars[all_valid_bars < history_cutoff]

        val_size = min(1440, max(480, int(len(avail_history_bars) * 0.22)))
        val_bars = avail_history_bars[-val_size:]
        train_cutoff = int(val_bars[0]) - embargo
        train_bars = avail_history_bars[avail_history_bars < train_cutoff]

        train_slice = raw_features[train_bars[0] - lookback + 1 : train_bars[-1] + 1]
        feat_mean = train_slice.mean(axis=0, keepdims=True)
        feat_std = np.maximum(train_slice.std(axis=0, keepdims=True), 1e-5)
        norm_features = ((raw_features - feat_mean) / feat_std).astype(np.float32)

        X_train_all = build_sliding_windows_numba(norm_features, train_bars, lookback)
        y_train_all = y_all[train_bars]
        X_val = build_sliding_windows_numba(norm_features, val_bars, lookback)
        y_val = y_all[val_bars]
        X_test_m = build_sliding_windows_numba(norm_features, test_month_bars, lookback)
        y_test_m = y_all[test_month_bars]

        if m_idx == 1:
            pos_rate = np.clip(y_train_all.mean(axis=0), 1e-3, 1.0 - 1e-3)
            init_bias = np.log(pos_rate / (1.0 - pos_rate))
            with torch.no_grad():
                model.head[-1].bias.copy_(torch.from_numpy(init_bias).float().to(device))
            epochs = 18
            lr = 6e-4
            patience = 5
            best_val_loss = float("inf")
        else:
            init_val_logits = predict_logits(model, X_val, device)
            best_val_loss = float(
                criterion(
                    torch.from_numpy(init_val_logits).float(),
                    torch.from_numpy(y_val).float(),
                ).item()
            )
            epochs = 10
            lr = 3e-4
            patience = 4

        best_state = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        no_improve = 0
        for epoch in range(1, epochs + 1):
            model.train()
            replay_idx = sample_experience_replay_indices(
                train_bars,
                recent_window_bars=2880,
                max_samples_per_epoch=4096,
                recent_weight=0.50,
                rng=rng,
            )
            loader = DataLoader(
                SequenceDataset(X_train_all[replay_idx], y_train_all[replay_idx]),
                batch_size=256,
                shuffle=True,
                drop_last=False,
            )
            for xb, yb in loader:
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

            scheduler.step()

            val_logits = predict_logits(model, X_val, device)
            val_loss = float(
                criterion(
                    torch.from_numpy(val_logits).float(),
                    torch.from_numpy(y_val).float(),
                ).item()
            )
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_state)

        val_logits_best = predict_logits(model, X_val, device)
        calibrator = PlattTemperatureCalibrator()
        cal_info = calibrator.fit(val_logits_best, y_val)

        test_logits_m = predict_logits(model, X_test_m, device)
        test_probs_m = calibrator.predict_proba(test_logits_m)

        prior_base_rate = np.clip(y_train_all.mean(axis=0, keepdims=True), 1e-7, 1.0 - 1e-7)
        naive_probs_m = np.repeat(prior_base_rate, len(test_month_bars), axis=0)

        oos_bars_list.append(test_month_bars)
        oos_probs_list.append(test_probs_m)
        oos_naive_probs_list.append(naive_probs_m)

        m_start_str = str(pd.Timestamp(timestamps[test_start_bar]))[:10]
        m_end_str = str(pd.Timestamp(timestamps[test_end_bar]))[:10]
        p_clip_m = np.clip(test_probs_m, 1e-7, 1.0 - 1e-7)
        m_logloss = float(
            -np.mean(y_test_m * np.log(p_clip_m) + (1.0 - y_test_m) * np.log(1.0 - p_clip_m))
        )
        n_signals_36 = int(((test_probs_m[:, 0] >= 0.36) | (test_probs_m[:, 1] >= 0.36)).sum())
        n_signals_38 = int(((test_probs_m[:, 0] >= 0.38) | (test_probs_m[:, 1] >= 0.38)).sum())

        monthly_logs.append(
            {
                "fold": m_idx,
                "test_month": f"M{m_idx+1:02d} ({m_start_str} -> {m_end_str})",
                "train_bars": int(len(train_bars)),
                "val_bars": int(len(val_bars)),
                "test_bars": int(len(test_month_bars)),
                "val_logloss": round(cal_info["val_logloss_post_cal"], 5),
                "test_logloss": round(m_logloss, 5),
                "signals_36": n_signals_36,
                "signals_38": n_signals_38,
            }
        )
        if m_idx % 6 == 0 or m_idx == 1 or m_idx == n_months - 1:
            print(
                f"  [{asset}] Fold {m_idx:02d}/{n_months-1:02d} | Test: {m_start_str}->{m_end_str} | "
                f"Replay Buffer={len(train_bars):,} bars | "
                f"Test LogLoss={m_logloss:.5f} | Signals(>=0.36/0.38)={n_signals_36}/{n_signals_38}"
            )

    print(f"[{asset} 15m Walk-Forward] Completed 23 out-of-sample monthly folds in {time.time() - t0:.1f}s")

    all_oos_bars = np.concatenate(oos_bars_list)
    all_oos_probs = np.vstack(oos_probs_list)
    all_oos_naive = np.vstack(oos_naive_probs_list)
    all_oos_y = y_all[all_oos_bars]

    return {
        "oos_bars": all_oos_bars,
        "oos_probs": all_oos_probs,
        "oos_naive": all_oos_naive,
        "oos_y": all_oos_y,
        "monthly_logs": monthly_logs,
    }


def evaluate_asset_walkforward(asset: str) -> Dict:
    df_15m = fetch_perp_ohlcv(base_asset=asset, timeframe="15m", days=730, cache_dir="data")
    base_data = build_raw_features_and_labels(
        df_15m, lookback_window=64, max_hold=48, tp_mult=3.0, sl_mult=1.5
    )

    cache_npz = f"outputs/wf_15m_{asset}_pure_replay.npz"
    if os.path.exists(cache_npz):
        loaded = np.load(cache_npz)
        if len(loaded["oos_bars"]) > 60000:
            print(f"[{asset}] Loaded cached 23-month walk-forward predictions from {cache_npz}")
            wf = {
                "oos_bars": loaded["oos_bars"],
                "oos_probs": loaded["oos_probs"],
                "oos_naive": loaded["oos_naive"],
                "oos_y": loaded["oos_y"],
                "monthly_logs": [],
            }
        else:
            wf = run_monthly_walkforward_15m(asset, base_data, n_months=24, seed=42)
            np.savez(
                cache_npz,
                oos_bars=wf["oos_bars"],
                oos_probs=wf["oos_probs"],
                oos_naive=wf["oos_naive"],
                oos_y=wf["oos_y"],
            )
    else:
        wf = run_monthly_walkforward_15m(asset, base_data, n_months=24, seed=42)
        np.savez(
            cache_npz,
            oos_bars=wf["oos_bars"],
            oos_probs=wf["oos_probs"],
            oos_naive=wf["oos_naive"],
            oos_y=wf["oos_y"],
        )

    oos_bars = wf["oos_bars"]
    oos_probs = wf["oos_probs"]
    oos_y = wf["oos_y"]

    wf_data_dict = {
        "y_train": oos_y,
        "y_test": oos_y,
        "test_bars": oos_bars,
        "close": base_data["close"],
        "atr": base_data["atr"],
        "atr_pct": base_data["atr_pct"],
        "timestamps": base_data["timestamps"],
        "exit_idx_long": base_data["exit_idx_long"],
        "exit_idx_short": base_data["exit_idx_short"],
        "outcome_long": base_data["outcome_long"],
        "outcome_short": base_data["outcome_short"],
    }

    cal_diag = compute_calibration_diagnostics(
        y_train=oos_y,
        y_test=oos_y,
        test_probs=oos_probs,
    )

    eps = 1e-7
    p_naive = np.clip(wf["oos_naive"], eps, 1.0 - eps)
    rolling_naive_logloss = float(
        -np.mean(oos_y * np.log(p_naive) + (1.0 - oos_y) * np.log(1.0 - p_naive))
    )
    rolling_naive_brier = float(np.mean((p_naive - oos_y) ** 2))

    friction_regimes = {
        "0_fee_0bp_slip": ("0 Fee + 0bp Slippage (0.00% drag)", 0.0, 0.0),
        "0_fee_1bp_rt_slip": ("0 Fee + 1bp Round-Trip Slippage (0.01% drag)", 0.0, 0.0001),
        "0_fee_1bp_side_slip": ("0 Fee + 1bp/Side Slippage (2bp RT = 0.02% drag)", 0.0, 0.0002),
        "5bp_fee_2bp_slip": ("5bp/Side Taker Fee + 2bp Slippage (0.12% drag)", 0.0005, 0.0002),
    }

    ts = base_data["timestamps"]
    oos_start_ts = pd.Timestamp(ts[oos_bars[0]])
    oos_end_ts = pd.Timestamp(ts[oos_bars[-1]])
    oos_years = (oos_end_ts - oos_start_ts).total_seconds() / (365.25 * 86400.0)
    mean_atr_pct = float(np.mean(base_data["atr_pct"][oos_bars]))

    print("\n" + "=" * 112)
    print(f"{asset}/USDT PERP — 15m CONTINUAL EXPERIENCE-REPLAY WALK-FORWARD (23 OOS MONTHS, ~{oos_years:.2f} YEARS)")
    print("=" * 112)
    print(
        f"Out-of-Sample Span: {oos_start_ts} -> {oos_end_ts} "
        f"({len(oos_bars):,} 15m bars = {oos_years:.2f} years) | Mean OOS ATR(14)%: {mean_atr_pct*100:.3f}%"
    )
    print(
        f"Out-of-Sample Base Win Rate: Long={cal_diag['test_base_rate_long']*100:.2f}%, "
        f"Short={cal_diag['test_base_rate_short']*100:.2f}%"
    )
    print(
        f"Probability Calibration:\n"
        f"  Walk-Forward Model Log-Loss : {cal_diag['test_logloss']:.5f} "
        f"(Long={cal_diag['test_logloss_long']:.5f}, Short={cal_diag['test_logloss_short']:.5f})\n"
        f"  Rolling Prior Naive Log-Loss: {rolling_naive_logloss:.5f} "
        f"(Delta = {rolling_naive_logloss - cal_diag['test_logloss']:+.5f})\n"
        f"  Walk-Forward Model Brier    : {cal_diag['test_brier']:.5f} vs. Rolling Naive Brier: "
        f"{rolling_naive_brier:.5f} (Delta = {rolling_naive_brier - cal_diag['test_brier']:+.5f})"
    )

    print(f"\n[{asset}] Binned Reliability Table (23-Month Out-of-Sample Combined Long & Short):")
    print(
        f"  {'Bin':<10} | {'Count':>8} | {'Mean Pred':>10} | {'Realized WR':>12} | {'Calib Gap':>10} | {'Long WR (n)':>18} | {'Short WR (n)':>18}"
    )
    print("  " + "-" * 96)
    for row in cal_diag["reliability_table"]:
        mp = f"{row['mean_pred_prob']*100:.2f}%" if not np.isnan(row["mean_pred_prob"]) else "N/A"
        rw = f"{row['realized_win_rate']*100:.2f}%" if not np.isnan(row["realized_win_rate"]) else "N/A"
        gp = f"{row['calibration_diff']*100:+.2f}%" if not np.isnan(row["calibration_diff"]) else "N/A"
        l_str = (
            f"{row['long_win_rate']*100:.1f}% ({row['long_count']})"
            if row["long_count"] > 0
            else "N/A (0)"
        )
        s_str = (
            f"{row['short_win_rate']*100:.1f}% ({row['short_count']})"
            if row["short_count"] > 0
            else "N/A (0)"
        )
        print(
            f"  {row['bin']:<10} | {row['count']:>8} | {mp:>10} | {rw:>12} | {gp:>10} | {l_str:>18} | {s_str:>18}"
        )

    regime_results = {}
    for r_key, (r_title, fee_side, slip_rt) in friction_regimes.items():
        regime_results[r_key] = {}
        print(f"\n  [{asset} 15m Walk-Forward ~{oos_years:.2f} Yrs] Regime: {r_title}")
        print(
            f"  {'Threshold':<10} | {'Trades (L/S)':>16} | {'TP WinRate':>11} | {'Total Return':>13} | {'Annual CAGR':>12} | {'Final Equity':>14} | {'PF':>6} | {'Sharpe':>7} | {'Max DD':>8}"
        )
        print("  " + "-" * 118)
        for th in [0.36, 0.38, 0.40]:
            bt = run_futures_backtest(
                data_dict=wf_data_dict,
                test_probs=oos_probs,
                threshold=th,
                initial_equity=10_000.0,
                risk_per_trade=0.02,
                sl_mult=1.5,
                tp_mult=3.0,
                taker_fee_per_side=fee_side,
                slippage_round_trip=slip_rt,
                timeframe="15m",
            )
            final_eq = bt["final_equity"]
            if final_eq > 0:
                cagr_pct = ((final_eq / 10_000.0) ** (1.0 / oos_years) - 1.0) * 100.0
            else:
                cagr_pct = -100.0
            bt["cagr_pct"] = cagr_pct
            regime_results[r_key][th] = bt

            tr_str = f"{bt['total_trades']} ({bt['long_trades']}/{bt['short_trades']})"
            print(
                f"  P >= {th:<5.2f} | {tr_str:>16} | {bt['tp_win_rate']:>10.2f}% | "
                f"{bt['net_compound_return_pct']:>+12.2f}% | {cagr_pct:>+11.2f}% | "
                f"${final_eq:>13,.2f} | {bt['profit_factor']:>6.2f} | "
                f"{bt['sharpe_ratio']:>7.2f} | {bt['max_drawdown_pct']:>7.2f}%"
            )

    return {
        "asset": asset,
        "oos_start": str(oos_start_ts),
        "oos_end": str(oos_end_ts),
        "oos_bars": int(len(oos_bars)),
        "oos_years": round(oos_years, 4),
        "mean_atr_pct": mean_atr_pct,
        "cal_diag": cal_diag,
        "rolling_naive_logloss": rolling_naive_logloss,
        "rolling_naive_brier": rolling_naive_brier,
        "regime_results": regime_results,
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    assets = ["ETH", "SOL"]
    all_res = {}
    for asset in assets:
        all_res[asset] = evaluate_asset_walkforward(asset)

    # Plot ETH vs SOL Walk-Forward Equity Curves & Reliability
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = {0.36: "#1f77b4", 0.38: "#2ca02c", 0.40: "#ff7f0e"}

    for col_idx, asset in enumerate(["ETH", "SOL"]):
        res = all_res[asset]
        oos_years = res["oos_years"]

        # Row 0: 0 Fee + 0bp Slippage
        ax0 = axes[0, col_idx]
        for th, col in colors.items():
            bt = res["regime_results"]["0_fee_0bp_slip"][th]
            ts_curve = pd.to_datetime(bt["equity_ts"])
            val_curve = bt["equity_val"]
            lbl = (
                f"P>={th:.2f} | n={bt['total_trades']} ({bt['long_trades']}L/{bt['short_trades']}S) | "
                f"WR={bt['tp_win_rate']:.1f}% | Tot={bt['net_compound_return_pct']:+.1f}% | CAGR={bt['cagr_pct']:+.1f}% (SR={bt['sharpe_ratio']:.2f})"
            )
            ax0.plot(ts_curve, val_curve, label=lbl, color=col, linewidth=1.8)
        ax0.axhline(10_000.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax0.set_title(f"{asset}/USDT 15m Walk-Forward ({oos_years:.2f} Yrs) — 0 Fee + 0bp Slippage")
        ax0.set_xlabel("Date (UTC)")
        ax0.set_ylabel("Account Equity (USDT)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best", fontsize=8.0)
        ax0.tick_params(axis="x", rotation=20)

        # Row 1: 0 Fee + 1bp Round-Trip Slippage
        ax1 = axes[1, col_idx]
        for th, col in colors.items():
            bt = res["regime_results"]["0_fee_1bp_rt_slip"][th]
            ts_curve = pd.to_datetime(bt["equity_ts"])
            val_curve = bt["equity_val"]
            lbl = (
                f"P>={th:.2f} | n={bt['total_trades']} ({bt['long_trades']}L/{bt['short_trades']}S) | "
                f"WR={bt['tp_win_rate']:.1f}% | Tot={bt['net_compound_return_pct']:+.1f}% | CAGR={bt['cagr_pct']:+.1f}% (SR={bt['sharpe_ratio']:.2f})"
            )
            ax1.plot(ts_curve, val_curve, label=lbl, color=col, linewidth=1.8)
        ax1.axhline(10_000.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.set_title(f"{asset}/USDT 15m Walk-Forward ({oos_years:.2f} Yrs) — 0 Fee + 1bp RT Slippage")
        ax1.set_xlabel("Date (UTC)")
        ax1.set_ylabel("Account Equity (USDT)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=8.0)
        ax1.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path = "outputs/walkforward_15m_ETH_SOL.png"
    plt.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved ETH & SOL walk-forward plot to: {plot_path}")

    summary_out = {}
    for asset, res in all_res.items():
        summary_out[asset] = {
            "oos_start": res["oos_start"],
            "oos_end": res["oos_end"],
            "oos_bars": res["oos_bars"],
            "oos_years": res["oos_years"],
            "mean_atr_pct": res["mean_atr_pct"],
            "test_logloss": res["cal_diag"]["test_logloss"],
            "rolling_naive_logloss": res["rolling_naive_logloss"],
            "test_brier": res["cal_diag"]["test_brier"],
            "rolling_naive_brier": res["rolling_naive_brier"],
            "reliability_table": res["cal_diag"]["reliability_table"],
            "regimes": {
                r_key: {
                    str(th): {
                        k: v
                        for k, v in bt.items()
                        if k not in ("trades", "equity_ts", "equity_val")
                    }
                    for th, bt in r_dict.items()
                }
                for r_key, r_dict in res["regime_results"].items()
            },
        }
    with open("outputs/walkforward_15m_ETH_SOL_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print("Saved ETH & SOL walk-forward JSON summary to: outputs/walkforward_15m_ETH_SOL_summary.json")


if __name__ == "__main__":
    main()
