from typing import Dict, List
import numpy as np
import pandas as pd


def compute_calibration_diagnostics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
) -> Dict:
    """
    Evaluate Test Log-Loss vs. Naive Baseline Log-Loss, Brier Score, and
    a binned Reliability Table ([0.30-0.35, 0.35-0.40, 0.40-0.45, >0.45]).
    """
    eps = 1e-7
    p_clip = np.clip(test_probs, eps, 1.0 - eps)

    # Naive baseline predicts the training empirical base rate for each direction
    base_rate = np.clip(y_train.mean(axis=0, keepdims=True), eps, 1.0 - eps)
    naive_probs = np.repeat(base_rate, len(y_test), axis=0)

    # Binary cross-entropy log-loss
    model_logloss_per_col = -np.mean(
        y_test * np.log(p_clip) + (1.0 - y_test) * np.log(1.0 - p_clip), axis=0
    )
    naive_logloss_per_col = -np.mean(
        y_test * np.log(naive_probs) + (1.0 - y_test) * np.log(1.0 - naive_probs),
        axis=0,
    )

    model_logloss = float(np.mean(model_logloss_per_col))
    naive_logloss = float(np.mean(naive_logloss_per_col))

    # Brier score
    model_brier_per_col = np.mean((test_probs - y_test) ** 2, axis=0)
    naive_brier_per_col = np.mean((naive_probs - y_test) ** 2, axis=0)
    model_brier = float(np.mean(model_brier_per_col))
    naive_brier = float(np.mean(naive_brier_per_col))

    # Binned reliability table across combined (Long + Short) and individual heads
    bins = [
        ("<0.30", 0.00, 0.30),
        ("0.30-0.35", 0.30, 0.35),
        ("0.35-0.40", 0.35, 0.40),
        ("0.40-0.45", 0.40, 0.45),
        (">0.45", 0.45, 1.01),
    ]

    flat_p = test_probs.reshape(-1)
    flat_y = y_test.reshape(-1)

    reliability_rows = []
    for label, lo, hi in bins:
        mask = (flat_p >= lo) & (flat_p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            pred_mean = float(flat_p[mask].mean())
            real_wr = float(flat_y[mask].mean())
            gap = real_wr - pred_mean
        else:
            pred_mean = float("nan")
            real_wr = float("nan")
            gap = float("nan")

        # Also compute Long and Short counts/win-rates in this bin
        mask_l = (test_probs[:, 0] >= lo) & (test_probs[:, 0] < hi)
        mask_s = (test_probs[:, 1] >= lo) & (test_probs[:, 1] < hi)
        cnt_l = int(mask_l.sum())
        cnt_s = int(mask_s.sum())
        wr_l = float(y_test[mask_l, 0].mean()) if cnt_l > 0 else float("nan")
        wr_s = float(y_test[mask_s, 1].mean()) if cnt_s > 0 else float("nan")

        reliability_rows.append(
            {
                "bin": label,
                "count": cnt,
                "mean_pred_prob": pred_mean,
                "realized_win_rate": real_wr,
                "calibration_diff": gap,
                "long_count": cnt_l,
                "long_win_rate": wr_l,
                "short_count": cnt_s,
                "short_win_rate": wr_s,
            }
        )

    return {
        "test_logloss": model_logloss,
        "naive_logloss": naive_logloss,
        "logloss_improvement": naive_logloss - model_logloss,
        "test_logloss_long": float(model_logloss_per_col[0]),
        "test_logloss_short": float(model_logloss_per_col[1]),
        "naive_logloss_long": float(naive_logloss_per_col[0]),
        "naive_logloss_short": float(naive_logloss_per_col[1]),
        "test_brier": model_brier,
        "naive_brier": naive_brier,
        "brier_improvement": naive_brier - model_brier,
        "train_base_rate_long": float(base_rate[0, 0]),
        "train_base_rate_short": float(base_rate[0, 1]),
        "test_base_rate_long": float(y_test[:, 0].mean()),
        "test_base_rate_short": float(y_test[:, 1].mean()),
        "reliability_table": reliability_rows,
    }


def run_futures_backtest(
    data_dict: Dict,
    test_probs: np.ndarray,
    threshold: float = 0.38,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.02,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    taker_fee_per_side: float = 0.0005,  # 0.05% per side = 0.10% round-trip
    slippage_round_trip: float = 0.0002,  # 0.02% slippage
    timeframe: str = "5m",
) -> Dict:
    """
    Compound Equity Simulation on the untouched Test set:
    - 2% fixed fractional account risk per trade (1.5x ATR SL / 3.0x ATR TP -> 2:1 R:R).
    - Enter only when calibrated P(Win) >= threshold.
    - Max 1 concurrent open position (no overlapping trades until active trade hits SL, TP, or max_hold timeout).
    - Dynamic position sizing: Pos_Notional = (0.02 * Equity) / (1.5 * ATR_pct).
    - Deduct exact fee & slippage drag: (2 * 0.05% + 0.02%) = 0.12% of Pos_Notional directly from trade PnL.
    """
    test_bars = data_dict["test_bars"]
    close = data_dict["close"]
    atr_pct = data_dict["atr_pct"]
    timestamps = data_dict["timestamps"]
    exit_idx_long = data_dict["exit_idx_long"]
    exit_idx_short = data_dict["exit_idx_short"]
    outcome_long = data_dict["outcome_long"]
    outcome_short = data_dict["outcome_short"]

    round_trip_cost_rate = 2.0 * taker_fee_per_side + slippage_round_trip  # 0.0012 (0.12%)

    equity = initial_equity
    peak_equity = initial_equity
    max_drawdown = 0.0

    next_free_bar = -1
    trades: List[Dict] = []
    equity_trace_ts = [timestamps[test_bars[0]]]
    equity_trace_val = [initial_equity]

    for i, t in enumerate(test_bars):
        # Enforce max 1 concurrent open position
        if t <= next_free_bar:
            continue

        p_long = float(test_probs[i, 0])
        p_short = float(test_probs[i, 1])

        if p_long < threshold and p_short < threshold:
            continue

        # Pick direction with higher calibrated win probability
        if p_long >= p_short:
            direction = 1  # Long
            p_win = p_long
            exit_bar = int(exit_idx_long[t])
            outcome = int(outcome_long[t])
        else:
            direction = -1  # Short
            p_win = p_short
            exit_bar = int(exit_idx_short[t])
            outcome = int(outcome_short[t])

        atr_p = float(atr_pct[t])
        if atr_p <= 1e-6:
            continue

        # Exact position notional formula: Pos_Notional = (0.02 * Equity) / (1.5 * ATR_pct)
        pos_notional = (risk_per_trade * equity) / (sl_mult * atr_p)

        # Gross trade PnL
        if outcome == 1:
            # TP hit (+3.0 * ATR_pct * Pos_Notional = +4% of Equity)
            gross_pnl = pos_notional * (tp_mult * atr_p)
            is_win = 1
        elif outcome == -1:
            # SL hit (-1.5 * ATR_pct * Pos_Notional = -2% of Equity)
            gross_pnl = -pos_notional * (sl_mult * atr_p)
            is_win = 0
        else:
            # Timeout at max_hold bar: mark to market at close[exit_bar]
            raw_ret = (close[exit_bar] - close[t]) / close[t] * direction
            gross_pnl = pos_notional * raw_ret
            is_win = 1 if gross_pnl > 0 else 0

        # Exact fee & slippage drag on Pos_Notional
        fee_slippage_drag = pos_notional * round_trip_cost_rate
        net_pnl = gross_pnl - fee_slippage_drag
        net_ret_pct = net_pnl / equity

        equity = max(0.0, equity + net_pnl)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

        next_free_bar = exit_bar

        trades.append(
            {
                "entry_bar": int(t),
                "exit_bar": int(exit_bar),
                "entry_time": timestamps[t],
                "exit_time": timestamps[min(exit_bar, len(timestamps) - 1)],
                "direction": "LONG" if direction == 1 else "SHORT",
                "prob": p_win,
                "outcome": outcome,
                "is_tp_win": 1 if outcome == 1 else 0,
                "is_net_win": 1 if net_pnl > 0 else 0,
                "atr_pct": atr_p,
                "leverage": pos_notional / max(1e-8, equity - net_pnl),
                "pos_notional": pos_notional,
                "gross_pnl": gross_pnl,
                "fee_drag": fee_slippage_drag,
                "net_pnl": net_pnl,
                "net_ret_pct": net_ret_pct,
                "equity_after": equity,
            }
        )
        equity_trace_ts.append(timestamps[min(exit_bar, len(timestamps) - 1)])
        equity_trace_val.append(equity)

        if equity <= 0.0:
            break

    n_trades = len(trades)
    if n_trades == 0:
        return {
            "threshold": threshold,
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "tp_win_rate": 0.0,
            "net_win_rate": 0.0,
            "net_compound_return_pct": 0.0,
            "final_equity": initial_equity,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_fees_paid": 0.0,
            "avg_leverage": 0.0,
            "trades": [],
            "equity_ts": equity_trace_ts,
            "equity_val": equity_trace_val,
        }

    tp_wins = sum(t["is_tp_win"] for t in trades)
    net_wins = sum(t["is_net_win"] for t in trades)
    long_trades = sum(1 for t in trades if t["direction"] == "LONG")
    short_trades = n_trades - long_trades

    gross_profits = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = (
        gross_profits / gross_losses
        if gross_losses > 1e-8
        else (float("inf") if gross_profits > 0 else 0.0)
    )

    # Annualized Sharpe Ratio based on test span duration and per-trade net percentage returns
    rets = np.array([t["net_ret_pct"] for t in trades], dtype=np.float64)
    test_start_ts = pd.Timestamp(timestamps[test_bars[0]])
    test_end_ts = pd.Timestamp(timestamps[test_bars[-1]])
    test_years = max((test_end_ts - test_start_ts).total_seconds() / (365.25 * 86400.0), 1e-3)
    trades_per_year = n_trades / test_years

    if len(rets) > 1 and np.std(rets, ddof=1) > 1e-9:
        sharpe = float((np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    net_compound_return_pct = ((equity / initial_equity) - 1.0) * 100.0
    total_fees = sum(t["fee_drag"] for t in trades)
    avg_lev = float(np.mean([t["leverage"] for t in trades]))

    return {
        "threshold": threshold,
        "total_trades": n_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "tp_win_rate": (tp_wins / n_trades) * 100.0,
        "net_win_rate": (net_wins / n_trades) * 100.0,
        "net_compound_return_pct": net_compound_return_pct,
        "final_equity": equity,
        "profit_factor": float(profit_factor),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100.0,
        "total_fees_paid": float(total_fees),
        "avg_leverage": avg_lev,
        "trades": trades,
        "equity_ts": equity_trace_ts,
        "equity_val": equity_trace_val,
    }


def run_confidence_scaled_backtest(
    data_dict: Dict,
    test_probs: np.ndarray,
    threshold: float = 0.38,
    prob_span: float = 0.12,
    min_risk: float = 0.01,
    max_risk: float = 0.02,
    initial_equity: float = 10_000.0,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    taker_fee_per_side: float = 0.0005,
    slippage_round_trip: float = 0.0002,
) -> Dict:
    """
    Confidence-Scaled Compound Equity Simulation:
    - Enter when calibrated P(Win) >= threshold.
    - Scale account risk linearly from `min_risk` (1.0%) at `P(Win) = threshold`
      up to `max_risk` (2.0%) at `P(Win) >= threshold + prob_span` (e.g. [0.38, 0.50]),
      matching the empirical [0.20, 0.55] operating range of the model.
    - Pos_Notional = (risk_frac * Equity) / (sl_mult * ATR_pct).
    """
    test_bars = data_dict["test_bars"]
    close = data_dict["close"]
    atr_pct = data_dict["atr_pct"]
    timestamps = data_dict["timestamps"]
    exit_idx_long = data_dict["exit_idx_long"]
    exit_idx_short = data_dict["exit_idx_short"]
    outcome_long = data_dict["outcome_long"]
    outcome_short = data_dict["outcome_short"]

    round_trip_cost_rate = 2.0 * taker_fee_per_side + slippage_round_trip

    equity = initial_equity
    peak_equity = initial_equity
    max_drawdown = 0.0

    next_free_bar = -1
    trades: List[Dict] = []
    equity_trace_ts = [timestamps[test_bars[0]]]
    equity_trace_val = [initial_equity]

    for i, t in enumerate(test_bars):
        if t <= next_free_bar:
            continue

        p_long = float(test_probs[i, 0])
        p_short = float(test_probs[i, 1])

        if p_long < threshold and p_short < threshold:
            continue

        if p_long >= p_short:
            direction = 1
            p_win = p_long
            exit_bar = int(exit_idx_long[t])
            outcome = int(outcome_long[t])
        else:
            direction = -1
            p_win = p_short
            exit_bar = int(exit_idx_short[t])
            outcome = int(outcome_short[t])

        atr_p = float(atr_pct[t])
        if atr_p <= 1e-6:
            continue

        # Linear confidence scaling within [threshold, threshold + prob_span] -> [min_risk, max_risk]
        conf_alpha = min(1.0, max(0.0, (p_win - threshold) / max(1e-6, prob_span)))
        risk_per_trade = min_risk + (max_risk - min_risk) * conf_alpha

        pos_notional = (risk_per_trade * equity) / (sl_mult * atr_p)

        if outcome == 1:
            gross_pnl = pos_notional * (tp_mult * atr_p)
        elif outcome == -1:
            gross_pnl = -pos_notional * (sl_mult * atr_p)
        else:
            raw_ret = (close[exit_bar] - close[t]) / close[t] * direction
            gross_pnl = pos_notional * raw_ret

        fee_slippage_drag = pos_notional * round_trip_cost_rate
        net_pnl = gross_pnl - fee_slippage_drag
        net_ret_pct = net_pnl / equity

        equity = max(0.0, equity + net_pnl)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

        next_free_bar = exit_bar

        trades.append(
            {
                "entry_bar": int(t),
                "exit_bar": int(exit_bar),
                "direction": "LONG" if direction == 1 else "SHORT",
                "prob": p_win,
                "risk_pct": risk_per_trade * 100.0,
                "outcome": outcome,
                "is_tp_win": 1 if outcome == 1 else 0,
                "is_net_win": 1 if net_pnl > 0 else 0,
                "net_pnl": net_pnl,
                "net_ret_pct": net_ret_pct,
                "fee_drag": fee_slippage_drag,
                "leverage": pos_notional / max(1e-8, equity - net_pnl),
            }
        )
        equity_trace_ts.append(timestamps[min(exit_bar, len(timestamps) - 1)])
        equity_trace_val.append(equity)

        if equity <= 0.0:
            break

    n_trades = len(trades)
    if n_trades == 0:
        return {
            "threshold": threshold,
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "tp_win_rate": 0.0,
            "avg_risk_pct": 0.0,
            "net_compound_return_pct": 0.0,
            "final_equity": initial_equity,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "total_fees_paid": 0.0,
            "equity_ts": equity_trace_ts,
            "equity_val": equity_trace_val,
        }

    tp_wins = sum(t["is_tp_win"] for t in trades)
    long_trades = sum(1 for t in trades if t["direction"] == "LONG")
    short_trades = n_trades - long_trades

    gross_profits = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = (
        gross_profits / gross_losses
        if gross_losses > 1e-8
        else (float("inf") if gross_profits > 0 else 0.0)
    )

    rets = np.array([t["net_ret_pct"] for t in trades], dtype=np.float64)
    test_start_ts = pd.Timestamp(timestamps[test_bars[0]])
    test_end_ts = pd.Timestamp(timestamps[test_bars[-1]])
    test_years = max((test_end_ts - test_start_ts).total_seconds() / (365.25 * 86400.0), 1e-3)
    trades_per_year = n_trades / test_years

    if len(rets) > 1 and np.std(rets, ddof=1) > 1e-9:
        sharpe = float((np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    net_compound_return_pct = ((equity / initial_equity) - 1.0) * 100.0
    total_fees = sum(t["fee_drag"] for t in trades)
    avg_risk_pct = float(np.mean([t["risk_pct"] for t in trades]))

    return {
        "threshold": threshold,
        "total_trades": n_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "tp_win_rate": (tp_wins / n_trades) * 100.0,
        "avg_risk_pct": avg_risk_pct,
        "net_compound_return_pct": net_compound_return_pct,
        "final_equity": equity,
        "profit_factor": float(profit_factor),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100.0,
        "total_fees_paid": float(total_fees),
        "equity_ts": equity_trace_ts,
        "equity_val": equity_trace_val,
    }
