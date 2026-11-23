import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import fetch_btc_perp_ohlcv, aggregate_15m_to_1h
from src.features_labels import prepare_dataset_and_purged_splits
from src.model import train_and_calibrate_model
from src.backtest import compute_calibration_diagnostics, run_futures_backtest

TIMEFRAMES = ["5m", "15m", "1h"]


def plot_diagnostics(results: dict, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    colors = {0.36: "#1f77b4", 0.38: "#2ca02c", 0.40: "#ff7f0e"}

    # 1. Equity Curve Comparison Plot (5m vs 15m vs 1h under realistic 5bp/side fee + 2bp slippage)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    for ax, tf in zip(axes, TIMEFRAMES):
        tf_res = results[tf]
        for th, col in colors.items():
            bt = tf_res["backtests"][th]
            ts = pd.to_datetime(bt["equity_ts"])
            val = bt["equity_val"]
            label = (
                f"P >= {th:.2f} | Trades={bt['total_trades']} | "
                f"WR={bt['tp_win_rate']:.1f}% | Net={bt['net_compound_return_pct']:+.1f}%"
            )
            ax.plot(ts, val, label=label, color=col, linewidth=1.8)
        ax.axhline(10_000.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"BTC/USDT Perp {tf} — 5bp/side Fee + 2bp Slip (0.12% RT)")
        ax.set_xlabel("Test Period Date (UTC)")
        ax.set_ylabel("Account Equity (USDT)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8.5)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    eq_path = os.path.join(output_dir, "equity_curves_5m_vs_15m.png")
    plt.savefig(eq_path, dpi=160)
    plt.close(fig)
    print(f"Saved equity curves plot to: {eq_path}")

    # 1b. Zero-Fee (0bp vs 1bp Slippage) Equity Curves across 5m, 15m, 1h
    fig, axes = plt.subplots(2, 3, figsize=(21, 10), sharey=False)
    for row_idx, (r_key, r_lbl) in enumerate([
        ("0_fee_0bp_slip", "0 Fee + 0bp Slippage"),
        ("0_fee_1bp_rt_slip", "0 Fee + 1bp Slippage"),
    ]):
        for col_idx, tf in enumerate(TIMEFRAMES):
            ax = axes[row_idx, col_idx]
            tf_res = results[tf]
            for th, col in colors.items():
                bt = tf_res["regime_backtests"][r_key][th]
                ts = pd.to_datetime(bt["equity_ts"])
                val = bt["equity_val"]
                label = (
                    f"P >= {th:.2f} | n={bt['total_trades']} | "
                    f"WR={bt['tp_win_rate']:.1f}% | Net={bt['net_compound_return_pct']:+.1f}% (SR={bt['sharpe_ratio']:.2f})"
                )
                ax.plot(ts, val, label=label, color=col, linewidth=1.8)
            ax.axhline(10_000.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(f"BTC/USDT Perp {tf} — {r_lbl} (2% Risk, 3x/1.5x ATR)")
            ax.set_xlabel("Test Period Date (UTC)")
            ax.set_ylabel("Account Equity (USDT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8.0)
            ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    eq0_path = os.path.join(output_dir, "equity_curves_zero_fee_0bp_1bp.png")
    plt.savefig(eq0_path, dpi=160)
    plt.close(fig)
    print(f"Saved zero-fee equity curves plot to: {eq0_path}")

    # 2. Calibration & Probability Distribution Plot (5m vs 15m vs 1h)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for col_idx, tf in enumerate(TIMEFRAMES):
        tf_res = results[tf]
        rel = tf_res["calibration"]["reliability_table"]

        valid_bins = [r for r in rel if r["count"] > 0 and not np.isnan(r["mean_pred_prob"])]
        pred_p = [r["mean_pred_prob"] for r in valid_bins]
        real_p = [r["realized_win_rate"] for r in valid_bins]

        ax_cal = axes[0, col_idx]
        ax_cal.plot([0.15, 0.55], [0.15, 0.55], "k--", alpha=0.6, label="Perfect Calibration (y=x)")
        ax_cal.axhline(0.34, color="red", linestyle=":", alpha=0.7, label="Compound Break-Even (34.0%)")
        ax_cal.plot(pred_p, real_p, "o-", color="#2ca02c", linewidth=2.0, markersize=7, label="Calibrated Model")
        for x_v, y_v, b_lbl in zip(pred_p, real_p, valid_bins):
            ax_cal.annotate(
                f"{b_lbl['bin']} ({b_lbl['count']})",
                (x_v, y_v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax_cal.set_title(
            f"{tf} Reliability | LogLoss={tf_res['calibration']['test_logloss']:.4f} "
            f"(Naive={tf_res['calibration']['naive_logloss']:.4f})"
        )
        ax_cal.set_xlabel("Mean Predicted Win Probability")
        ax_cal.set_ylabel("Realized Win Rate (Test Set)")
        ax_cal.grid(True, alpha=0.3)
        ax_cal.legend(loc="upper left", fontsize=8.5)

        ax_hist = axes[1, col_idx]
        test_probs = tf_res["test_probs"]
        ax_hist.hist(test_probs[:, 0], bins=40, alpha=0.6, label="P(Long Win)", color="#1f77b4", density=True)
        ax_hist.hist(test_probs[:, 1], bins=40, alpha=0.6, label="P(Short Win)", color="#d62728", density=True)
        ax_hist.axvline(0.34, color="black", linestyle=":", linewidth=1.2, label="Break-Even (0.34)")
        ax_hist.axvline(0.38, color="green", linestyle="--", linewidth=1.5, label="Primary Entry (0.38)")
        ax_hist.set_title(f"{tf} Test Set Calibrated Probability Distribution")
        ax_hist.set_xlabel("Calibrated Win Probability")
        ax_hist.set_ylabel("Density")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    cal_path = os.path.join(output_dir, "calibration_diagnostics_5m_vs_15m.png")
    plt.savefig(cal_path, dpi=160)
    plt.close(fig)
    print(f"Saved calibration diagnostics plot to: {cal_path}")


def print_full_report(results: dict) -> None:
    print("\n" + "=" * 100)
    print("BTC/USDT PERPETUAL FUTURES — TRIPLE BARRIER (1.5x ATR SL / 3.0x ATR TP) DIAGNOSTIC REPORT (5m / 15m / 1h)")
    print("=" * 100)

    for tf in TIMEFRAMES:
        r = results[tf]
        cal = r["calibration"]
        print(f"\n--- TIMEFRAME: {tf} ---")
        print(
            f"Purged Split Sizes (Embargo = {r['embargo_bars']} bars):\n"
            f"  Train : {r['n_train']:,} bars ({r['train_start']} -> {r['train_end']})\n"
            f"  Val   : {r['n_val']:,} bars ({r['val_start']} -> {r['val_end']})\n"
            f"  Test  : {r['n_test']:,} bars ({r['test_start']} -> {r['test_end']})"
        )
        print(
            f"Base Win Rate (Test): Long={cal['test_base_rate_long']*100:.2f}%, Short={cal['test_base_rate_short']*100:.2f}% | "
            f"Mean Test ATR%: {r['mean_test_atr_pct']*100:.3f}%"
        )
        print(
            f"Probability Calibration (Test Set):\n"
            f"  Model Log-Loss : {cal['test_logloss']:.5f} (Long={cal['test_logloss_long']:.5f}, Short={cal['test_logloss_short']:.5f})\n"
            f"  Naive Log-Loss : {cal['naive_logloss']:.5f} (Delta = {cal['logloss_improvement']:+.5f})\n"
            f"  Model Brier    : {cal['test_brier']:.5f} vs. Naive Brier: {cal['naive_brier']:.5f} (Delta = {cal['brier_improvement']:+.5f})"
        )
        print("\nBinned Reliability Table (Test Set Combined Long & Short):")
        print(
            f"  {'Bin':<10} | {'Count':>8} | {'Mean Pred':>10} | {'Realized WR':>12} | {'Calib Gap':>10} | {'Long WR (n)':>18} | {'Short WR (n)':>18}"
        )
        print("  " + "-" * 96)
        for row in cal["reliability_table"]:
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

        print("\nCompound Equity Backtest across Fee & Slippage Regimes (2% Risk/Trade, Max 1 Concurrent Position):")
        regime_titles = [
            ("0_fee_0bp_slip", "0 Fee + 0bp Slippage (0.00% drag)"),
            ("0_fee_1bp_rt_slip", "0 Fee + 1bp Round-Trip Slippage (0.01% drag)"),
            ("0_fee_1bp_side_slip", "0 Fee + 1bp/Side Slippage (2bp RT = 0.02% drag)"),
            ("5bp_fee_2bp_slip", "5bp/Side Taker Fee + 2bp Slippage (0.12% drag)"),
        ]
        for r_key, r_title in regime_titles:
            print(f"\n  [{tf}] Regime: {r_title}")
            print(
                f"  {'Threshold':<10} | {'Trades (L/S)':>15} | {'TP WinRate':>11} | {'Net Return':>12} | {'Final Equity':>13} | {'Profit Factor':>13} | {'Sharpe':>8} | {'Max DD':>9} | {'Friction':>10}"
            )
            print("  " + "-" * 116)
            for th in [0.36, 0.38, 0.40]:
                bt = r["regime_backtests"][r_key][th]
                tr_str = f"{bt['total_trades']} ({bt['long_trades']}/{bt['short_trades']})"
                print(
                    f"  P >= {th:<5.2f} | {tr_str:>15} | {bt['tp_win_rate']:>10.2f}% | "
                    f"{bt['net_compound_return_pct']:>+11.2f}% | ${bt['final_equity']:>12,.2f} | {bt['profit_factor']:>13.2f} | "
                    f"{bt['sharpe_ratio']:>8.2f} | {bt['max_drawdown_pct']:>8.2f}% | ${bt['total_fees_paid']:>9.2f}"
                )


def main():
    os.makedirs("outputs", exist_ok=True)
    results = {}

    df_5m = fetch_btc_perp_ohlcv(timeframe="5m", days=730, cache_dir="data")
    df_15m = fetch_btc_perp_ohlcv(timeframe="15m", days=730, cache_dir="data")
    df_1h = aggregate_15m_to_1h(df_15m, cache_dir="data")

    tf_dfs = {
        "5m": df_5m,
        "15m": df_15m,
        "1h": df_1h,
    }

    for tf in TIMEFRAMES:
        df = tf_dfs[tf]

        data_dict = prepare_dataset_and_purged_splits(
            df,
            lookback_window=64,
            max_hold=48,
            tp_mult=3.0,
            sl_mult=1.5,
        )

        # Reuse cached calibrated test probabilities for 5m/15m if already computed, or train fresh
        cached_probs_path = f"outputs/test_probs_{tf}.npy"
        if tf in ("5m", "15m") and os.path.exists(cached_probs_path):
            test_probs = np.load(cached_probs_path)
            if len(test_probs) == len(data_dict["X_test"]):
                print(f"[{tf}] Loaded cached calibrated test probabilities from {cached_probs_path}")
                train_out = {
                    "best_epoch": 12 if tf == "5m" else 2,
                    "calibration_info": {},
                    "test_probs": test_probs,
                }
            else:
                train_out = train_and_calibrate_model(
                    data_dict, timeframe=tf, epochs=30, batch_size=512, lr=5e-4, weight_decay=1e-2, patience=7, seed=42
                )
                np.save(cached_probs_path, train_out["test_probs"])
        else:
            train_out = train_and_calibrate_model(
                data_dict,
                timeframe=tf,
                epochs=30,
                batch_size=256 if tf == "1h" else 512,
                lr=5e-4,
                weight_decay=1e-2,
                patience=7,
                seed=42,
            )
            np.save(cached_probs_path, train_out["test_probs"])

        cal_diag = compute_calibration_diagnostics(
            y_train=data_dict["y_train"],
            y_test=data_dict["y_test"],
            test_probs=train_out["test_probs"],
        )

        friction_regimes = {
            "0_fee_0bp_slip": (0.0, 0.0),
            "0_fee_1bp_rt_slip": (0.0, 0.0001),
            "0_fee_1bp_side_slip": (0.0, 0.0002),
            "5bp_fee_2bp_slip": (0.0005, 0.0002),
        }
        backtests = {}
        regime_backtests = {r_name: {} for r_name in friction_regimes}
        for r_name, (fee_side, slip_rt) in friction_regimes.items():
            for th in [0.36, 0.38, 0.40]:
                bt_res = run_futures_backtest(
                    data_dict=data_dict,
                    test_probs=train_out["test_probs"],
                    threshold=th,
                    initial_equity=10_000.0,
                    risk_per_trade=0.02,
                    sl_mult=1.5,
                    tp_mult=3.0,
                    taker_fee_per_side=fee_side,
                    slippage_round_trip=slip_rt,
                    timeframe=tf,
                )
                regime_backtests[r_name][th] = bt_res
                if r_name == "5bp_fee_2bp_slip":
                    backtests[th] = bt_res

        train_bars = data_dict["train_bars"]
        val_bars = data_dict["val_bars"]
        test_bars = data_dict["test_bars"]
        ts = data_dict["timestamps"]
        results[tf] = {
            "n_train": int(len(data_dict["X_train"])),
            "n_val": int(len(data_dict["X_val"])),
            "n_test": int(len(data_dict["X_test"])),
            "embargo_bars": int(data_dict["embargo_bars"]),
            "train_start": str(pd.Timestamp(ts[train_bars[0]])),
            "train_end": str(pd.Timestamp(ts[train_bars[-1]])),
            "val_start": str(pd.Timestamp(ts[val_bars[0]])),
            "val_end": str(pd.Timestamp(ts[val_bars[-1]])),
            "test_start": str(pd.Timestamp(ts[test_bars[0]])),
            "test_end": str(pd.Timestamp(ts[test_bars[-1]])),
            "mean_test_atr_pct": float(np.mean(data_dict["atr_pct"][test_bars])),
            "best_epoch": train_out["best_epoch"],
            "calibration_fit": train_out["calibration_info"],
            "calibration": cal_diag,
            "backtests": backtests,
            "regime_backtests": regime_backtests,
            "test_probs": train_out["test_probs"],
        }

    print_full_report(results)
    plot_diagnostics(results, output_dir="outputs")

    json_summary = {}
    for tf in TIMEFRAMES:
        r = results[tf]
        json_summary[tf] = {
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "n_test": r["n_test"],
            "embargo_bars": r["embargo_bars"],
            "train_start": r["train_start"],
            "train_end": r["train_end"],
            "val_start": r["val_start"],
            "val_end": r["val_end"],
            "test_start": r["test_start"],
            "test_end": r["test_end"],
            "mean_test_atr_pct": r["mean_test_atr_pct"],
            "best_epoch": r["best_epoch"],
            "calibration_fit": r["calibration_fit"],
            "calibration": r["calibration"],
            "regime_backtests": {
                r_key: {
                    str(th): {
                        k: v
                        for k, v in bt.items()
                        if k not in ("trades", "equity_ts", "equity_val")
                    }
                    for th, bt in r_bts.items()
                }
                for r_key, r_bts in r["regime_backtests"].items()
            },
        }
    with open("outputs/results_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print("Saved JSON metrics summary to: outputs/results_summary.json")


if __name__ == "__main__":
    main()
