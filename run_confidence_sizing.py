import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import fetch_perp_ohlcv
from src.backtest import run_futures_backtest, run_confidence_scaled_backtest
from run_walkforward_15m import build_raw_features_and_labels, run_monthly_walkforward_15m


def get_or_run_wf_predictions(
    asset: str, sl_mult: float, tp_mult: float
) -> dict:
    df_15m = fetch_perp_ohlcv(base_asset=asset, timeframe="15m", days=730, cache_dir="data")
    base_data = build_raw_features_and_labels(
        df_15m, lookback_window=64, max_hold=48, tp_mult=tp_mult, sl_mult=sl_mult
    )

    if sl_mult == 1.5 and tp_mult == 3.0:
        cache_path = f"outputs/wf_15m_{asset}_pure_replay.npz"
    else:
        tag = f"{asset}_SL{sl_mult:.1f}_TP{tp_mult:.1f}".replace(".", "p")
        cache_path = f"outputs/wf_15m_{tag}_pure_replay.npz"

    if os.path.exists(cache_path):
        loaded = np.load(cache_path)
        if len(loaded["oos_bars"]) > 60000:
            print(f"[{asset} | {sl_mult:.1f}x/{tp_mult:.1f}x] Loaded cached 23-month OOS predictions from {cache_path}")
            return {
                "base_data": base_data,
                "oos_bars": loaded["oos_bars"],
                "oos_probs": loaded["oos_probs"],
                "oos_y": loaded["oos_y"],
            }

    wf = run_monthly_walkforward_15m(asset, base_data, n_months=24, seed=42)
    np.savez(
        cache_path,
        oos_bars=wf["oos_bars"],
        oos_probs=wf["oos_probs"],
        oos_naive=wf["oos_naive"],
        oos_y=wf["oos_y"],
    )
    return {
        "base_data": base_data,
        "oos_bars": wf["oos_bars"],
        "oos_probs": wf["oos_probs"],
        "oos_y": wf["oos_y"],
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    eval_configs = [
        ("ETH", "1.5x SL / 3.0x TP (2:1 R:R)", 1.5, 3.0, [0.36, 0.38, 0.40], 0.12),
        ("BTC", "1.5x SL / 3.0x TP (2:1 R:R)", 1.5, 3.0, [0.36, 0.38, 0.40], 0.12),
        ("SOL", "1.5x SL / 3.0x TP (2:1 R:R)", 1.5, 3.0, [0.36, 0.38, 0.40], 0.12),
    ]
    regimes = [
        ("0_fee_0bp_slip", "0 Fee + 0bp Slippage (0.00%)", 0.0, 0.0),
        ("0_fee_1bp_rt_slip", "0 Fee + 1bp RT Slippage (0.01%)", 0.0, 0.0001),
        ("0_fee_1bp_side_slip", "0 Fee + 1bp/Side Slippage (2bp RT = 0.02%)", 0.0, 0.0002),
    ]

    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    plot_cols = {"BTC": 0, "ETH": 1, "SOL": 2}

    for asset, setup_lbl, sl_m, tp_m, th_list, p_span in eval_configs:
        wf_out = get_or_run_wf_predictions(asset, sl_m, tp_m)
        base_data = wf_out["base_data"]
        oos_bars = wf_out["oos_bars"]
        oos_probs = wf_out["oos_probs"]
        oos_y = wf_out["oos_y"]

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

        ts = base_data["timestamps"]
        oos_years = (
            pd.Timestamp(ts[oos_bars[-1]]) - pd.Timestamp(ts[oos_bars[0]])
        ).total_seconds() / (365.25 * 86400.0)

        print("\n" + "=" * 134)
        print(
            f"{asset}/USDT 15m | {setup_lbl} — FIXED 2.0% vs. CONFIDENCE-SCALED (1.0%->2.0%, 0.50%->2.0%, 0.25%->2.0%)"
        )
        print("=" * 134)

        setup_key = f"{asset}_SL{sl_m:.1f}_TP{tp_m:.1f}"
        summary[setup_key] = {}

        for r_key, r_title, fee_side, slip_rt in regimes:
            summary[setup_key][r_key] = {}
            print(f"\n  [{asset} | {setup_lbl}] Friction: {r_title}")
            print(
                f"  {'Sizing Mode':<26} | {'Thresh':<6} | {'AvgRisk':>7} | {'Trades (L/S)':>16} | {'TP WR':>7} | {'Total Return':>13} | {'Annual CAGR':>12} | {'Final Equity':>13} | {'PF':>5} | {'Sharpe':>6} | {'Max DD':>7}"
            )
            print("  " + "-" * 140)

            for th in th_list:
                bt_fixed = run_futures_backtest(
                    data_dict=wf_data_dict,
                    test_probs=oos_probs,
                    threshold=th,
                    initial_equity=10_000.0,
                    risk_per_trade=0.02,
                    sl_mult=sl_m,
                    tp_mult=tp_m,
                    taker_fee_per_side=fee_side,
                    slippage_round_trip=slip_rt,
                    timeframe="15m",
                )
                bt_100 = run_confidence_scaled_backtest(
                    data_dict=wf_data_dict,
                    test_probs=oos_probs,
                    threshold=th,
                    prob_span=p_span,
                    min_risk=0.0100,
                    max_risk=0.0200,
                    initial_equity=10_000.0,
                    sl_mult=sl_m,
                    tp_mult=tp_m,
                    taker_fee_per_side=fee_side,
                    slippage_round_trip=slip_rt,
                )
                bt_050 = run_confidence_scaled_backtest(
                    data_dict=wf_data_dict,
                    test_probs=oos_probs,
                    threshold=th,
                    prob_span=p_span,
                    min_risk=0.0050,
                    max_risk=0.0200,
                    initial_equity=10_000.0,
                    sl_mult=sl_m,
                    tp_mult=tp_m,
                    taker_fee_per_side=fee_side,
                    slippage_round_trip=slip_rt,
                )
                bt_025 = run_confidence_scaled_backtest(
                    data_dict=wf_data_dict,
                    test_probs=oos_probs,
                    threshold=th,
                    prob_span=p_span,
                    min_risk=0.0025,
                    max_risk=0.0200,
                    initial_equity=10_000.0,
                    sl_mult=sl_m,
                    tp_mult=tp_m,
                    taker_fee_per_side=fee_side,
                    slippage_round_trip=slip_rt,
                )

                modes = [
                    ("Fixed 2.00% Risk", "fixed_2pct", bt_fixed, 2.00, "#d62728"),
                    (f"Conf 1.00%->2% [{th:.2f}-{th+p_span:.2f}]", "conf_1p0_to_2pct", bt_100, bt_100["avg_risk_pct"], "#1f77b4"),
                    (f"Conf 0.50%->2% [{th:.2f}-{th+p_span:.2f}]", "conf_0p5_to_2pct", bt_050, bt_050["avg_risk_pct"], "#2ca02c"),
                    (f"Conf 0.25%->2% [{th:.2f}-{th+p_span:.2f}]", "conf_0p25_to_2pct", bt_025, bt_025["avg_risk_pct"], "#9467bd"),
                ]

                summary[setup_key][r_key][str(th)] = {}
                for mode_lbl, m_key, bt, avg_r, _ in modes:
                    feq = bt["final_equity"]
                    cagr = ((feq / 10_000.0) ** (1.0 / oos_years) - 1.0) * 100.0 if feq > 0 else -100.0
                    bt["cagr_pct"] = cagr
                    tr_str = f"{bt['total_trades']} ({bt['long_trades']}/{bt['short_trades']})"
                    print(
                        f"  {mode_lbl:<26} | >={th:<4.2f} | {avg_r:>6.2f}% | {tr_str:>16} | "
                        f"{bt['tp_win_rate']:>6.2f}% | {bt['net_compound_return_pct']:>+12.2f}% | "
                        f"{cagr:>+11.2f}% | ${feq:>12,.2f} | {bt['profit_factor']:>5.2f} | "
                        f"{bt['sharpe_ratio']:>6.2f} | {bt['max_drawdown_pct']:>6.2f}%"
                    )
                    summary[setup_key][r_key][str(th)][m_key] = {
                        k: v for k, v in bt.items() if k not in ("trades", "equity_ts", "equity_val")
                    }

                if th == 0.38 and r_key in ("0_fee_0bp_slip", "0_fee_1bp_rt_slip"):
                    row_i = 0 if r_key == "0_fee_0bp_slip" else 1
                    col_i = plot_cols[asset]
                    ax = axes[row_i, col_i]
                    for mode_lbl, _, bt, avg_r, col in modes:
                        short_m = mode_lbl.split("[")[0].strip()
                        ax.plot(
                            pd.to_datetime(bt["equity_ts"]),
                            bt["equity_val"],
                            label=f"{short_m} (avg {avg_r:.2f}%) | CAGR={bt['cagr_pct']:+.1f}% | DD={bt['max_drawdown_pct']:.1f}% | SR={bt['sharpe_ratio']:.2f}",
                            color=col,
                            linewidth=1.8,
                        )
                    ax.axhline(10_000.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
                    r_short = "0bp Slippage" if r_key == "0_fee_0bp_slip" else "1bp RT Slippage"
                    ax.set_title(f"{asset}/USDT 15m — {setup_lbl} (P>=0.38, {r_short})")
                    ax.set_xlabel("Date (UTC)")
                    ax.set_ylabel("Account Equity (USDT)")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=7.5)
                    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path = "outputs/walkforward_15m_confidence_sizing.png"
    plt.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved confidence sizing comparison plot to: {plot_path}")

    with open("outputs/walkforward_15m_confidence_sizing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved confidence sizing JSON summary to: outputs/walkforward_15m_confidence_sizing_summary.json")


if __name__ == "__main__":
    main()
