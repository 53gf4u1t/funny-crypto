# BTC/USDT & ETH/USDT 15m Perpetual Futures — Triple-Barrier Probabilistic Classifier

End-to-end PyTorch probabilistic sequence classifier and Numba-accelerated walk-forward backtester for **BTC/USDT** and **ETH/USDT** Perpetual Futures on the **15m** timeframe.

- **Final Strategy Specification & Full Research Report**: [FINAL_STRATEGY.md](./FINAL_STRATEGY.md)

## Quick Start

```bash
# 1. Fixed Purged Split (70/15/15) across 5m, 15m, and 1h
/opt/homebrew/bin/python3 run_pipeline.py

# 2. 23-Month (1.91-Year) Continual Experience-Replay Walk-Forward on 15m
/opt/homebrew/bin/python3 run_walkforward_15m.py

# 3. Barrier Geometry Ablation (2.0x SL / 3.0x TP vs 2.0x SL / 4.0x TP across BTC, ETH, SOL)
/opt/homebrew/bin/python3 run_walkforward_candidates.py

# 4. Confidence-Scaled Position Sizing (Fixed 2% vs 1%->2%, 0.5%->2%, 0.25%->2%)
/opt/homebrew/bin/python3 run_confidence_sizing.py
```

## Repository Structure

- `src/data.py` — Parallel `ccxt` perpetual futures OHLCV downloader with local Parquet caching and `15m -> 1h` aggregation.
- `src/features_labels.py` — Strictly causal `ATR(14)` / `ATR(100)`, 21 stationary ATR-normalized features, Numba `@njit` Triple-Barrier first-passage labeling with intra-bar loss conservatism, and chronological purged split (`embargo = max_hold + lookback_window = 112` bars).
- `src/model.py` — Regularized 1D-ResNet + BiGRU + temporal attention classifier (`LayerNorm`, `Dropout(0.3)`, `AdamW(weight_decay=1e-2)`), on-the-fly vectorized adjacent candle swaps (protecting `[t-2, t-1, t]`) + ATR-scaled Gaussian jitter `N(0, 0.04)`, and Centered Temperature Scaling (`PlattTemperatureCalibrator`).
- `src/backtest.py` — Non-overlapping compound futures simulator supporting both Fixed Risk and linear Confidence-Scaled Risk (`0.50% -> 2.00%` over `[threshold, threshold + 0.12]`) with exact notional fee and slippage deduction.
- `FINAL_STRATEGY.md` — Complete empirical summary and production specification for the finalized **BTC + ETH 15m (`1.5x SL / 3.0x TP`, `P >= 0.38`, `0.50% -> 2.00%` Confidence-Scaled Risk)** strategy.
