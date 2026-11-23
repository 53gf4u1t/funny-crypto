from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import ccxt
import numpy as np
import pandas as pd


def _fetch_chunk(ex_id: str, symbol: str, timeframe: str, start_ms: int, end_ms: int, tf_ms: int, limit: int):
    ex_cls = getattr(ccxt, ex_id)
    ex = ex_cls({"enableRateLimit": True, "timeout": 20000})
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        for attempt in range(3):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        if not batch:
            break
        for r in batch:
            if r[0] < end_ms:
                rows.append(r)
        last_ts = batch[-1][0]
        if last_ts <= cursor or last_ts >= end_ms - tf_ms:
            break
        cursor = last_ts + tf_ms
    return rows


def fetch_perp_ohlcv(
    base_asset: str = "BTC",
    timeframe: str = "5m",
    days: int = 730,
    cache_dir: str = "data",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical <BASE>/USDT perpetual futures OHLCV data (2 years = 730 days)
    via ccxt (Binance USD-M / Bybit / OKX) with local parquet/CSV caching.
    """
    base_asset = base_asset.upper()
    os.makedirs(cache_dir, exist_ok=True)
    parquet_path = os.path.join(cache_dir, f"{base_asset}USDT_perp_{timeframe}.parquet")
    csv_path = os.path.join(cache_dir, f"{base_asset}USDT_perp_{timeframe}.csv")

    if not force_download:
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            print(f"[{base_asset} {timeframe}] Loaded {len(df):,} bars from cache: {parquet_path}")
            return df
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            print(f"[{base_asset} {timeframe}] Loaded {len(df):,} bars from cache: {csv_path}")
            return df

    tf_ms_map = {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "1h": 60 * 60_000,
    }
    tf_ms = tf_ms_map[timeframe]
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - days * 86_400_000

    ccxt_symbol = f"{base_asset}/USDT:USDT"
    exchanges_to_try = [
        ("binanceusdm", ccxt_symbol, 1500),
        ("bybit", ccxt_symbol, 1000),
        ("okx", ccxt_symbol, 300),
    ]

    all_rows = []
    used_exchange = None
    num_workers = 8
    total_span = now_ms - since_ms
    chunk_span = total_span // num_workers
    chunks = [
        (since_ms + i * chunk_span, since_ms + (i + 1) * chunk_span if i < num_workers - 1 else now_ms)
        for i in range(num_workers)
    ]

    for ex_id, symbol, limit in exchanges_to_try:
        try:
            print(f"[{base_asset} {timeframe}] Fetching {days} days of {symbol} perpetual futures via ccxt ({ex_id})...")
            t0 = time.time()
            all_rows = []
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [
                    pool.submit(_fetch_chunk, ex_id, symbol, timeframe, c_start, c_end, tf_ms, limit)
                    for c_start, c_end in chunks
                ]
                for fut in as_completed(futures):
                    all_rows.extend(fut.result())

            if len(all_rows) > 1000:
                used_exchange = ex_id
                print(
                    f"[{base_asset} {timeframe}] Downloaded {len(all_rows):,} raw bars from {ex_id} in {time.time() - t0:.1f}s"
                )
                break
        except Exception as e:
            print(f"[{base_asset} {timeframe}] Warning: {ex_id} failed ({e}), trying next exchange...")
            continue

    if not all_rows:
        raise RuntimeError(f"Failed to download {base_asset}/USDT perpetual data for {timeframe} from all exchanges.")

    df = pd.DataFrame(
        all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[{base_asset} {timeframe}] Cached {len(df):,} bars ({used_exchange}) -> {parquet_path}")
    except Exception:
        df.to_csv(csv_path, index=False)
        print(f"[{base_asset} {timeframe}] Cached {len(df):,} bars ({used_exchange}) -> {csv_path}")

    return df


def fetch_btc_perp_ohlcv(
    timeframe: str = "5m",
    days: int = 730,
    cache_dir: str = "data",
    force_download: bool = False,
) -> pd.DataFrame:
    return fetch_perp_ohlcv("BTC", timeframe=timeframe, days=days, cache_dir=cache_dir, force_download=force_download)


def aggregate_15m_to_1h(df_15m: pd.DataFrame, cache_dir: str = "data") -> pd.DataFrame:
    """
    Aggregate 15m BTC/USDT perpetual futures OHLCV candles to 1h candles.
    """
    os.makedirs(cache_dir, exist_ok=True)
    parquet_path = os.path.join(cache_dir, "BTCUSDT_perp_1h_from_15m.parquet")

    df = df_15m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    ohlcv_1h = (
        df.resample("1h", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    ohlcv_1h["timestamp_ms"] = (ohlcv_1h["timestamp"].astype("int64") // 10**6).astype(np.int64)
    ohlcv_1h = ohlcv_1h[["timestamp_ms", "open", "high", "low", "close", "volume", "timestamp"]]
    try:
        ohlcv_1h.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    print(f"[1h] Aggregated {len(df_15m):,} 15m bars -> {len(ohlcv_1h):,} 1h bars")
    return ohlcv_1h

