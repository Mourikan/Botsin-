"""Agente 1 - Descarga OHLCV historico desde la API REST publica de Binance (sin API keys)."""
import argparse
import os
import time

import pandas as pd
import requests

HOSTS = (
    "https://api.binance.com",          # principal
    "https://data-api.binance.vision",  # espejo publico de market data (fallback si hay geo-bloqueo)
)
PATH = "/api/v3/klines"
MAX_CANDLES = 5000
PAGE = 1000
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


def _get(params):
    last = None
    for host in HOSTS:
        for attempt in range(3):
            try:
                r = requests.get(host + PATH, params=params, timeout=15)
                if r.status_code == 200:
                    return r.json()
                last = f"{host} HTTP {r.status_code}: {r.text[:120]}"
                if r.status_code in (403, 418, 451):  # bloqueado: probar siguiente host
                    break
            except requests.RequestException as e:
                last = f"{host} {e!r}"
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Binance klines no disponible: {last}")


def fetch_klines(symbol="BTCUSDT", interval="5m", limit=1500):
    """Devuelve las ultimas `limit` velas CERRADAS (paginando hacia atras)."""
    limit = min(int(limit), MAX_CANDLES)
    rows, end = [], None
    while len(rows) < limit + 1:
        params = {"symbol": symbol, "interval": interval, "limit": PAGE}
        if end is not None:
            params["endTime"] = end
        batch = _get(params)
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < PAGE:
            break
    df = pd.DataFrame([r[:7] for r in rows], columns=COLS)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df["close_time"] < int(time.time() * 1000)]  # descarta vela aun abierta (anti look-ahead)
    df = df.tail(limit).reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    return df.drop(columns="close_time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", choices=["BTCUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="5m", choices=["1m", "5m"])
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--out", default="data/sample_ohlcv.csv")
    a = ap.parse_args()
    df = fetch_klines(a.symbol, a.interval, a.limit)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    t = pd.to_datetime(df["open_time"], unit="ms")
    print(f"{a.symbol} {a.interval}: {len(df)} velas [{t.iloc[0]} -> {t.iloc[-1]}] -> {a.out}")


if __name__ == "__main__":
    main()
