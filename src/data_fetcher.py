"""Agente 1 - OHLCV + taker volume desde la API REST publica de Binance (sin API keys).

Por defecto: Futuros Perpetuos USDT-M (fapi/v1/klines). `--market spot` queda como alternativa explicita.
"""
import argparse
import io
import json
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

MARKETS = {  # nombre -> (url, max velas por request)
    "futures": ("https://fapi.binance.com/fapi/v1/klines", 1500),
    "spot": ("https://api.binance.com/api/v3/klines", 1000),
}
ARCHIVE = "https://data.binance.vision/data/futures/um/daily/klines/{s}/{i}/{s}-{i}-{d}.zip"
MAX_CANDLES = 5000
NUM = ["open", "high", "low", "close", "volume", "taker_buy_volume"]


def _get(url, params):
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:160]}"
            if r.status_code in (403, 418, 451):  # bloqueo de red/geo: reintentar no ayuda
                break
        except requests.RequestException as e:
            last = repr(e)[:300]
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Binance klines no disponible ({url}): {last}")


def fetch_archive(symbol, interval, limit, end=None):
    """Futuros USDT-M desde el archivo publico oficial (zips diarios, publicados con ~1 dia de retraso)."""
    day = date.fromisoformat(end) + timedelta(days=1) if end else datetime.now(timezone.utc).date()
    rows, misses = [], 0   # `end` = ultimo dia incluido (YYYY-MM-DD) para fijar ventanas historicas
    while len(rows) < limit and misses < 4:
        day -= timedelta(days=1)
        r = requests.get(ARCHIVE.format(s=symbol, i=interval, d=day.isoformat()), timeout=30)
        if r.status_code != 200:
            if r.status_code == 403:
                r.raise_for_status()
            misses += 1
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        d = d[pd.to_numeric(d[0], errors="coerce").notna()]      # algunos zips traen cabecera
        rows = d.values.tolist() + rows
    if not rows:
        raise RuntimeError("archivo data.binance.vision sin datos")
    return rows


def fetch_klines(symbol="SOLUSDT", interval="5m", limit=5000, market="futures", end=None):
    """Ultimas `limit` velas CERRADAS, paginando hacia atras."""
    limit = min(int(limit), MAX_CANDLES)
    if market == "futures-archive":
        return _frame(fetch_archive(symbol, interval, limit, end), limit)
    url, page = MARKETS[market]
    rows, end = [], None
    while len(rows) < limit + 1:
        params = {"symbol": symbol, "interval": interval, "limit": page}
        if end is not None:
            params["endTime"] = end
        batch = _get(url, params)
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < page:
            break
        time.sleep(0.25)
    return _frame(rows, limit)


def _frame(rows, limit):
    df = pd.DataFrame([[r[0], r[1], r[2], r[3], r[4], r[5], r[9], r[6]] for r in rows],
                      columns=["open_time"] + NUM + ["close_time"])
    df = df.apply(pd.to_numeric)
    for k in ("open_time", "close_time"):
        df[k] = df[k].astype("int64")
        if df[k].iloc[-1] > 10**14:                              # microsegundos -> ms
            df[k] //= 1000
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df["close_time"] < int(time.time() * 1000)]  # descarta la vela aun abierta (anti look-ahead)
    df = df.tail(limit).reset_index(drop=True).drop(columns="close_time")
    df[NUM] = df[NUM].astype("float64")
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    df["volume_delta"] = df["taker_buy_volume"] - df["taker_sell_volume"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT", choices=["BTCUSDT", "SOLUSDT"])
    ap.add_argument("--interval", default="5m", choices=["1m", "5m"])
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--market", default="futures", choices=list(MARKETS) + ["futures-archive"])
    ap.add_argument("--out", default="data/sample_ohlcv.csv")
    ap.add_argument("--end", default=None, help="ultimo dia incluido YYYY-MM-DD (solo archivo historico)")
    a = ap.parse_args()
    if a.end:
        a.market = "futures-archive"
    try:
        df = fetch_klines(a.symbol, a.interval, a.limit, a.market, a.end)
    except RuntimeError as e:
        if a.market != "futures" or "451" not in str(e):
            raise
        print("AVISO: fapi geobloqueado (HTTP 451) -> usando archivo oficial data.binance.vision (mismas velas de Futuros)")
        a.market = "futures-archive"
        df = fetch_klines(a.symbol, a.interval, a.limit, a.market, a.end)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    t = pd.to_datetime(df["open_time"], unit="ms")
    meta = {"symbol": a.symbol, "interval": a.interval, "market": a.market, "rows": len(df),
            "first": str(t.iloc[0]), "last": str(t.iloc[-1])}
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(f"{a.market} {a.symbol} {a.interval}: {len(df)} velas [{meta['first']} -> {meta['last']}] -> {a.out}")


if __name__ == "__main__":
    main()
