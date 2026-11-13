"""Agente 2 - FRVP anclado al ultimo swing estructural + senales de absorcion en VAL/VAH.

Causalidad: en la vela i (cerrada) el perfil usa velas <= i-1 y solo pivotes confirmados <= i-1.
ATR/RSI/RVOL/mecha/delta usan la propia vela i, ya cerrada cuando se decide.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

try:
    from src.frvp_engine import profile_levels, synthetic
except ImportError:
    from frvp_engine import profile_levels, synthetic

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_benchmark.json")
MODELS = {"A": "Pure FRVP + Delta", "B": "FRVP + RSI7 MicroDiv", "C": "FRVP + RSI7 35/65"}
PARAMS = dict(bins=60, va_pct=0.70, pivot_k=12, min_len=36, max_len=288, atr_n=14, rsi_n=7,
              rvol_n=20, rvol_min=1.2, wick_min=0.35, rsi_lo=35.0, rsi_hi=65.0)


def active_model():
    """Modelo por defecto = ganador guardado por el ratchet (A si aun no hay benchmark)."""
    try:
        with open(BENCH) as f:
            return json.load(f)["winner"]
    except (OSError, KeyError, ValueError):
        return "A"


def _wilder(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _last_pivot(x, k, is_high):
    """Indice del ultimo pivote (k velas a cada lado) confirmado en <= i-1, por vela i. -1 si no hay."""
    n = len(x)
    w = swv(x, 2 * k + 1)
    ext = w.max(axis=1) if is_high else w.min(axis=1)
    j = np.flatnonzero(x[k:n - k] == ext) + k          # pivote en j, confirmado al cierre de j+k
    conf = np.full(n, -1, np.int64)
    conf[j + k] = j
    conf = np.maximum.accumulate(conf)
    out = np.full(n, -1, np.int64)
    out[1:] = conf[:-1]
    return out


def build_features(df, **kw):
    p = {**PARAMS, **kw}
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    n, K, mn, mx = len(df), p["pivot_k"], p["min_len"], p["max_len"]
    i = np.arange(n)
    ph, pl = _last_pivot(h, K, True), _last_pivot(l, K, False)
    ok = (ph >= 0) & (pl >= 0) & (i >= mn)
    start = np.clip(np.minimum(ph, pl), i - mx, i - mn)  # inicio del ultimo swing (pivote mas antiguo de los 2)
    rows = i[ok]
    f = df.copy()
    for k_ in ("poc", "vah", "val", "swing_low", "swing_high"):
        f[k_] = np.nan
    if len(rows):
        cols = rows[:, None] - mx + np.arange(mx)        # velas [i-mx, i-1]
        mask = cols >= start[rows][:, None]
        cols = np.clip(cols, 0, None)
        lv = profile_levels(h[cols], l[cols], v[cols], p["bins"], p["va_pct"], mask=mask)
        for k_, a in zip(("poc", "vah", "val", "swing_low", "swing_high"), lv):
            f.loc[rows, k_] = a
    pc = np.r_[c[0], c[:-1]]
    f["atr"] = _wilder(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), p["atr_n"])
    d = np.diff(c, prepend=c[0])
    up, dn = _wilder(np.clip(d, 0, None), p["rsi_n"]), _wilder(np.clip(-d, 0, None), p["rsi_n"])
    f["rsi"] = 100.0 - 100.0 / (1.0 + up / np.maximum(dn, 1e-12))
    f["rvol"] = v / pd.Series(v).rolling(p["rvol_n"]).mean().shift(1).to_numpy()
    rng = np.maximum(h - l, 1e-12)
    f["wick_lo"] = (np.minimum(o, c) - l) / rng
    f["wick_hi"] = (h - np.maximum(o, c)) / rng
    return f


def generate_signals(f, model=None, **kw):
    """+1 long / -1 short al cierre de la vela i.

    Base (A): mecha toca VAL/VAH y cierra dentro del area de valor, con (RVOL o mecha) y
    volume_delta a favor del rechazo. B y C anaden un filtro RSI(7) sobre A.
    """
    p = {**PARAMS, **kw}
    model = model or active_model()
    h, l, c = f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()
    dlt, rsi = f["volume_delta"].to_numpy(), f["rsi"].to_numpy()
    strong = f["rvol"].to_numpy() >= p["rvol_min"]
    long = ((l <= f["val"]) & (c > f["val"]) & (c < f["poc"]) & (dlt > 0)
            & (strong | (f["wick_lo"] >= p["wick_min"]))).to_numpy()
    short = ((h >= f["vah"]) & (c < f["vah"]) & (c > f["poc"]) & (dlt < 0)
             & (strong | (f["wick_hi"] >= p["wick_min"]))).to_numpy()
    if model == "B":      # micro-divergencia t-1 -> t: precio hace nuevo extremo, RSI no
        pr, pl_, ph_ = np.r_[np.nan, rsi[:-1]], np.r_[np.nan, l[:-1]], np.r_[np.nan, h[:-1]]
        long = long & (l <= pl_) & (rsi > pr)
        short = short & (h >= ph_) & (rsi < pr)
    elif model == "C":    # agotamiento simple
        long = long & (rsi < p["rsi_lo"])
        short = short & (rsi > p["rsi_hi"])
    elif model != "A":
        raise ValueError(f"modelo desconocido: {model}")
    sig = np.zeros(len(f), np.int8)
    sig[long] = 1
    sig[short & ~long] = -1
    return sig


def selftest():
    """Causalidad: alterar velas > k no puede cambiar features ni senales en filas <= k."""
    df, k = synthetic(900, 11), 700
    d2 = df.copy()
    d2.loc[k + 1:, ["open", "high", "low", "close"]] *= 1.4
    d2.loc[k + 1:, ["volume", "volume_delta"]] *= 7
    a, b = build_features(df), build_features(d2)
    cols = ["poc", "vah", "val", "swing_low", "swing_high", "atr", "rsi", "rvol"]
    ok = np.allclose(a.loc[:k, cols], b.loc[:k, cols], equal_nan=True)
    ok &= all(np.array_equal(generate_signals(a, m)[:k + 1], generate_signals(b, m)[:k + 1]) for m in MODELS)
    x = a.dropna(subset=["poc"])
    ok &= len(x) > 0 and bool(((x.swing_low <= x.val) & (x.val <= x.poc) & (x.poc <= x.vah)
                               & (x.vah <= x.swing_high)).all())
    print("SELFTEST strategy", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print("modelo activo:", active_model(), "-", MODELS[active_model()])
