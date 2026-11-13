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
PARAMS = dict(bins=60, va_pct=0.70, pivot_k=12, min_len=36, max_len=288, atr_n=14, rsi_n=7,
              rvol_n=20, rvol_min=1.2, wick_min=0.35, rsi_lo=35.0, rsi_hi=65.0,
              anomaly_rvol=1.7, anomaly_side=1.5, vwap_n=288)

# feat -> build_features | sig -> generate_signals | sim -> futures_simulator.simulate
# parent = modelo sin ese filtro: el ratchet exige superarlo en Net PnL y Profit Factor.
_WIDE = dict(pivot_k=24, min_len=72, max_len=500)
_WIDE_SIM = dict(min_width=0.0045, max_hold=None)
MODELS = {
    "A": dict(name="Pure FRVP + Delta"),
    "B": dict(name="FRVP + RSI7 MicroDiv", sig=dict(rsi="div"), parent="A"),
    "C": dict(name="FRVP + RSI7 35/65", sig=dict(rsi="extreme"), parent="A"),
    "A1": dict(name="Anclaje Amplio 24v", feat=_WIDE, sim=_WIDE_SIM),
    "A2": dict(name="Anomalia de Volumen", sig=dict(anomaly=True), parent="A"),
    # Fase 4: A2 + amplitud 0.45% + sin timeout + regimen VWAP 24h (solo longs sobre VWAP, solo shorts bajo VWAP)
    "R1": dict(name="Regimen VWAP 24h", sig=dict(anomaly=True, regime="vwap"), sim=_WIDE_SIM),
    # Diagnostico Fase 3: A1 descompuesto en sus 3 variables (no elegibles como modelo por defecto)
    "V1": dict(name="Solo pivotes 24v", feat=_WIDE),
    "V2": dict(name="Solo amplitud 0.45%", sim=dict(min_width=0.0045)),
    "V3": dict(name="Solo sin timeout", sim=dict(max_hold=None)),
    "A3": dict(name="Sniper Institucional", feat=_WIDE, sim=_WIDE_SIM, sig=dict(anomaly=True), parent="A1"),
}


def spec(model):
    m = MODELS[model]
    return dict(name=m["name"], feat=m.get("feat", {}), sig=m.get("sig", {}), sim=m.get("sim", {}),
                parent=m.get("parent"))


def active_model():
    """Modelo por defecto = ganador guardado por el ratchet (A si aun no hay benchmark)."""
    try:
        with open(BENCH) as f:
            w = json.load(f)["winner"]
        return w if w in MODELS else "A"
    except (OSError, KeyError, ValueError):
        return "A"


def _wilder(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _prev_sma(x, n):
    """SMA de las n velas previas (excluye la actual)."""
    return pd.Series(x).rolling(n).mean().shift(1).to_numpy()


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
    dlt = df["volume_delta"].to_numpy(np.float64)
    buy, sell = (v + dlt) / 2.0, (v - dlt) / 2.0         # taker buy / taker sell
    f["rvol"] = v / _prev_sma(v, p["rvol_n"])
    f["buy_rvol"] = buy / _prev_sma(buy, p["rvol_n"])
    f["sell_rvol"] = sell / _prev_sma(sell, p["rvol_n"])
    tpv = pd.Series((h + l + c) / 3.0 * v).rolling(p["vwap_n"]).sum()   # VWAP rolling, incluye la vela i (cerrada)
    f["vwap"] = (tpv / pd.Series(v).rolling(p["vwap_n"]).sum()).to_numpy()
    rng = np.maximum(h - l, 1e-12)
    f["wick_lo"] = (np.minimum(o, c) - l) / rng
    f["wick_hi"] = (h - np.maximum(o, c)) / rng
    return f


def generate_signals(f, model=None, **kw):
    """+1 long / -1 short al cierre de la vela i.

    Base: mecha toca VAL/VAH, cierra dentro del area de valor y volume_delta a favor, mas
      - normal:  RVOL >= 1.2 o mecha >= 35% del rango
      - anomaly: RVOL >= 1.7 o volumen taker del lado del trade > 1.5x su media de 20 velas
    Filtros RSI(7) opcionales: "div" (micro-divergencia t-1 -> t) o "extreme" (<35 / >65).
    """
    p = {**PARAMS, **kw}
    sg = spec(model or active_model())["sig"]
    h, l, c = f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()
    dlt, rsi, rvol = f["volume_delta"].to_numpy(), f["rsi"].to_numpy(), f["rvol"].to_numpy()
    if sg.get("anomaly"):
        big = rvol >= p["anomaly_rvol"]
        abs_l = big | (f["buy_rvol"].to_numpy() > p["anomaly_side"])
        abs_s = big | (f["sell_rvol"].to_numpy() > p["anomaly_side"])
    else:
        abs_l = (rvol >= p["rvol_min"]) | (f["wick_lo"].to_numpy() >= p["wick_min"])
        abs_s = (rvol >= p["rvol_min"]) | (f["wick_hi"].to_numpy() >= p["wick_min"])
    long = ((l <= f["val"]) & (c > f["val"]) & (c < f["poc"])).to_numpy() & (dlt > 0) & abs_l
    short = ((h >= f["vah"]) & (c < f["vah"]) & (c > f["poc"])).to_numpy() & (dlt < 0) & abs_s
    if sg.get("rsi") == "div":        # precio hace nuevo extremo, RSI no
        pr, pl_, ph_ = np.r_[np.nan, rsi[:-1]], np.r_[np.nan, l[:-1]], np.r_[np.nan, h[:-1]]
        long = long & (l <= pl_) & (rsi > pr)
        short = short & (h >= ph_) & (rsi < pr)
    elif sg.get("rsi") == "extreme":  # agotamiento simple
        long = long & (rsi < p["rsi_lo"])
        short = short & (rsi > p["rsi_hi"])
    if sg.get("regime") == "vwap":    # reversion solo a favor del flujo macro
        vw = f["vwap"].to_numpy()
        long = long & (c > vw)
        short = short & (c < vw)
    sig = np.zeros(len(f), np.int8)
    sig[long] = 1
    sig[short & ~long] = -1
    return sig


def selftest():
    """Causalidad: alterar velas > k no puede cambiar features ni senales en filas <= k."""
    df, k = synthetic(1500, 11), 1200
    d2 = df.copy()
    d2.loc[k + 1:, ["open", "high", "low", "close"]] *= 1.4
    d2.loc[k + 1:, ["volume", "volume_delta"]] *= 7
    cols = ["poc", "vah", "val", "swing_low", "swing_high", "atr", "rsi", "rvol", "buy_rvol", "sell_rvol", "vwap"]
    ok = True
    for m in MODELS:
        s = spec(m)
        a, b = build_features(df, **s["feat"]), build_features(d2, **s["feat"])
        ok &= np.allclose(a.loc[:k, cols], b.loc[:k, cols], equal_nan=True)
        ok &= np.array_equal(generate_signals(a, m)[:k + 1], generate_signals(b, m)[:k + 1])
        x = a.dropna(subset=["poc"])
        ok &= len(x) > 0 and bool(((x.swing_low <= x.val) & (x.val <= x.poc) & (x.poc <= x.vah)
                                   & (x.vah <= x.swing_high)).all())
    print("SELFTEST strategy", "OK" if ok else "FAIL")
    return bool(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print("modelo activo:", active_model(), "-", MODELS[active_model()]["name"])
