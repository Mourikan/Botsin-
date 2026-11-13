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
    from src.regime_arbiter import arbitrate
except ImportError:
    from frvp_engine import profile_levels, synthetic
    from regime_arbiter import arbitrate

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_benchmark.json")
PARAMS = dict(bins=60, va_pct=0.70, pivot_k=12, min_len=36, max_len=288, atr_n=14, rsi_n=7,
              rvol_n=20, rvol_min=1.2, wick_min=0.35, rsi_lo=35.0, rsi_hi=65.0,
              anomaly_rvol=1.7, anomaly_side=1.5, vwap_n=288)

# feat -> build_features | sig -> generate_signals | sim -> futures_simulator.simulate
# parent = modelo sin ese filtro: el ratchet exige superarlo en Net PnL y Profit Factor.
_WIDE = dict(pivot_k=24, min_len=72, max_len=500)
_WIDE_SIM = dict(min_width=0.0045, max_hold=None)
# BASE PERMANENTE (Fase 4): pivotes 12v (PARAMS), amplitud >= 0.45%, sin timeout, TP1 en el POC.
# Los modelos anteriores (A..A3, V*, R0) quedan solo como registro historico reproducible.
BASE_SIM = dict(min_width=0.0045, max_hold=None, tp1_path=1.0, tp1_min_r=0.0)
MODELS = {
    "A": dict(name="Pure FRVP + Delta"),
    "B": dict(name="FRVP + RSI7 MicroDiv", sig=dict(rsi="div"), parent="A"),
    "C": dict(name="FRVP + RSI7 35/65", sig=dict(rsi="extreme"), parent="A"),
    "A1": dict(name="Anclaje Amplio 24v", feat=_WIDE, sim=_WIDE_SIM),
    "A2": dict(name="Anomalia de Volumen", sig=dict(anomaly=True), parent="A"),
    # Fase 4a (descartado): A2 + amplitud 0.45% + sin timeout + regimen VWAP 24h
    "R0": dict(name="Regimen VWAP 24h", sig=dict(anomaly=True, regime="vwap"), sim=_WIDE_SIM),
    # Diagnostico Fase 3: A1 descompuesto en sus 3 variables (no elegibles como modelo por defecto)
    "V1": dict(name="Solo pivotes 24v", feat=_WIDE),
    "V2": dict(name="Solo amplitud 0.45%", sim=dict(min_width=0.0045)),
    "V3": dict(name="Solo sin timeout", sim=dict(max_hold=None)),
    "A3": dict(name="Sniper Institucional", feat=_WIDE, sim=_WIDE_SIM, sig=dict(anomaly=True), parent="A1"),
    # Fase 4b: arbitro de regimen (src/regime_arbiter.py) sobre la BASE PERMANENTE
    "R1": dict(name="Trend-Aligned Fade", arb=dict(use_bias=True, g2=False), sim=BASE_SIM),
    "R2": dict(name="Absorcion Pura", arb=dict(use_bias=False, g2=False), sim=BASE_SIM),
    "R3": dict(name="Arbitro Completo AMT", arb=dict(use_bias=True, g2=True), sim=BASE_SIM),
}


def spec(model):
    m = MODELS[model]
    return dict(name=m["name"], feat=m.get("feat", {}), sig=m.get("sig", {}), sim=m.get("sim", {}),
                parent=m.get("parent"), arb=m.get("arb"))


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
    model = model or active_model()
    if spec(model)["arb"] is not None:
        return prepare(f, model)[1]
    sg = spec(model)["sig"]
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


def prepare(f, model=None):
    """(frame, sig) listos para futures_simulator.simulate. Los modelos del arbitro pueden devolver
    una copia del frame con los niveles de salida del Gatillo 2 en sus filas de senal."""
    model = model or active_model()
    arb = spec(model)["arb"]
    return arbitrate(f, **arb) if arb is not None else (f, generate_signals(f, model))


# ---------------- Fase 5: FRVP de sesion fija + barrido en 2 tiempos (src/session_frvp.py) ----------------
SESSION_VARIANTS = {
    "S1": dict(name="Sesion + Barrido 2T", cvd=False, week=False),
    "S2": dict(name="S1 + Divergencia CVD", cvd=True, week=False),
    "S3": dict(name="S2 + Regimen 5 dias", cvd=True, week=True),
}
SESSION_RULES = dict(k_max=12, sl_atr=0.35, min_stop=0.0035)
_S_SIM = dict(min_width=0.0045, max_hold=None, sl_atr=0.35, tp1_path=1.0, tp1_min_r=0.0)
SESSION_EXITS = {"TP1_100%": dict(_S_SIM, split=(1.0, 0.0, 0.0)),
                 "Escalera": dict(_S_SIM, split=(0.5, 0.5, 0.0))}


def session_signals(f, variant, **kw):
    """+1/-1 al cierre de t. LONG: alguna vela t-k (1..12, misma sesion) toco VAL_prev (sweep_low_1 = su minimo);
    t barre por debajo (low_t < sweep_low_1) y cierra de vuelta sobre VAL_prev y bajo POC_prev.
    cvd: CVD_t > CVD_(t-k) (divergencia de flujo). week: close sobre el POC de 5 dias. Espejo en VAH_prev.
    Regla anti-fees: se descarta si |entry - SL| / entry < 0.35%, SL = low_t - 0.35*ATR."""
    p, v = {**SESSION_RULES, **kw}, SESSION_VARIANTS[variant]
    h, l, c, atr, val, vah, poc, cvd, pw = (f[k].to_numpy(np.float64) for k in (
        "high", "low", "close", "atr", "val_prev", "vah_prev", "poc_prev", "cvd", "poc_week"))
    day, n = f["day"].to_numpy(), len(f)
    lo1, hi1 = np.full(n, np.inf), np.full(n, -np.inf)
    jl, jh = np.full(n, -1), np.full(n, -1)
    for k in range(1, p["k_max"] + 1):
        idx = np.arange(n) - k
        same = np.zeros(n, bool)
        same[k:] = day[k:] == day[:-k]
        lk, hk = np.r_[np.full(k, np.nan), l[:-k]], np.r_[np.full(k, np.nan), h[:-k]]
        with np.errstate(invalid="ignore"):
            cl, ch = same & (lk <= val) & (lk < lo1), same & (hk >= vah) & (hk > hi1)
        lo1, jl = np.where(cl, lk, lo1), np.where(cl, idx, jl)
        hi1, jh = np.where(ch, hk, hi1), np.where(ch, idx, jh)
    with np.errstate(invalid="ignore"):
        long = (jl >= 0) & (l < lo1) & (c > val) & (c < poc)
        short = (jh >= 0) & (h > hi1) & (c < vah) & (c > poc)
        if v["cvd"]:
            long &= cvd > cvd[np.maximum(jl, 0)]
            short &= cvd < cvd[np.maximum(jh, 0)]
        if v["week"]:
            long &= c > pw
            short &= c < pw
        long &= (c - (l - p["sl_atr"] * atr)) / c >= p["min_stop"]
        short &= ((h + p["sl_atr"] * atr) - c) / c >= p["min_stop"]
    sig = np.zeros(n, np.int8)
    sig[long] = 1
    sig[short & ~long] = -1
    return sig


def session_sim_frame(f):
    """Columnas que espera futures_simulator: TP1 = POC_prev, TP2 = VAH_prev/VAL_prev (TP3 = TP2)."""
    return f.assign(poc=f["poc_prev"], vah=f["vah_prev"], val=f["val_prev"],
                    swing_high=f["vah_prev"], swing_low=f["val_prev"])


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
        (fa, sa), (fb, sb) = prepare(a, m), prepare(b, m)
        ok &= np.array_equal(sa[:k + 1], sb[:k + 1])
        ok &= np.allclose(fa.loc[:k, cols], fb.loc[:k, cols], equal_nan=True)
        x = a.dropna(subset=["poc"])
        ok &= len(x) > 0 and bool(((x.swing_low <= x.val) & (x.val <= x.poc) & (x.poc <= x.vah)
                                   & (x.vah <= x.swing_high)).all())
    print("SELFTEST strategy", "OK" if ok else "FAIL")
    return bool(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print("modelo activo:", active_model(), "-", MODELS[active_model()]["name"])
