"""Agente 2/3 - Arbitro de direccion y regimen (AMT) sobre el frame de build_features.

Causal: todo usa la vela i ya cerrada y niveles FRVP de velas <= i-1. Vectorizado numpy/pandas.
  A. Sesgo: close vs VWAP_288 + migracion del POC (24 velas). ER_48 mide tendencia vs rango.
  B. Gatillo 1: reversion por absorcion pasiva en VAL/VAH (RVOL + mecha de rechazo).
  C. Gatillo 2: continuacion (ruptura del VA + retest) cuando ER_48 >= 0.28.
El simulador no se toca: para el Gatillo 2 se devuelve una COPIA del frame con atr/poc/vah|val/swing
sobreescritos solo en las filas de senal, de modo que el simulador derive SL = entry -/+ 0.8*ATR y
TP1/TP2/TP3 = 1.3R / 2.5R / 4.0R (tras TP2 el stop sube al nivel de TP1).
"""
import numpy as np
import pandas as pd

CFG = dict(er_n=48, er_min=0.28, poc_lag=24, poc_tol=0.0005, slope_n=12,
           g1_rvol=1.4, g1_wick=0.40,
           g2_rvol=1.5, g2_lookback=3, g2_retest=0.0015, g2_sl_atr=0.8, g2_r=(1.3, 2.5, 4.0),
           min_width=0.0045, sim_sl_atr=0.5)


def regime(f, **kw):
    """Estado de subasta por vela: vwap_slope_pct, er, poc_mig y bias (+1 long / -1 short / 0 neutro)."""
    p = {**CFG, **kw}
    c, vw = f["close"], f["vwap"]
    r = pd.DataFrame(index=f.index)
    r["vwap_slope_pct"] = (vw / vw.shift(p["slope_n"]) - 1.0) * 100.0
    r["er"] = (c - c.shift(p["er_n"])).abs() / c.diff().abs().rolling(p["er_n"]).sum()
    r["poc_mig"] = f["poc"] / f["poc"].shift(p["poc_lag"]) - 1.0
    bias = np.zeros(len(f), np.int8)
    bias[((c > vw) & (r["poc_mig"] >= -p["poc_tol"])).to_numpy()] = 1    # prohibido short
    bias[((c < vw) & (r["poc_mig"] <= p["poc_tol"])).to_numpy()] = -1    # prohibido long
    r["bias"] = bias
    return r


def _recent(flag, n):
    """True si `flag` fue cierto en alguna de las n velas previas (excluye la actual)."""
    return pd.Series(flag.astype(np.float64)).shift(1).rolling(n).max().to_numpy() == 1.0


def arbitrate(f, use_bias=True, g2=False, **kw):
    """Devuelve (frame, sig). sig: +1 long / -1 short al cierre de la vela i."""
    p = {**CFG, **kw}
    o, h, l, c = (f[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    val, vah, poc, atr = (f[k].to_numpy(np.float64) for k in ("val", "vah", "poc", "atr"))
    rvol, dlt = f["rvol"].to_numpy(np.float64), f["volume_delta"].to_numpy(np.float64)
    wlo, whi = f["wick_lo"].to_numpy(np.float64), f["wick_hi"].to_numpy(np.float64)
    reg = regime(f, **kw)
    bias, er = reg["bias"].to_numpy(), reg["er"].to_numpy()
    n = len(f)
    yes = np.ones(n, bool)
    can_l, can_s = ((bias == 1), (bias == -1)) if use_bias else (yes, yes)
    sig = np.zeros(n, np.int8)
    with np.errstate(invalid="ignore"):
        # --- Gatillo 1: barrido del extremo + cierre dentro + esfuerzo (RVOL) + mecha de rechazo.
        # Delta: absorcion pasiva (agresores atrapados en la mecha y cierre en contra) o giro de delta.
        effort = rvol >= p["g1_rvol"]
        g1_l = ((l <= val) & (c > val) & (c < poc) & effort & (wlo >= p["g1_wick"])
                & ((dlt > 0) | ((dlt < 0) & (c > o))))
        g1_s = ((h >= vah) & (c < vah) & (c > poc) & effort & (whi >= p["g1_wick"])
                & ((dlt < 0) | ((dlt > 0) & (c < o))))
        sig[g1_l & can_l] = 1
        sig[g1_s & can_s & (sig == 0)] = -1
        if not g2:
            return f, sig
        # --- Gatillo 2: ruptura con participacion en las velas previas + retest del nivel roto.
        trend = er >= p["er_min"]
        width = (vah - val) / poc >= p["min_width"]
        up = _recent((c > vah) & (rvol >= p["g2_rvol"]) & (dlt > 0), p["g2_lookback"])
        dn = _recent((c < val) & (rvol >= p["g2_rvol"]) & (dlt < 0), p["g2_lookback"])
        ok = trend & width & np.isfinite(atr) & (sig == 0)
        g2_l = ok & (bias == 1) & up & (l <= vah * (1 + p["g2_retest"])) & (c > vah) & (wlo > 0)
        g2_s = ok & (bias == -1) & dn & (h >= val * (1 - p["g2_retest"])) & (c < val) & (whi > 0) & ~g2_l
    f = f.copy()
    risk = p["g2_sl_atr"] * atr
    r1, r2, r3 = p["g2_r"]
    for mask, sd, ext, far, swing in ((g2_l, 1.0, l, "vah", "swing_high"), (g2_s, -1.0, h, "val", "swing_low")):
        i = np.flatnonzero(mask)
        e, r = c[i], risk[i]
        f.loc[i, "atr"] = (r - sd * (e - ext[i])) / p["sim_sl_atr"]   # => SL del simulador = entry -/+ 0.8*ATR
        f.loc[i, "poc"] = e + sd * r1 * r                             # TP1
        f.loc[i, far] = e + sd * r2 * r                               # TP2
        f.loc[i, swing] = e + sd * r3 * r                             # TP3
        sig[i] = int(sd)
    return f, sig
