"""Agente 2 - FRVP de sesion fija UTC, sin repintado.

Para cada vela del dia D: POC/VAH/VAL del perfil de las 288 velas del dia D-1 completo (congelado todo D)
y POC del perfil de los 5 dias previos (D-5..D-1) como regimen semanal. CVD acumulado desde 00:00 UTC.
Causal por construccion: el dia D solo ve dias ya cerrados; CVD/ATR usan velas <= i.
"""
import sys

import numpy as np
import pandas as pd

try:
    from src.frvp_engine import profile_levels, synthetic
except ImportError:
    from frvp_engine import profile_levels, synthetic

DAY_MS, BAR_MS = 86_400_000, 300_000
CFG = dict(bins=60, va_pct=0.70, min_bars=280, week_days=5, atr_n=14)


def _day_matrix(t, *cols):
    """(n_days, 288) por columna + mascara de velas presentes + indice de dia por vela."""
    day = t // DAY_MS
    d0 = day.min()
    di, slot = (day - d0).astype(np.int64), ((t % DAY_MS) // BAR_MS).astype(np.int64)
    nd, per = int(di.max()) + 1, DAY_MS // BAR_MS
    mask = np.zeros((nd, per), bool)
    mask[di, slot] = True
    mats = []
    for c in cols:
        m = np.zeros((nd, per))
        m[di, slot] = c
        mats.append(m)
    return di, mask, mats


def session_levels(df, **kw):
    """Anade poc_prev, vah_prev, val_prev, poc_week, cvd, atr al frame (copia)."""
    p = {**CFG, **kw}
    t = df["open_time"].to_numpy(np.int64)
    h, l, c, v = (df[k].to_numpy(np.float64) for k in ("high", "low", "close", "volume"))
    di, mask, (H, L, V) = _day_matrix(t, h, l, v)
    nd = mask.shape[0]
    full = mask.sum(axis=1) >= p["min_bars"]
    H = np.where(mask, H, 1.0); L = np.where(mask, L, 1.0)          # relleno inocuo (mascara lo excluye)
    lv = np.full((nd, 3), np.nan)
    ok = np.flatnonzero(full)
    if len(ok):
        poc, vah, val, _, _ = profile_levels(H[ok], L[ok], V[ok], p["bins"], p["va_pct"], mask=mask[ok])
        lv[ok] = np.c_[poc, vah, val]
    wk = np.full(nd, np.nan)
    W = p["week_days"]
    rows = [d for d in range(W, nd) if full[d - W:d].all()]
    for s in range(0, len(rows), 16):                                # perfil de 5 dias (1440 velas)
        r = np.array(rows[s:s + 16])
        idx = r[:, None] - W + np.arange(W)
        cat = lambda M: M[idx].reshape(len(r), -1)
        wk[r] = profile_levels(cat(H), cat(L), cat(V), p["bins"], p["va_pct"], mask=cat(mask))[0]
    prev = np.full((nd, 3), np.nan)
    prev[1:] = lv[:-1]                                               # dia D usa el perfil del dia D-1
    f = df.copy()
    f["poc_prev"], f["vah_prev"], f["val_prev"] = prev[di, 0], prev[di, 1], prev[di, 2]
    f["poc_week"] = wk[di]
    f["day"] = di
    f["cvd"] = df.groupby(di)["volume_delta"].cumsum().to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    f["atr"] = pd.Series(tr).ewm(alpha=1.0 / p["atr_n"], adjust=False, min_periods=p["atr_n"]).mean().to_numpy()
    return f


def selftest():
    """Causalidad: alterar velas > k no cambia niveles/CVD/ATR ni senales en filas <= k; niveles fijos intradia."""
    try:
        from src.frvp_strategy import SESSION_VARIANTS, session_signals
    except ImportError:
        from frvp_strategy import SESSION_VARIANTS, session_signals
    df, k = synthetic(288 * 9, 5), 288 * 7 + 100
    d2 = df.copy()
    d2.loc[k + 1:, ["open", "high", "low", "close"]] *= 1.3
    d2.loc[k + 1:, ["volume", "volume_delta"]] *= 5
    a, b = session_levels(df), session_levels(d2)
    cols = ["poc_prev", "vah_prev", "val_prev", "poc_week", "cvd", "atr"]
    ok = np.allclose(a.loc[:k, cols], b.loc[:k, cols], equal_nan=True)
    ok &= bool((a.groupby("day")["poc_prev"].nunique(dropna=False) <= 1).all())
    x = a.dropna(subset=["poc_prev"])
    ok &= len(x) > 0 and bool(((x.val_prev <= x.poc_prev) & (x.poc_prev <= x.vah_prev)).all())
    for v in SESSION_VARIANTS:
        ok &= np.array_equal(session_signals(a, v)[:k + 1], session_signals(b, v)[:k + 1])
    print("SELFTEST session", "OK" if ok else "FAIL")
    return bool(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
