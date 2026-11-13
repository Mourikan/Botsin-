"""Agente 2 - Fixed Range Volume Profile rolling, vectorizado en NumPy.

Causalidad: el perfil de la vela i usa SOLO las N velas cerradas previas [i-N, i-1].
"""
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv


def _value_area(prof, poc, va_pct):
    """Expande desde el POC anadiendo el bin vecino de mayor volumen hasta cubrir va_pct."""
    W, B = prof.shape
    r = np.arange(W)
    lo, hi = poc.copy(), poc.copy()
    acc = prof[r, poc].copy()
    target = va_pct * prof.sum(axis=1)
    for _ in range(B - 1):
        active = acc < target
        if not active.any():
            break
        up = np.where(hi + 1 < B, prof[r, np.minimum(hi + 1, B - 1)], -1.0)
        dn = np.where(lo - 1 >= 0, prof[r, np.maximum(lo - 1, 0)], -1.0)
        go_up = active & (up >= dn) & (up >= 0)
        go_dn = active & ~go_up & (dn >= 0)
        hi = hi + go_up
        lo = lo - go_dn
        acc = acc + np.where(go_up, up, 0.0) + np.where(go_dn, dn, 0.0)
    return lo, hi


def compute_frvp(df, lookback=120, bins=48, va_pct=0.70, chunk=256):
    """Anade poc_price, vah_price, val_price, poc_dist_pct y rvol. Primeras `lookback` filas = NaN.

    El volumen de cada vela se reparte uniformemente sobre los bins que cubre su rango [low, high].
    """
    h, l, c, v = (df[k].to_numpy(np.float64) for k in ("high", "low", "close", "volume"))
    n = len(df)
    out = {k: np.full(n, np.nan) for k in ("poc_price", "vah_price", "val_price", "rvol")}
    if n <= lookback:
        return df.assign(**out, poc_dist_pct=np.nan)

    # ventana w cubre velas [w, w+N-1] y alimenta a la vela i = w+N  -> se descarta la ultima ventana
    H, L, V = (swv(a, lookback)[:-1] for a in (h, l, v))
    W = H.shape[0]
    poc = np.empty(W); vah = np.empty(W); val = np.empty(W)
    for s in range(0, W, chunk):
        Hc, Lc, Vc = H[s:s + chunk], L[s:s + chunk], V[s:s + chunk]
        pmin = Lc.min(axis=1)
        step = np.maximum((Hc.max(axis=1) - pmin) / bins, 1e-12)
        edges = pmin[:, None] + step[:, None] * np.arange(bins + 1)        # (w, B+1)
        b_lo, b_hi = edges[:, None, :-1], edges[:, None, 1:]               # (w, 1, B)
        ov = np.minimum(Hc[:, :, None], b_hi) - np.maximum(Lc[:, :, None], b_lo)
        np.clip(ov, 0.0, None, out=ov)                                     # (w, N, B)
        rng = (Hc - Lc)[:, :, None]
        flat = rng <= 0                                                    # doji sin rango: todo a su bin
        idx = np.clip(((Hc - pmin[:, None]) / step[:, None]).astype(np.int64), 0, bins - 1)
        frac = np.where(flat, np.arange(bins) == idx[:, :, None], ov / np.where(flat, 1.0, rng))
        prof = (frac * Vc[:, :, None]).sum(axis=1)                         # (w, B)
        p = prof.argmax(axis=1)
        lo, hi = _value_area(prof, p, va_pct)
        poc[s:s + chunk] = pmin + (p + 0.5) * step
        val[s:s + chunk] = pmin + lo * step
        vah[s:s + chunk] = pmin + (hi + 1) * step

    out["poc_price"][lookback:] = poc
    out["vah_price"][lookback:] = vah
    out["val_price"][lookback:] = val
    out["rvol"][lookback:] = v[lookback:] / np.maximum(V.mean(axis=1), 1e-12)  # vela i cerrada vs media previa
    res = df.assign(**out)
    res["poc_dist_pct"] = (c - res["poc_price"]) / res["poc_price"] * 100.0
    return res


def selftest():
    """Causalidad: alterar velas >= k no puede cambiar niveles en filas <= k."""
    rs = np.random.default_rng(7)
    n, k = 600, 400
    c = 100 + np.cumsum(rs.normal(0, 0.3, n))
    o = c + rs.normal(0, 0.1, n)
    df = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) + rs.random(n) * 0.3,
                       "low": np.minimum(o, c) - rs.random(n) * 0.3, "volume": rs.random(n) * 100 + 1})
    a = compute_frvp(df)
    d2 = df.copy()
    d2.loc[k:, ["open", "high", "low", "close"]] *= 1.5
    d2.loc[k:, "volume"] *= 9
    b = compute_frvp(d2)
    cols = ["poc_price", "vah_price", "val_price"]
    ok = np.allclose(a.loc[:k, cols], b.loc[:k, cols], equal_nan=True)
    x = a.dropna()
    ok &= bool(((x.val_price <= x.poc_price) & (x.poc_price <= x.vah_price)).all())
    print("SELFTEST", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    d = compute_frvp(pd.read_csv("data/sample_ohlcv.csv"))
    print(d[["close", "poc_price", "vah_price", "val_price", "poc_dist_pct", "rvol"]].tail(5).round(3).to_string())
