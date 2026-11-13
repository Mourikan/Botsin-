"""Agente 1 - Simulador USD de Binance Futures USDT-M con escalera TP1/TP2/TP3 + breakeven.

Senal al cierre de i -> orden LIMIT al close[i], valida solo durante la vela i+1 (se llena si el
precio la atraviesa). Entrada y TPs = maker. Stops (SL, BE, POC) y timeout = taker + slippage.
Dentro de una vela se evalua SIEMPRE el stop antes que los TP (peor caso); no hay TP en la vela de fill.
No modela funding (se evita operar alrededor del cobro).
"""
import numpy as np
import pandas as pd

CFG = dict(capital=1000.0, risk_pct=0.01, max_lev=10.0, maker=0.0002, taker=0.0005, slip=0.0002,
           min_width=0.0035, funding_guard_min=3, sl_atr=0.5, tp1_path=0.90, tp1_min_r=1.2,
           be_offset=0.0008, split=(0.4, 0.4, 0.2), max_hold=144)
FUNDING_MS = 8 * 3600 * 1000


def pretrade_ok(f, guard_min=3, min_width=0.0035):
    """Por vela de senal i: rango suficiente y vela de entrada (i+1) fuera de +-guard de funding."""
    t = f["open_time"].to_numpy(np.int64)
    step = int(np.median(np.diff(t)))
    te = t + step                                   # apertura de la vela de entrada
    g = guard_min * 60_000
    m = te % FUNDING_MS
    near = (m < g) | (m + step > FUNDING_MS - g)    # la vela de entrada pisa [F-g, F+g]
    width = ((f["vah"] - f["val"]) / f["poc"]).to_numpy()
    return (width >= min_width) & ~near


def simulate(f, sig, **kw):
    """Devuelve DataFrame de trades (gross, fees, net en USD). Una posicion a la vez."""
    p = {**CFG, **kw}
    o, h, l, c = (f[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    poc, vah, val, atr = (f[k].to_numpy(np.float64) for k in ("poc", "vah", "val", "atr"))
    s_lo, s_hi = f["swing_low"].to_numpy(np.float64), f["swing_high"].to_numpy(np.float64)
    ok = pretrade_ok(f, p["funding_guard_min"], p["min_width"])
    n, equity, risk_usd = len(f), p["capital"], p["capital"] * p["risk_pct"]
    stop_cost = p["taker"] + p["slip"]
    trades, free = [], 0
    for i in np.flatnonzero(sig):
        e = i + 1
        if i < free or e >= n or not ok[i] or not np.isfinite(atr[i]) or equity <= 0:
            continue
        sd = float(sig[i])
        entry = c[i]
        if not (l[e] < entry if sd > 0 else h[e] > entry):     # limit no atravesada -> cancelada
            continue
        sl = l[i] - p["sl_atr"] * atr[i] if sd > 0 else h[i] + p["sl_atr"] * atr[i]
        r = (entry - sl) * sd
        if r <= 0:
            continue
        tp1 = entry + sd * max(p["tp1_path"] * (poc[i] - entry) * sd, p["tp1_min_r"] * r)
        tp2 = vah[i] if sd > 0 else val[i]
        tp3 = s_hi[i] if sd > 0 else s_lo[i]
        if not ((tp2 - tp1) * sd > 0 and (tp3 - tp2) * sd >= 0):  # escalera incoherente -> no trade
            continue
        notional = min(risk_usd / (r / entry), p["max_lev"] * equity)
        qty = notional / entry
        tps = (tp1, tp2, tp3)
        gross, fees, rem, stage, stop = 0.0, notional * p["maker"], 1.0, 0, sl
        last = n - 1 if p["max_hold"] is None else min(e + p["max_hold"], n) - 1  # None = sin timeout
        eod = False
        t = e
        for t in range(e, last + 1):
            if (l[t] <= stop) if sd > 0 else (h[t] >= stop):
                px = stop if t == e else (min(stop, o[t]) if sd > 0 else max(stop, o[t]))  # gap => open
                gross += rem * qty * (px - entry) * sd
                fees += rem * qty * px * stop_cost
                rem = 0.0
                break
            if t == e:
                continue
            while stage < 3 and ((h[t] >= tps[stage]) if sd > 0 else (l[t] <= tps[stage])):
                part = p["split"][stage]
                gross += part * qty * (tps[stage] - entry) * sd
                fees += part * qty * tps[stage] * p["maker"]
                rem -= part
                stage += 1
                if stage == 1:
                    stop = entry * (1 + sd * p["be_offset"])               # breakeven neto
                elif stage == 2:
                    stop = max(stop, poc[i]) if sd > 0 else min(stop, poc[i])
            if stage == 3 or rem <= 1e-9:                                  # posicion ya cerrada por los TP
                rem = 0.0
                break
        if rem > 1e-9:                                                     # timeout / fin de datos (mark-to-market)
            eod = p["max_hold"] is None or t == n - 1
            gross += rem * qty * (c[t] - entry) * sd
            fees += rem * qty * c[t] * stop_cost
        net = gross - fees
        equity += net
        free = t
        trades.append((i, t, sd, entry, sl, notional, stage, gross, fees, net, equity, eod))
    return pd.DataFrame(trades, columns=["sig_bar", "exit_bar", "side", "entry", "sl", "notional",
                                         "tps_hit", "gross", "fees", "net", "equity", "open_at_end"])
