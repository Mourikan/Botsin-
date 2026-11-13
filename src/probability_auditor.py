"""Agente 3 - Auditor de probabilidad: reversion al POC desde VAH/VAL con costes reales.

Senal al cierre de i (niveles de [i-N, i-1]) -> entrada al open de i+1 -> TP = POC, SL = sl_mult * distancia al TP.
Coste por lado = 0.05% taker + 0.02% slippage. TP y SL en la misma vela => SL (peor caso).
"""
import argparse
import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

try:
    from src.frvp_engine import compute_frvp
except ImportError:  # ejecucion directa: python src/probability_auditor.py
    from frvp_engine import compute_frvp

FEE, SLIP = 0.0005, 0.0002
COST_SIDE = FEE + SLIP


def generate_signals(d):
    """+1 long: cierre cruza bajo VAL. -1 short: cierre cruza sobre VAH."""
    c, pc = d["close"].to_numpy(), d["close"].shift(1).to_numpy()
    vah, val, pvah, pval = (d["vah_price"].to_numpy(), d["val_price"].to_numpy(),
                            d["vah_price"].shift(1).to_numpy(), d["val_price"].shift(1).to_numpy())
    sig = np.zeros(len(d), np.int8)
    sig[(c < val) & (pc >= pval)] = 1
    sig[(c > vah) & (pc <= pvah)] = -1
    return sig


def simulate(d, sig, notional=1000.0, sl_mult=1.0, max_hold=48):
    """PnL neto USD por trade. Salidas vectorizadas; solo el filtro de no-solape es secuencial."""
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    n = len(d)
    s = np.flatnonzero(sig)
    s = s[s + max_hold < n]                      # necesita ventana futura completa
    if not len(s):
        return np.array([])
    side = sig[s].astype(np.float64)
    e = s + 1
    entry, tp = o[e], d["poc_price"].to_numpy()[s]
    dist = (tp - entry) * side
    ok = dist > 0                                # el gap de apertura no se comio el edge
    s, e, side, entry, tp, dist = s[ok], e[ok], side[ok], entry[ok], tp[ok], dist[ok]
    sl = entry - side * sl_mult * dist

    fh, fl = swv(h, max_hold)[e], swv(l, max_hold)[e]          # (trades, max_hold) desde la vela de entrada
    long = (side > 0)[:, None]
    hit_tp = np.where(long, fh >= tp[:, None], fl <= tp[:, None])
    hit_sl = np.where(long, fl <= sl[:, None], fh >= sl[:, None])
    t_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), max_hold)
    t_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), max_hold)
    t_exit = np.minimum(np.minimum(t_tp, t_sl), max_hold - 1)
    exit_px = np.where(t_sl <= t_tp, sl, tp)                   # empate => SL
    timeout = (t_tp == max_hold) & (t_sl == max_hold)
    exit_px = np.where(timeout, c[e + max_hold - 1], exit_px)

    qty = notional / entry
    pnl = side * (exit_px - entry) * qty - COST_SIDE * qty * (entry + exit_px)

    keep, free = [], -1                                         # una posicion a la vez
    for i in range(len(e)):
        if e[i] > free:
            keep.append(i)
            free = e[i] + t_exit[i]
    return pnl[keep]


def binom_sf(k, n, p):
    """P(X >= k), X ~ Binomial(n, p). Exacto, sin scipy."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    return min(1.0, sum(math.exp(lg(n + 1) - lg(j + 1) - lg(n - j + 1) + j * lp + (n - j) * lq)
                        for j in range(k, n + 1)))


def audit(pnl):
    """H0: win rate <= win rate de breakeven implicado por el payoff neto (cola derecha)."""
    n = len(pnl)
    if n == 0:
        return dict(n=0, wr=0.0, payoff=float("nan"), exp=0.0, pf=float("nan"), be=float("nan"), p=1.0)
    w, ls = pnl[pnl > 0], pnl[pnl <= 0]
    aw = w.mean() if len(w) else 0.0
    al = -ls.mean() if len(ls) else 0.0
    pw = len(w) / n
    be = al / (aw + al) if (aw + al) > 0 else 1.0
    return dict(n=n, wr=pw * 100, payoff=aw / al if al > 0 else float("inf"),
                exp=pw * aw - (1 - pw) * al,
                pf=w.sum() / -ls.sum() if ls.sum() < 0 else float("inf"),
                be=be * 100, p=binom_sf(len(w), n, be))


def report(m, label):
    edge = m["exp"] > 0 and m["p"] < 0.05
    rows = [("Total Trades", f"{m['n']}"), ("Win Rate %", f"{m['wr']:.2f}"),
            ("Payoff Ratio Neto", f"{m['payoff']:.3f}"), ("Esperanza ($/trade)", f"{m['exp']:+.4f}"),
            ("Profit Factor", f"{m['pf']:.3f}"), ("P-Value (binomial)", f"{m['p']:.4f}"),
            ("WR Breakeven %", f"{m['be']:.2f}")]
    out = [f"AUDITOR FRVP | {label}", "-" * 44]
    out += [f"{k:<24}{v:>20}" for k, v in rows]
    out.append(f"{'VEREDICTO':<24}{('EDGE SIGNIFICATIVO' if edge else 'SIN EDGE (no operar)'):>20}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_ohlcv.csv")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--sl-mult", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=48)
    a = ap.parse_args()
    d = compute_frvp(pd.read_csv(a.csv), a.lookback, a.bins)
    pnl = simulate(d, generate_signals(d), a.notional, a.sl_mult, a.max_hold)
    print(report(audit(pnl), f"N={a.lookback} SLx{a.sl_mult} ${a.notional:.0f}/trade"))


if __name__ == "__main__":
    main()
