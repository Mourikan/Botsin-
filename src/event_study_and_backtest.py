"""Agente 3 - Fase 5: validacion en 2 etapas con holdout intocable.

Reglas PRE-REGISTRADAS (fijadas antes de ver resultados):
  SPLIT   : 6 meses 5m. Desarrollo = primer 70% de dias UTC. Holdout = 30% restante, excluyendo dias >= 2026-10-25
            (ya usados para disenar en Fases 1-4).
  ETAPA A : event study en Desarrollo SOL. Retorno a favor de la senal desde close[t] a +6/+12/+24 velas, en ATR(t).
            Una variante PASA si ret+12 >= 0.30 ATR (coste aprox. de fees) y p-value (t-test 2 colas) < 0.05.
            BTC Desarrollo se muestra como validacion cruzada, no decide.
  ETAPA B : solo las variantes que pasan van al simulador en Desarrollo y Holdout (SOL; BTC = validacion cruzada).
            Si ninguna pasa, el simulador corre SOLO en Desarrollo (diagnostico) y el holdout queda sellado.
  RATCHET : best_benchmark.json solo se actualiza si una variante es positiva en el Holdout SOL con PF > 1.25.
El holdout se ejecuta una sola vez: deja constancia en data/holdout_used.json y se niega a repetir sin --force.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from src.ablation_runner import metrics
from src.frvp_strategy import BENCH, SESSION_EXITS, SESSION_VARIANTS, session_signals, session_sim_frame
from src.futures_simulator import simulate
from src.session_frvp import session_levels

DATA = {"SOL": "data/solusdt_5m_6m.csv.gz", "BTC": "data/btcusdt_5m_6m.csv.gz"}
DEV_FRAC, CONTAMINATED_FROM = 0.70, "2026-10-25"
HORIZONS, PASS_RET, PASS_P = (6, 12, 24), 0.30, 0.05
SEAL = "data/holdout_used.json"


def t_pvalue(x):
    """p-value 2 colas del t-test de una muestra contra 0 (integracion numerica de la t de Student)."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return 1.0
    t, nu = abs(x.mean() / (x.std(ddof=1) / math.sqrt(n))), n - 1
    g = t + np.geomspace(1e-9, 1e6, 400_001) - 1e-9
    logc = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    pdf = np.exp(logc - (nu + 1) / 2 * np.log1p(g * g / nu))
    return float(min(1.0, 2 * np.sum((pdf[1:] + pdf[:-1]) / 2 * np.diff(g))))


def segments(f):
    days = np.sort(f["day"].unique())
    cut = days[int(len(days) * DEV_FRAC)]
    stop = (pd.Timestamp(CONTAMINATED_FROM).value // 10**6) // 86_400_000 - f["open_time"].iloc[0] // 86_400_000
    d = f["day"].to_numpy()
    return {"Desarrollo": d < cut, "Holdout": (d >= cut) & (d < stop)}


def event_study(f, seg, variant):
    sig = session_signals(f, variant)
    c, atr = f["close"].to_numpy(), f["atr"].to_numpy()
    i = np.flatnonzero((sig != 0) & seg)
    i = i[i + max(HORIZONS) < len(f)]
    sd = sig[i].astype(float)
    r = {h: (c[i + h] - c[i]) * sd / atr[i] for h in HORIZONS}
    return dict(n=len(i), **{f"r{h}": float(np.nanmean(r[h])) if len(i) else np.nan for h in HORIZONS},
                p=t_pvalue(r[12]) if len(i) else 1.0)


def backtest(f, seg, variant, exit_name):
    sf = session_sim_frame(f)
    sig = session_signals(f, variant)
    idx = np.flatnonzero(seg)
    sub, s = sf.iloc[idx[0]:idx[-1] + 1].reset_index(drop=True), sig[idx[0]:idx[-1] + 1].copy()
    return metrics(simulate(sub, s, **SESSION_EXITS[exit_name]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="permite re-ejecutar el holdout (queda registrado)")
    a = ap.parse_args()
    F = {k: session_levels(pd.read_csv(p)) for k, p in DATA.items()}
    S = {k: segments(f) for k, f in F.items()}
    print(f"{'ETAPA A - Event Study (Desarrollo)':<38}| {'N':>5} | {'+6v ATR':>8} | {'+12v ATR':>8} | "
          f"{'+24v ATR':>8} | {'P(+12)':>7}")
    passed, es = [], {}
    for sym in DATA:
        for v in SESSION_VARIANTS:
            r = es[(sym, v)] = event_study(F[sym], S[sym]["Desarrollo"], v)
            ok = sym == "SOL" and r["n"] > 0 and r["r12"] >= PASS_RET and r["p"] < PASS_P
            if ok:
                passed.append(v)
            print(f"{sym + ' ' + v + ' ' + SESSION_VARIANTS[v]['name']:<38}| {r['n']:>5} | {r['r6']:>+8.3f} | "
                  f"{r['r12']:>+8.3f} | {r['r24']:>+8.3f} | {r['p']:>7.4f}" + ("  PASA" if ok else ""))
    run_holdout = bool(passed)
    if run_holdout and os.path.exists(SEAL) and not a.force:
        raise SystemExit(f"Holdout ya usado ({SEAL}); repetirlo invalida la prueba. Usa --force si es deliberado.")
    variants = passed or list(SESSION_VARIANTS)
    tramos = [("Desarrollo", "SOL")] + ([("Holdout", "SOL"), ("Holdout", "BTC")] if run_holdout else [])
    print(f"ETAPA B - Simulador | {'Variante':<8}| {'Salida':<9}| {'Trades':>6} | {'WR %':>6} | {'Gross $':>8} | "
          f"{'Fees $':>7} | {'Net $':>8} | {'PF':>5} | {'P-Val':>6}")
    res = {}
    for tramo, sym in tramos:
        for v in variants:
            for ex in SESSION_EXITS:
                r = res[(tramo, sym, v, ex)] = backtest(F[sym], S[sym][tramo], v, ex)
                print(f"{tramo + ' ' + sym:<20}| {v:<8}| {ex:<9}| {r['trades']:>6} | {r['win_rate']:>6.2f} | "
                      f"{r['gross']:>+8.2f} | {r['fees']:>7.2f} | {r['net']:>+8.2f} | {r['pf']:>5.2f} | {r['p_value']:>6.4f}")
    if not run_holdout:
        print("VEREDICTO: ninguna variante pasa el Event Study -> Holdout SELLADO (no ejecutado). Etapa B = diagnostico IS.")
        return
    with open(SEAL, "w") as fh:
        json.dump({"used_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "variants": passed}, fh)
    win = [(k, r) for k, r in res.items() if k[0] == "Holdout" and k[1] == "SOL" and r["net"] > 0 and r["pf"] > 1.25]
    if not win:
        print("VEREDICTO: ninguna variante es positiva en el Holdout SOL con PF > 1.25 -> benchmark intacto.")
        return
    (_, _, v, ex), r = max(win, key=lambda kr: kr[1]["net"])
    with open(BENCH) as fh:
        b = json.load(fh)
    b["session_phase5"] = {"variant": v, "exit": ex, "holdout_sol": r,
                           "all": {" | ".join(k): x for k, x in res.items()},
                           "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    with open(BENCH, "w") as fh:
        json.dump(b, fh, indent=1)
    print(f"VEREDICTO: {v} {ex} positiva en Holdout SOL (Net ${r['net']:+.2f}, PF {r['pf']:.2f}) -> best_benchmark.json")


if __name__ == "__main__":
    main()
