"""Agente 3 - Fase 3: aislamiento de variables de A1 + validacion out-of-sample (OOS).

Metrica reina del ratchet: Net PnL ($ USD) fuera de muestra. Un modelo solo queda "validado" si en
TODOS los datasets OOS tiene Net > 0 y p-value < 0.05. El resultado se anota en best_benchmark.json.
"""
import argparse
import json
import os
import time

import pandas as pd

try:
    from src.ablation_runner import run_models
    from src.frvp_strategy import BENCH, MODELS
except ImportError:
    from ablation_runner import run_models
    from frvp_strategy import BENCH, MODELS

OOS = {"BTC reciente": "data/oos_btc_recent.csv", "SOL 2024-03 tendencia": "data/oos_sol_2024_03.csv"}


def isolate(csv):
    res = run_models(pd.read_csv(csv), ["V1", "V2", "V3"])
    print(f"AISLAMIENTO A1 | {csv}")
    print(f"{'Modelo':<24}|{'Gross PnL ($)':>14} |{'Net PnL ($ USD)':>16}")
    for k, r in res.items():
        print(f"{k + ' ' + MODELS[k]['name']:<24}|{r['gross']:>+14.2f} |{r['net']:>+16.2f}")


def validate(models):
    hdr = (f"{'Dataset':<22}|{'Modelo':>7} |{'Trades':>7} |{'Win Rate %':>11} |{'Gross PnL ($)':>14} |"
           f"{'Fees ($)':>9} |{'Net PnL ($ USD)':>16} |{'P-Value':>8}")
    print("VALIDACION OUT-OF-SAMPLE")
    print(hdr)
    print("-" * len(hdr))
    out, spans = {}, {}
    for name, csv in OOS.items():
        df = pd.read_csv(csv)
        t = pd.to_datetime(df["open_time"], unit="ms")
        spans[name] = f"{len(df)} velas {t.iloc[0]:%Y-%m-%d} -> {t.iloc[-1]:%Y-%m-%d}"
        out[name] = run_models(df, models)
        for k, r in out[name].items():
            print(f"{name:<22}|{k:>7} |{r['trades']:>7} |{r['win_rate']:>11.2f} |{r['gross']:>+14.2f} |"
                  f"{r['fees']:>9.2f} |{r['net']:>+16.2f} |{r['p_value']:>8.4f}")
    print("-" * len(hdr))
    total = {k: round(sum(out[d][k]["net"] for d in out), 2) for k in models}
    passed = {k: all(out[d][k]["net"] > 0 and out[d][k]["p_value"] < 0.05 for d in out) for k in models}
    best = max(models, key=lambda k: total[k])
    print(" | ".join(f"{k}: Net OOS total ${total[k]:+.2f} {'VALIDADO' if passed[k] else 'NO VALIDADO'}" for k in models))
    print("VEREDICTO:", "edge confirmado OOS" if any(passed.values())
          else "ningun modelo sobrevive OOS -> rediseno de estructura antes de continuar")
    if os.path.exists(BENCH):
        with open(BENCH) as fh:
            b = json.load(fh)
        b.update(winner=best, winner_name=MODELS[best]["name"], oos_validated=bool(passed[best]),
                 oos={"datasets": spans, "metrics": out, "net_total": total, "passed": passed,
                      "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
        with open(BENCH, "w") as fh:
            json.dump(b, fh, indent=1)
        print(f"RATCHET OOS: modelo por defecto = {best} (mayor Net OOS), oos_validated = {passed[best]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolate", action="store_true", help="micro-test de las 3 variables de A1")
    ap.add_argument("--csv", default="data/sample_ohlcv.csv")
    ap.add_argument("--models", default="A,A2")
    a = ap.parse_args()
    isolate(a.csv) if a.isolate else validate(a.models.split(","))
