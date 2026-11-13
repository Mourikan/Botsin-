"""Agente 3 - Torneo In-Sample + Out-of-Sample.

Metrica reina: Net PnL ($ USD) consolidado. El modelo por defecto es el de mayor Net consolidado,
pero solo queda `oos_validated` si en TODOS los datasets OOS tiene Net > 0 y p-value < 0.05.
"""
import argparse
import json
import os
import time

import pandas as pd

try:
    from src.ablation_runner import metrics
    from src.frvp_strategy import BENCH, MODELS, build_features, prepare, spec
    from src.futures_simulator import simulate
except ImportError:
    from ablation_runner import metrics
    from frvp_strategy import BENCH, MODELS, build_features, prepare, spec
    from futures_simulator import simulate

DATASETS = {  # nombre -> (csv, es_oos)
    "SOL reciente (IS)": ("data/sample_ohlcv.csv", False),
    "SOL 2024-03 tendencia": ("data/oos_sol_2024_03.csv", True),
    "BTC reciente": ("data/oos_btc_recent.csv", True),
}


def run(df, models):
    cache, res = {}, {}
    for k in models:
        s = spec(k)
        key = tuple(sorted(s["feat"].items()))
        if key not in cache:
            cache[key] = build_features(df, **s["feat"])
        frame, sig = prepare(cache[key], k)
        res[k] = metrics(simulate(frame, sig, **s["sim"]))
    return res


def isolate(csv, models):
    res = run(pd.read_csv(csv), models)
    print(f"{'Modelo':<24}|{'Gross PnL ($)':>14} |{'Net PnL ($ USD)':>16}")
    for k, r in res.items():
        print(f"{k + ' ' + MODELS[k]['name']:<24}|{r['gross']:>+14.2f} |{r['net']:>+16.2f}")


def tournament(models):
    hdr = (f"{'Dataset':<22}|{'Modelo':>7} |{'Trades':>7} |{'Win Rate %':>11} |{'Gross PnL ($)':>14} |"
           f"{'Fees ($)':>9} |{'Net PnL ($ USD)':>16} |{'Profit Factor':>14} |{'P-Value':>8}")
    print(hdr)
    print("-" * len(hdr))
    out, spans = {}, {}
    for name, (csv, _) in DATASETS.items():
        df = pd.read_csv(csv)
        t = pd.to_datetime(df["open_time"], unit="ms")
        spans[name] = f"{len(df)} velas {t.iloc[0]:%Y-%m-%d} -> {t.iloc[-1]:%Y-%m-%d}"
        out[name] = run(df, models)
        for k, r in out[name].items():
            print(f"{name:<22}|{k:>7} |{r['trades']:>7} |{r['win_rate']:>11.2f} |{r['gross']:>+14.2f} |"
                  f"{r['fees']:>9.2f} |{r['net']:>+16.2f} |{r['pf']:>14.3f} |{r['p_value']:>8.4f}")
    print("-" * len(hdr))
    oos = [d for d, (_, is_oos) in DATASETS.items() if is_oos]
    total = {k: round(sum(out[d][k]["net"] for d in out), 2) for k in models}
    trades = {k: sum(out[d][k]["trades"] for d in out) for k in models}
    passed = {k: all(out[d][k]["net"] > 0 and out[d][k]["p_value"] < 0.05 for d in oos) for k in models}
    for k in models:
        print(f"CONSOLIDADO {k} {MODELS[k]['name']:<22}| {trades[k]:>4} trades | Net ${total[k]:+9.2f} | "
              f"{'OOS VALIDADO' if passed[k] else 'OOS NO VALIDADO'}")
    best = max(models, key=lambda k: total[k])
    b = {}
    if os.path.exists(BENCH):
        with open(BENCH) as fh:
            b = json.load(fh)
    first = next(iter(DATASETS))
    b.update(winner=best, winner_name=MODELS[best]["name"], winner_spec=spec(best),
             edge_significant=bool(passed[best]), oos_validated=bool(passed[best]),
             metrics=out[first],  # in-sample, formato que lee ablation_runner
             tournament={"datasets": spans, "metrics": out, "net_total": total, "trades_total": trades,
                         "passed": passed, "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
    b.pop("oos", None)
    with open(BENCH, "w") as fh:
        json.dump(b, fh, indent=1)
    print(f"RATCHET: modelo por defecto = {best} (mayor Net consolidado ${total[best]:+.2f}) | "
          f"oos_validated = {passed[best]}" + ("" if passed[best] else " -> NO OPERAR EN REAL"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolate", action="store_true", help="micro-tabla Gross/Net sobre un solo csv")
    ap.add_argument("--csv", default="data/sample_ohlcv.csv")
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    if a.isolate:
        isolate(a.csv, (a.models or "V1,V2,V3").split(","))
    else:
        tournament((a.models or "R1,R2,R3").split(","))
