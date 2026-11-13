"""Agente 3 - Torneo de ablacion sobre el mismo dataset + ratchet anti-retroceso.

Ratchet:
  1. Un modelo con `parent` (= el mismo modelo sin ese filtro) solo es elegible si supera al padre
     en Net PnL y no baja su Profit Factor.
  2. best_benchmark.json (y con el el modelo por defecto) solo se sobreescribe si el mejor elegible
     supera el Net PnL guardado. --rebase unicamente al cambiar de dataset.
"""
import argparse
import json
import os
import time

import pandas as pd

try:
    from src.frvp_strategy import BENCH, MODELS, PARAMS, build_features, generate_signals, spec
    from src.futures_simulator import CFG, simulate
    from src.probability_auditor import audit
except ImportError:
    from frvp_strategy import BENCH, MODELS, PARAMS, build_features, generate_signals, spec
    from futures_simulator import CFG, simulate
    from probability_auditor import audit


def metrics(tr):
    m = audit(tr["net"].to_numpy())
    return dict(trades=int(m["n"]), win_rate=round(m["wr"], 2), gross=round(float(tr["gross"].sum()), 2),
                fees=round(float(tr["fees"].sum()), 2), net=round(float(tr["net"].sum()), 2),
                pf=round(min(m["pf"], 999.0), 3) if m["n"] else 0.0, expectancy=round(m["exp"], 4),
                p_value=round(m["p"], 4), open_at_end=int(tr["open_at_end"].sum()))


def run_models(df, models):
    cache, res = {}, {}
    for k in models:
        s = spec(k)
        key = tuple(sorted(s["feat"].items()))
        if key not in cache:
            cache[key] = build_features(df, **s["feat"])
        f = cache[key]
        res[k] = metrics(simulate(f, generate_signals(f, k), **s["sim"]))
    return res


def pick_winner(res):
    ok = []
    for k, r in res.items():
        par = spec(k)["parent"]
        if par in res and not (r["net"] > res[par]["net"] and r["pf"] >= res[par]["pf"]):
            continue                                   # el filtro resta frente a su modelo base
        if r["trades"]:
            ok.append(k)
    return max(ok, key=lambda k: res[k]["net"]) if ok else next(iter(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_ohlcv.csv")
    ap.add_argument("--models", default="A,A1,A2,A3", help=f"lista separada por comas de {list(MODELS)}")
    ap.add_argument("--rebase", action="store_true", help="fuerza reescritura del benchmark (dataset nuevo)")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    try:
        with open(os.path.splitext(a.csv)[0] + ".meta.json") as fh:
            meta = json.load(fh)
    except OSError:
        meta = {"rows": len(df)}
    res = run_models(df, a.models.split(","))
    win = pick_winner(res)

    hdr = (f"{'Modelo':<25}|{'Trades':>7} |{'Win Rate %':>11} |{'Gross PnL ($)':>14} |{'Fees ($)':>9} |"
           f"{'Net PnL ($ USD)':>16} |{'Profit Factor':>14} |{'Expectancy ($/trade)':>21} |{'P-Value':>8}")
    print(f"TORNEO ABLACION | {meta.get('market', '?')} {meta.get('symbol', '?')} {meta.get('interval', '?')} "
          f"| {len(df)} velas | {meta.get('first', '?')} -> {meta.get('last', '?')}")
    print(f"Capital ${CFG['capital']:.0f} | riesgo {CFG['risk_pct']:.1%}/trade | lev max {CFG['max_lev']:.0f}x "
          f"| maker {CFG['maker']:.2%} | stop taker {CFG['taker']:.2%} + slip {CFG['slip']:.2%}")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k, r in res.items():
        print(f"{k + ' ' + MODELS[k]['name']:<25}|{r['trades']:>7} |{r['win_rate']:>11.2f} |{r['gross']:>+14.2f} |"
              f"{r['fees']:>9.2f} |{r['net']:>+16.2f} |{r['pf']:>14.3f} |{r['expectancy']:>+21.4f} |{r['p_value']:>8.4f}")
    print("-" * len(hdr))
    w = res[win]
    edge = w["net"] > 0 and w["p_value"] < 0.05
    print(f"MEJOR ELEGIBLE: {win} ({MODELS[win]['name']}) | Net ${w['net']:+.2f} | "
          f"{'EDGE SIGNIFICATIVO' if edge else 'SIN EDGE ESTADISTICO (p >= 0.05 o Net <= 0): no operar en real'}")

    prev = None
    if os.path.exists(BENCH):
        with open(BENCH) as fh:
            prev = json.load(fh)
    prev_net = prev["metrics"][prev["winner"]]["net"] if prev else None
    if prev is None or a.rebase or w["net"] > prev_net:
        with open(BENCH, "w") as fh:
            json.dump({"winner": win, "winner_name": MODELS[win]["name"], "edge_significant": edge,
                       "dataset": meta, "metrics": res, "winner_spec": spec(win), "base_params": PARAMS,
                       "base_sim_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
                       "previous": {"winner": prev["winner"], "net": prev_net} if prev else None,
                       "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, fh, indent=1)
        print(f"RATCHET: supera ${prev_net if prev else 0:+.2f} -> best_benchmark.json actualizado, modelo por defecto = {win}")
    else:
        print(f"RATCHET: nadie supera el benchmark ({prev['winner']} ${prev_net:+.2f}); "
              f"modelo por defecto intacto = {prev['winner']}")


if __name__ == "__main__":
    main()
