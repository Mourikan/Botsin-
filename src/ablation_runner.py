"""Agente 3 - Torneo de ablacion A/B/C sobre el mismo dataset + ratchet anti-retroceso.

Ratchet: un filtro RSI (B o C) solo puede quedar activo si mejora a la vez Net PnL y Profit Factor
frente al modelo A sin filtro. best_benchmark.json solo se sobreescribe si el Net PnL mejora
(o con --rebase al cambiar de dataset).
"""
import argparse
import json
import os
import time

import pandas as pd

try:
    from src.frvp_strategy import BENCH, MODELS, PARAMS, build_features, generate_signals
    from src.futures_simulator import CFG, simulate
    from src.probability_auditor import audit
except ImportError:
    from frvp_strategy import BENCH, MODELS, PARAMS, build_features, generate_signals
    from futures_simulator import CFG, simulate
    from probability_auditor import audit


def metrics(tr):
    m = audit(tr["net"].to_numpy())
    return dict(trades=int(m["n"]), win_rate=round(m["wr"], 2), gross=round(float(tr["gross"].sum()), 2),
                fees=round(float(tr["fees"].sum()), 2), net=round(float(tr["net"].sum()), 2),
                pf=round(m["pf"], 3) if m["n"] else 0.0, expectancy=round(m["exp"], 4), p_value=round(m["p"], 4))


def pick_winner(res):
    """A por defecto; un filtro solo gana si supera a A en Net PnL Y en Profit Factor."""
    best = "A"
    for k in ("B", "C"):
        if res[k]["trades"] and res[k]["net"] > res[best]["net"] and res[k]["pf"] >= res["A"]["pf"] \
                and res[k]["net"] > res["A"]["net"]:
            best = k
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_ohlcv.csv")
    ap.add_argument("--rebase", action="store_true", help="fuerza reescritura del benchmark (dataset nuevo)")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    try:
        with open(os.path.splitext(a.csv)[0] + ".meta.json") as fh:
            meta = json.load(fh)
    except OSError:
        meta = {"rows": len(df)}
    f = build_features(df)
    res = {k: metrics(simulate(f, generate_signals(f, k))) for k in MODELS}
    win = pick_winner(res)

    hdr = (f"{'Modelo':<24}|{'Trades':>7} |{'Win Rate %':>11} |{'Gross PnL ($)':>14} |{'Fees ($)':>9} |"
           f"{'Net PnL ($ USD)':>16} |{'Profit Factor':>14} |{'Expectancy ($/trade)':>21} |{'P-Value':>8}")
    print(f"TORNEO ABLACION RSI | {meta.get('market', '?')} {meta.get('symbol', '?')} {meta.get('interval', '?')} "
          f"| {len(df)} velas | {meta.get('first', '?')} -> {meta.get('last', '?')}")
    print(f"Capital ${CFG['capital']:.0f} | riesgo {CFG['risk_pct']:.1%}/trade | lev max {CFG['max_lev']:.0f}x "
          f"| maker {CFG['maker']:.2%} | stop taker {CFG['taker']:.2%} + slip {CFG['slip']:.2%}")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k, name in MODELS.items():
        r = res[k]
        print(f"{k + ' ' + name:<24}|{r['trades']:>7} |{r['win_rate']:>11.2f} |{r['gross']:>+14.2f} |{r['fees']:>9.2f} |"
              f"{r['net']:>+16.2f} |{r['pf']:>14.3f} |{r['expectancy']:>+21.4f} |{r['p_value']:>8.4f}")
    print("-" * len(hdr))
    w = res[win]
    edge = w["net"] > 0 and w["p_value"] < 0.05
    print(f"GANADOR: Modelo {win} ({MODELS[win]}) | Net ${w['net']:+.2f} | "
          f"{'EDGE SIGNIFICATIVO' if edge else 'SIN EDGE ESTADISTICO (p >= 0.05 o Net <= 0): no operar en real'}")

    prev = None
    if os.path.exists(BENCH):
        with open(BENCH) as fh:
            prev = json.load(fh)
    if prev is None or a.rebase or w["net"] > prev["metrics"][prev["winner"]]["net"]:
        with open(BENCH, "w") as fh:
            json.dump({"winner": win, "winner_name": MODELS[win], "edge_significant": edge, "dataset": meta,
                       "metrics": res, "strategy_params": PARAMS,
                       "sim_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
                       "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, fh, indent=1)
        print(f"RATCHET: best_benchmark.json actualizado -> modelo por defecto = {win}")
    else:
        pw = prev["winner"]
        print(f"RATCHET: sin mejora (guardado {pw} ${prev['metrics'][pw]['net']:+.2f} >= ${w['net']:+.2f}); "
              f"benchmark y modelo por defecto intactos")


if __name__ == "__main__":
    main()
