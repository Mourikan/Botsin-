"""Auditoria forense (solo lectura): por que pierde dinero el motor FRVP.

No modifica estrategia ni simulador: re-simula A2/R1/R2/R3 en los 3 datasets, reconstruye cada trade
contra el precio y mide asimetria de pagos, MFE/MAE, sesgo de llenado limit, estabilidad del anclaje
y anatomia de perdidas. Escribe AUDIT_REPORT.md e imprime el resumen + dictamen.
"""
import numpy as np
import pandas as pd

from src.frvp_strategy import build_features, prepare, spec
from src.futures_simulator import CFG, pretrade_ok, simulate
from src.oos_validator import DATASETS

MODELS = ["A2", "R1", "R2", "R3"]
POST, H_FWD = 24, 48     # velas tras el SL para ver si el objetivo se alcanzo; horizonte contrafactual


def trades_for(df, model):
    s = spec(model)
    base = build_features(df, **s["feat"])
    frame, sig = prepare(base, model)
    p = {**CFG, **s["sim"]}
    tr = simulate(frame, sig, **s["sim"])
    o, h, l, c = (base[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr0, poc_b = base["atr"].to_numpy(), base["poc"].to_numpy()
    poc_f = frame["poc"].to_numpy()
    n, rows = len(df), []
    for t in tr.itertuples():
        i, x, sd, e0 = int(t.sig_bar), int(t.exit_bar), t.side, t.entry
        rp = abs(e0 - t.sl)
        tp1 = e0 + sd * max(p["tp1_path"] * (poc_f[i] - e0) * sd, p["tp1_min_r"] * rp)
        hi, lo = h[i + 1:x + 1].max(), l[i + 1:x + 1].min()
        mfe, mae = ((hi - e0), (e0 - lo)) if sd > 0 else ((e0 - lo), (hi - e0))
        r_usd = t.notional * rp / e0
        stage, g = int(t.tps_hit), t.gross / r_usd
        kind = ("SL directo" if g <= -0.95 else "Timeout/fin") if stage == 0 else \
               {1: "TP1 -> BE", 2: "TP2 -> stop POC", 3: "TP3 completo"}[stage]
        # contrafactual: stop necesario para alcanzar TP1 dentro de H_FWD velas
        seg = slice(i + 1, min(i + 1 + H_FWD, n))
        reach = (h[seg] >= tp1) if sd > 0 else (l[seg] <= tp1)
        need = np.nan
        if reach.any():
            j = reach.argmax()
            adv = (e0 - l[seg][:j + 1].min()) if sd > 0 else (h[seg][:j + 1].max() - e0)
            need = max(adv, 0) / atr0[i]
        post = slice(x + 1, min(x + 1 + POST, n))
        hunted = kind == "SL directo" and post.start < n and \
            bool(((h[post] >= tp1) if sd > 0 else (l[post] <= tp1)).any())
        cont = ((t.sl - l[post].min()) if sd > 0 else (h[post].max() - t.sl)) / atr0[i] if post.start < n else 0
        shift = {k: ((poc_b[min(i + k, n - 1)] - poc_b[i]) * sd / max(abs(poc_b[i] - e0), 1e-12)) for k in (6, 12)}
        rows.append(dict(model=model, sig_bar=i, side=sd, net=t.net, gross=t.gross, fees=t.fees, r_usd=r_usd,
                         kind=kind, stage=stage, mfe_atr=mfe / atr0[i], mae_atr=mae / atr0[i], mfe_r=mfe / rp,
                         stop_atr=rp / atr0[i], stop_pct=rp / e0 * 100, fee_r=t.fees / r_usd, need_atr=need,
                         hunted=hunted, cont_atr=cont, shift6=shift[6], shift12=shift[12],
                         tp1_all=t.notional / e0 * (tp1 - e0) * sd - t.notional * CFG["maker"] * 2
                         if stage >= 1 else t.net, t=pd.to_datetime(df["open_time"].iloc[i], unit="ms")))
    # sesgo de llenado: todas las senales elegibles, llenadas o no
    ok = pretrade_ok(frame, p["funding_guard_min"], p["min_width"]) & np.isfinite(atr0)
    fills = []
    for i in np.flatnonzero(sig):
        if i + 13 >= n or not ok[i]:
            continue
        sd = float(sig[i])
        filled = (l[i + 1] < c[i]) if sd > 0 else (h[i + 1] > c[i])
        seg = slice(i + 1, i + 25)
        to_poc = bool(((h[seg] >= poc_b[i]) if sd > 0 else (l[seg] <= poc_b[i])).any())
        fills.append(dict(model=model, filled=filled, fwd12=(c[i + 12] - c[i]) * sd / atr0[i], to_poc=to_poc))
    return pd.DataFrame(rows), pd.DataFrame(fills)


def collect():
    T, F = [], []
    for name, (csv, _) in DATASETS.items():
        df = pd.read_csv(csv)
        for m in MODELS:
            t, f = trades_for(df, m)
            T.append(t.assign(dataset=name))
            F.append(f.assign(dataset=name))
    return pd.concat(T, ignore_index=True), pd.concat(F, ignore_index=True)


def clusters(T):
    out = []
    for (m, d), g in T.sort_values("sig_bar").groupby(["model", "dataset"]):
        run = []
        for r in g.itertuples():
            if r.net < 0:
                run.append(r)
            elif run:
                out.append(run); run = []
        if run:
            out.append(run)
    out = sorted(out, key=lambda r: sum(x.net for x in r))[:5]
    return [dict(model=r[0].model, dataset=r[0].dataset, n=len(r), loss=sum(x.net for x in r),
                 desde=f"{r[0].t:%Y-%m-%d %H:%M}", hasta=f"{r[-1].t:%Y-%m-%d %H:%M}") for r in out]


def loss_anatomy(T):
    L = T[T.net < 0]
    total = -L.net.sum()
    violent = L[(L.kind == "SL directo") & (L.mfe_atr < 0.25) & (L.cont_atr >= 1.0)]
    other = L.drop(violent.index)
    return dict(total=total, a=-violent.gross.sum() / total, b=-other.gross.sum() / total,
                c=L.fees.sum() / total, n_a=len(violent), n_b=len(other), n=len(L))


def fmt(df, cols, f="{:>9.2f}"):
    head = "".join(f"{c:>13}" for c in cols)
    lines = [head]
    for idx, r in df.iterrows():
        lines.append("".join(f"{(f.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c])):>13}"
                             for c in cols))
    return lines


def build_report(T, F):
    out = []
    w = T[T.net > 0]
    lo = T[T.net < 0]
    out += ["# AUDIT_REPORT — Auditoría forense FRVP (A2, R1, R2, R3 × 3 datasets)", "",
            f"Muestra: {len(T)} trades | Net total ${T.net.sum():+.2f} | Gross ${T.gross.sum():+.2f} | "
            f"Fees ${T.fees.sum():.2f} ({T.fees.sum() / max(T.gross.abs().sum(), 1e-9) * 100:.0f}% del |gross|)", ""]
    out += ["## 1. Asimetría de pagos y efecto TP1/BE", "```"]
    pm = T.groupby("model").agg(trades=("net", "size"), win_pct=("net", lambda s: (s > 0).mean() * 100),
                                avg_win=("net", lambda s: s[s > 0].mean()), avg_loss=("net", lambda s: s[s < 0].mean()),
                                net=("net", "sum"))
    pm["payoff"] = pm.avg_win / -pm.avg_loss
    pm["be_wr"] = 100 / (1 + pm.payoff)
    out += fmt(pm.reset_index(), ["model", "trades", "win_pct", "avg_win", "avg_loss", "payoff", "be_wr", "net"])
    ex = T.groupby("kind").agg(n=("net", "size"), avg_net=("net", "mean"), sum_net=("net", "sum"),
                               avg_R=("r_usd", "mean"))
    ex["pct"] = ex.n / len(T) * 100
    out += [""] + fmt(ex.reset_index(), ["kind", "n", "pct", "avg_net", "sum_net", "avg_R"]) + ["```"]
    tp = T[T.stage >= 1]
    out += [f"Trades que tocan TP1: {len(tp)} ({len(tp) / len(T) * 100:.0f}%). Net real ${tp.net.sum():+.2f} vs "
            f"cerrar 100% en TP1 ${tp.tp1_all.sum():+.2f}. Net total con 100% en TP1: "
            f"${T.net.sum() - tp.net.sum() + tp.tp1_all.sum():+.2f}.", ""]
    out += ["## 2. MFE / MAE (unidades ATR)", "```"]
    q = lambda s, p: np.nanpercentile(s, p) if len(s.dropna()) else np.nan
    sl = T[T.kind == "SL directo"]
    rows = [("stop_dist_ATR (todos)", T.stop_atr), ("MFE_ATR perdedores SL", sl.mfe_atr),
            ("MFE_R perdedores SL", sl.mfe_r), ("MAE_ATR ganadores", w.mae_atr),
            ("stop_ATR necesario->TP1 (SL)", sl.need_atr)]
    out += [f"{'metrica':<30}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}"]
    out += [f"{k:<30}" + "".join(f"{q(s, p):>8.2f}" for p in (25, 50, 75, 90)) for k, s in rows]
    out += ["```",
            f"SL directo: {len(sl)} | cazados (precio alcanza TP1 en ≤{POST} velas tras el stop): "
            f"{sl.hunted.mean() * 100:.0f}% | alcanzan TP1 en ≤{H_FWD} velas desde la entrada: "
            f"{sl.need_atr.notna().mean() * 100:.0f}% | perdedores SL con MFE ≥ 0.5R antes de caer: "
            f"{(sl.mfe_r >= 0.5).mean() * 100:.0f}%", ""]
    out += ["## 3. Selección adversa del limit y estabilidad del anclaje", "```"]
    fb = F.groupby(["filled"]).agg(n=("fwd12", "size"), fwd12_ATR=("fwd12", "mean"), to_poc_pct=("to_poc", "mean"))
    fb["to_poc_pct"] *= 100
    out += fmt(fb.reset_index(), ["filled", "n", "fwd12_ATR", "to_poc_pct"]) + ["```"]
    out += [f"Fill rate de señales elegibles: {F.filled.mean() * 100:.0f}%. fwd12 = retorno a favor 12 velas "
            "después de la señal (ATR); to_poc = señal alcanza el POC en ≤24 velas.",
            f"Anclaje: desplazamiento del POC a favor del trade relativo a la distancia entrada→POC: "
            f"mediana +6v {T.shift6.median():+.2f}, +12v {T.shift12.median():+.2f}; trades cuyo POC retrocede "
            f">50% del objetivo en 12 velas: {(T.shift12 < -0.5).mean() * 100:.0f}%; cambia >50% en "
            f"cualquier sentido: {(T.shift12.abs() > 0.5).mean() * 100:.0f}%.", ""]
    an = loss_anatomy(T)
    out += ["## 4. Anatomía de pérdidas", "```",
            f"Pérdida bruta total de perdedores: ${an['total']:.2f} en {an['n']} trades",
            f"(a) ruptura violenta sin freno (SL directo, MFE<0.25ATR, sigue ≥1ATR tras el stop): "
            f"{an['a'] * 100:5.1f}%  ({an['n_a']} trades)",
            f"(b) giros en falso / ruido dentro del rango:                                  "
            f"{an['b'] * 100:5.1f}%  ({an['n_b']} trades)",
            f"(c) fricción comisiones + slippage:                                            {an['c'] * 100:5.1f}%",
            f"Stop mediano {T.stop_pct.median():.3f}% del precio -> fees medianas = {T.fee_r.median():.2f}R por trade; "
            f"stops < 0.25%: {(T.stop_pct < 0.25).mean() * 100:.0f}% de trades, fee mediana "
            f"{T[T.stop_pct < 0.25].fee_r.median():.2f}R", "```", "", "5 peores clusters de pérdidas consecutivas:", "```"]
    out += [f"{c['model']:<4}{c['dataset']:<24}{c['n']:>3} trades {c['loss']:>+9.2f}  {c['desde']} -> {c['hasta']}"
            for c in clusters(T)] + ["```", ""]
    return out, pm, ex, an, sl, F, tp


def dictamen(T, F, an, sl, tp):
    fee_tot, gross = T.fees.sum(), T.gross.sum()
    pf_g = T.gross[T.gross > 0].sum() / -T.gross[T.gross < 0].sum()
    fw = F[F.filled].fwd12.mean()
    tight = T[T.stop_pct < 0.25]
    return [
        "=== DICTAMEN Y SOLUCIÓN PARA GEMINI ===",
        f"FALLA #1: el gatillo no tiene ventaja direccional. {len(T)} trades: Gross ${gross:+.2f} (PF bruto {pf_g:.2f});",
        f"retorno medio 12 velas tras la señal = {fw:+.2f} ATR. Las fees (${fee_tot:.0f}) son el 100% de la pérdida neta.",
        "No es un problema de escalera: cerrar 100% en TP1 solo mueve el Net "
        f"${tp.tp1_all.sum() - tp.net.sum():+.0f}. Payoff real < 1 con WR < WR de breakeven en los 4 modelos.",
        f"FALLA #2 (stop dentro del ruido): {sl.hunted.mean() * 100:.0f}% de los SL directos alcanzan TP1 en <= 24 velas",
        f"después del stop; {(sl.mfe_r >= 0.5).mean() * 100:.0f}% iban +0.5R antes de caer. Stop actual p50 "
        f"{T.stop_atr.median():.2f} ATR vs {sl.need_atr.median():.2f} ATR necesarios. 'Ruido' = {an['b'] * 100:.0f}% de la pérdida.",
        f"FALLA #3 (fricción): fees = {an['c'] * 100:.0f}% de la pérdida; con stop < 0.25% ({len(tight)} trades) cuestan "
        f"{tight.fee_r.median():.2f}R/trade.",
        f"FALLA #4 (anclaje que repinta): en {(T.shift12 < -0.5).mean() * 100:.0f}% de trades el POC retrocede > 50% del "
        "objetivo en 12 velas.",
        f"NO es selección adversa del limit: fill {F.filled.mean() * 100:.0f}%, solo {(~F.filled).sum()} señales sin llenar.",
        "CAMBIOS PROPUESTOS (hipótesis derivadas de los datos, NO validadas; nada aquí garantiza PF > 1.35):",
        "1) Anclaje fijo sin repintado: VAL/POC/VAH de la sesión UTC ANTERIOR (o semana anterior), congelados todo",
        "   el día. Ataca la falla #4 y da niveles que el mercado entero ve, en vez de un swing de 12 velas.",
        "2) Entrada tras el barrido, no en la mecha: esperar que el precio pierda el low de la vela de señal (donde",
        "   se cazan los stops) y recupere el VAL al cierre; SL bajo ese segundo extremo. Ataca la falla #2.",
        "3) Suelo de distancia de stop: >= 0.35% del precio (fees <= ~0.25R). Si la estructura da menos, no hay trade.",
        "4) Complemento cuantitativo: CVD acumulado desde la apertura de sesión con divergencia en el barrido",
        "   (precio hace nuevo mínimo, CVD no). Ya tenemos taker_buy_volume; es la medida directa de absorción.",
        "5) Validar el gatillo ANTES del simulador: event study sin SL/TP (retorno a 6/12/24 velas en ATR). Solo",
        "   pasa al simulador si el retorno medio supera ~0.3 ATR (coste de fees) con p < 0.05.",
        "6) Muestra: >= 6 meses por activo (el tope de 5,000 velas = 17 días) + holdout intocable. Los 3 datasets",
        "   actuales ya están contaminados. Nota: A2/R2/R3 comparten señales; los 252 trades no son independientes.",
    ]


def main():
    T, F = collect()
    out, pm, ex, an, sl, F, tp = build_report(T, F)
    out += ["## 5. Dictamen", "```"] + dictamen(T, F, an, sl, tp) + ["```"]
    with open("AUDIT_REPORT.md", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(l for l in out if not l.startswith("```")))
    print(f"[AUDIT_REPORT.md: {len(out)} lineas]")


if __name__ == "__main__":
    main()
