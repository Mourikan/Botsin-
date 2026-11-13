# Botsin — Motor autónomo de scalping FRVP (POC / VAH / VAL)

Corre 100% en nube/navegador. Sin hardware local, sin TradingView, sin API keys.

## [POLÍTICA ESTRICTA DE AHORRO DE TOKENS]
1. **LECTURA QUIRÚRGICA:** Prohibido leer archivos completos >80 líneas; usa `grep -n` o `sed -n 'X,Yp'`.
2. **CERO ECO DE CÓDIGO:** Prohibido imprimir el código de los archivos en el chat. Solo confirma ediciones y muestra la tabla final.
3. **EJECUCIÓN SILENCIOSA:** Corre descargas/backtests en background con logs a archivo y muestra ÚNICAMENTE las últimas 16 líneas (`tail -n 16`).
4. **PROTOCOLO RATCHET ANTI-RETROCESO:** El mejor resultado vive en `best_benchmark.json`. Prohibido dejar activa cualquier regla o filtro (p. ej. RSI) si reduce el `Net PnL ($ USD)` o el `Profit Factor` frente al modelo sin ese filtro. El benchmark solo se sobreescribe si el Net PnL mejora (`--rebase` únicamente al cambiar de dataset).
5. **AUTO-REPORTE:** Al final de cada respuesta, 1 línea con archivos tocados y estimado de tokens.

## Agentes internos
- **Agente 1 — Datos & Simulador (`src/data_fetcher.py`, `src/futures_simulator.py`):** solo `numpy`, `pandas`, `requests`/`aiohttp`. Máx. 5,000 velas en REST/diario; el histórico mensual (`--months`, Fase 5) es la excepción para muestras largas.
  - Datos: Binance Futures USDT-M (`fapi/v1/klines`) con `taker_buy_volume`, `taker_sell_volume`, `volume_delta`. Si `fapi` devuelve 451 (geobloqueo del servidor cloud) se usa el archivo público oficial `data.binance.vision` (mismas velas, ~1 día de retraso). No se evade el geobloqueo.
  - Simulador: capital $1,000, riesgo fijo 1% ($10) por trade, tope 10x. Entrada y TP = Maker 0.02%; stops y timeout = Taker 0.05% + 0.02% slippage.
  - Filtro pre-trade: no operar si `(VAH−VAL)/POC < 0.35%` ni si la vela de entrada pisa ±3 min de funding (00/08/16 UTC).
- **Agente 2 — Estrategia (`src/frvp_engine.py`, `src/frvp_strategy.py`):** FRVP de 60 bins anclado al último swing estructural (pivotes confirmados), VA 70%, ATR(14).
  - Entrada: mecha toca VAL/VAH y cierra dentro del área de valor, con (`RVOL ≥ 1.2` o mecha ≥ 35%) y `volume_delta` a favor.
  - SL: extremo del barrido ± `0.5·ATR`. TP1 40% (90% del camino al POC, mín. 1.2R) → SL a BE neto (±0.08%); TP2 40% (extremo opuesto del VA) → SL al POC; TP3 20% (swing externo).
- **Modelos (`MODELS` en `src/frvp_strategy.py`; el activo es `winner` de `best_benchmark.json`):**
  - `A` base Fase 1 (pivotes 12v, perfil 36–288 velas, amplitud 0.35%, timeout 144 velas). `B`/`C` = A + filtros RSI (descartados).
  - `A1` anclaje amplio: pivotes 24v, perfil 72–500 velas, amplitud 0.45%, sin timeout (solo SL o escalera TP).
  - `A2` = A + anomalía de volumen: `RVOL ≥ 1.7` o volumen taker del lado del trade `> 1.5x` su media de 20 velas (sustituye a RVOL 1.2/mecha).
  - `A3` = A1 + anomalía de volumen.
  - Cada filtro declara su `parent`; solo es elegible si supera al padre en Net PnL sin bajar el Profit Factor.
- **Agente 3 — Auditor (`src/probability_auditor.py`, `src/ablation_runner.py`):** Esperanza neta, Profit Factor, test binomial (`p < 0.05`) y torneo A/B/C. Ninguna estrategia se opera en real sin `p < 0.05` y Net > 0.

## ESTADO (Fase 3, 2026-11-13): PREMISA BASE NO VALIDADA — REDISEÑO OBLIGATORIO
- **OOS fallido:** A y A2 pierden en los 4 backtests fuera de muestra (BTCUSDT reciente y SOLUSDT 2024-03 en tendencia). Net OOS total: A −$516.93, A2 −$283.20; todos los p-value > 0.85.
- La premisa "FRVP anclado a pivotes de 12 velas en 5m + reversión por absorción en VAL/VAH" **no tiene edge**. El +$38/+$41 in-sample era ruido.
- **Prohibido** añadir más filtros o tunear parámetros sobre esta estructura. Antes de continuar hay que rediseñar el anclaje (p. ej. VWAP anclado, bloques de liquidez, perfil de sesión) y/o añadir filtro de régimen (no revertir contra tendencia).
- Aislamiento de A1 (in-sample): el daño vino de los pivotes 24v (Net −$218.86). Amplitud 0.45% (+$64.66) y sin timeout (+$78.51) mejoraban a A por separado, pero no están validados OOS.
- **Ratchet:** la métrica reina es Net PnL OOS. Todo candidato nuevo pasa por `python -m src.oos_validator` y solo se opera con Net > 0 y p < 0.05 en todos los datasets OOS. `best_benchmark.json` marca `oos_validated: false`.
- Datasets OOS: `python -m src.data_fetcher --symbol BTCUSDT --market futures-archive --out data/oos_btc_recent.csv` y `python -m src.data_fetcher --symbol SOLUSDT --end 2024-03-18 --out data/oos_sol_2024_03.csv`.

## ESTADO (Fase 4, 2026-11-13): FILTRO DE RÉGIMEN VWAP NO RESCATA LA ESTRUCTURA
- `R0` (llamado R1 en su momento) = A2 + amplitud 0.45% + sin timeout + régimen Rolling VWAP 24h (288 velas; solo longs con `close > VWAP`, solo shorts con `close < VWAP`).
- **Falla el test del ácido:** SOL 2024-03: 15 trades, WR 20%, Gross −$100.75, Net −$122.71. BTC reciente: 4 trades, WR 0%, Gross −$40.00, Net −$53.41. `oos_validated` sigue en `false`.
- R0 pierde menos que A2 en total solo porque opera menos; por trade pierde más (SOL −$8.18 vs −$4.48). No es operable.
- Conclusión: alinear la reversión con el VWAP no arregla el gatillo "mecha en VAL/VAH del perfil de swing de 12 velas". El problema está en la entrada/estructura, no solo en la dirección.
- **Los datasets OOS ya están contaminados** (se han usado para diseñar en Fases 3 y 4). El próximo candidato necesita un holdout nuevo que no se mire hasta el final.

## ESTADO (Fase 4b, 2026-11-13): ÁRBITRO DE RÉGIMEN — NINGÚN MODELO GANA DINERO
- **Base permanente** (`BASE_SIM` en `src/frvp_strategy.py`): pivotes 12v, amplitud ≥ 0.45%, sin timeout, TP1 en el POC. Pivotes 24v descartados.
- `src/regime_arbiter.py`: sesgo (VWAP_288 + migración del POC a 24 velas, ±0.05%), `ER_48`, Gatillo 1 (absorción: `RVOL ≥ 1.4` + mecha ≥ 40% + delta de absorción o de giro) y Gatillo 2 (ruptura con `RVOL ≥ 1.5` en las 3 velas previas + retest, `ER_48 ≥ 0.28`; SL 0.8·ATR, TP 1.3R/2.5R/4R).
- `R1` = sesgo + Gatillo 1 · `R2` = Gatillo 1 sin sesgo · `R3` = sesgo + Gatillo 1 + Gatillo 2.
- **Torneo (SOL reciente IS + SOL 2024-03 + BTC reciente), Net consolidado:** R1 −$28.92 (7 trades), R2 −$170.39 (48), R3 −$219.70 (51). Los 9 backtests tienen Net negativo; `oos_validated: false`.
- `winner = R1` solo porque casi no opera (7 trades en 15,000 velas): **no es un edge ni es operable**.
- El sesgo sí ayuda al Gatillo 1 en tendencia (SOL 2024-03: R1 WR 80% y Gross +$3 vs R2 −$15), pero con 5 trades no es evidencia. El Gatillo 2 resta en SOL y sus fees triplican el bruto en BTC (stop de 0.8·ATR ⇒ nocional alto).
- Los 3 datasets están contaminados por diseño iterativo. Próximo candidato: holdout nuevo, intocable hasta el final, y más historia (el tope de 5,000 velas deja muestras de 1–20 trades).
- Torneo: `python -m src.oos_validator > logs/tournament.log 2>&1; tail -n 20 logs/tournament.log`. Con el árbitro hay que usar `prepare(f, modelo)` (no `generate_signals`) antes de `simulate`.

## ESTADO (Fase 5, 2026-11-13): SESIÓN FIJA + BARRIDO 2 TIEMPOS — NO PASA EL EVENT STUDY, HOLDOUT SELLADO
- Datos: 6 meses 5m (2026-05-01 → 2026-10-31, 52,992 velas) SOL y BTC en `data/*_5m_6m.csv.gz` (`python -m src.data_fetcher --symbol SOLUSDT --months 2026-05:2026-10 --out data/solusdt_5m_6m.csv.gz`).
- Split pre-registrado (`src/event_study_and_backtest.py`): Desarrollo = primer 70% de días (hasta 2026-09-06); Holdout = resto hasta 2026-10-24 (los días ≥ 2026-10-25 se excluyen: contaminados por Fases 1–4).
- `src/session_frvp.py`: POC/VAH/VAL del día UTC anterior congelados todo el día, POC de 5 días, CVD de sesión. `session_signals` (S1/S2/S3) en `src/frvp_strategy.py`.
- **Event Study SOL Desarrollo:** S1 n=30, +12v +0.377 ATR, p=0.315 · S2 n=17, +0.416, p=0.373 · S3 n=5, +0.166, p=0.793. Regla de paso (≥ 0.30 ATR y p < 0.05): **ninguna pasa**. BTC no confirma (S1 +0.115, p=0.82).
- Diagnóstico IS (no validado): S1 TP1_100% Net +$36.24, PF 1.19 (28 trades); la escalera empata. S2/S3 pierden.
- **El holdout no se ha ejecutado.** Prohibido bajar el umbral o tunear mirando estos números para "hacer pasar" S1.
- Problema real: tamaño de muestra. El suelo de stop 0.35% deja 30 de 83 señales en 128 días. Con la dispersión observada (~2 ATR) detectar +0.38 ATR con p < 0.05 requiere ~450 señales: pool de varios perpetuos líquidos y/o ≥ 2 años.
- `futures_simulator`: corregido que una posición cerrada al 100% por los TP siguiera bloqueando nuevas entradas (no afecta a modelos anteriores).

## PROHIBIDO: look-ahead bias
- POC/VAH/VAL de la vela `i` usan **solo** velas cerradas `≤ i-1` y pivotes confirmados `≤ i-1`.
- La señal se evalúa al cierre de `i`; la orden limit al `close[i]` solo vive durante `i+1` y se llena si el precio la atraviesa.
- Dentro de una vela el stop se evalúa **antes** que los TP (peor caso); sin TP en la vela de fill.
- La vela aún abierta se descarta en el fetcher.
- Todo cambio debe pasar `python -m src.frvp_engine --selftest`, `python -m src.frvp_strategy --selftest` y `python -m src.session_frvp --selftest`.

## Comandos
```bash
python -m src.data_fetcher --symbol SOLUSDT --interval 5m --limit 5000 > logs/fetch.log 2>&1
python -m src.ablation_runner > logs/ablation.log 2>&1; tail -n 16 logs/ablation.log
```
