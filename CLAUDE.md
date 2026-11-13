# Botsin — Motor autónomo de scalping FRVP (POC / VAH / VAL)

Corre 100% en nube/navegador. Sin hardware local, sin TradingView, sin API keys.

## [POLÍTICA ESTRICTA DE AHORRO DE TOKENS]
1. **LECTURA QUIRÚRGICA:** Prohibido leer archivos completos >80 líneas; usa `grep -n` o `sed -n 'X,Yp'`.
2. **CERO ECO DE CÓDIGO:** Prohibido imprimir el código de los archivos en el chat. Solo confirma ediciones y muestra la tabla final.
3. **EJECUCIÓN SILENCIOSA:** Corre descargas/backtests en background con logs a archivo y muestra ÚNICAMENTE las últimas 16 líneas (`tail -n 16`).
4. **PROTOCOLO RATCHET ANTI-RETROCESO:** El mejor resultado vive en `best_benchmark.json`. Prohibido dejar activa cualquier regla o filtro (p. ej. RSI) si reduce el `Net PnL ($ USD)` o el `Profit Factor` frente al modelo sin ese filtro. El benchmark solo se sobreescribe si el Net PnL mejora (`--rebase` únicamente al cambiar de dataset).
5. **AUTO-REPORTE:** Al final de cada respuesta, 1 línea con archivos tocados y estimado de tokens.

## Agentes internos
- **Agente 1 — Datos & Simulador (`src/data_fetcher.py`, `src/futures_simulator.py`):** solo `numpy`, `pandas`, `requests`/`aiohttp`. Máx. 5,000 velas.
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

## PROHIBIDO: look-ahead bias
- POC/VAH/VAL de la vela `i` usan **solo** velas cerradas `≤ i-1` y pivotes confirmados `≤ i-1`.
- La señal se evalúa al cierre de `i`; la orden limit al `close[i]` solo vive durante `i+1` y se llena si el precio la atraviesa.
- Dentro de una vela el stop se evalúa **antes** que los TP (peor caso); sin TP en la vela de fill.
- La vela aún abierta se descarta en el fetcher.
- Todo cambio debe pasar `python -m src.frvp_engine --selftest` y `python -m src.frvp_strategy --selftest`.

## Comandos
```bash
python -m src.data_fetcher --symbol SOLUSDT --interval 5m --limit 5000 > logs/fetch.log 2>&1
python -m src.ablation_runner > logs/ablation.log 2>&1; tail -n 16 logs/ablation.log
```
