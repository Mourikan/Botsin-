# Botsin — Motor autónomo de scalping FRVP (POC / VAH / VAL)

Corre 100% en nube/navegador. Sin hardware local, sin TradingView, sin API keys.

## [POLÍTICA ESTRICTA DE AHORRO DE TOKENS]
1. **LECTURA QUIRÚRGICA:** Prohibido leer archivos completos >80 líneas; usa `grep -n` o `sed -n 'X,Yp'`.
2. **CERO ECO DE CÓDIGO:** Prohibido imprimir bloques de código completos en el chat tras crearlos o editarlos. Solo confirma rutas y diffs mínimos.
3. **EJECUCIÓN SILENCIOSA:** Ejecuta descargas o backtests redirigiendo logs a archivo e imprime en consola SOLO las últimas 12 líneas (`tail -n 12`).
4. **AUTO-REPORTE:** Al final de cada respuesta, imprime 1 línea con archivos tocados y estimado de tokens consumidos.

## Agentes internos
- **Agente 1 — Python Cloud/Low-Spec:** estructura modular ligera. Dependencias permitidas: `numpy`, `pandas`, `requests`/`aiohttp`. Nada de scipy, TA-Lib ni frameworks pesados. Máx. 5,000 velas en memoria.
- **Agente 2 — Quant Volume Profile:** `src/frvp_engine.py`. Fixed Range Volume Profile sobre ventana deslizante; extrae `POC`, `VAH`/`VAL` (Value Area 70%) y `RVOL`.
- **Agente 3 — Probabilidad Matemática:** `src/probability_auditor.py`. Evalúa cada señal con Esperanza Neta en USD
  `E = (P_win · AvgWin_net) − (P_loss · AvgLoss_net)`, descontando **0.05% Taker + 0.02% slippage por lado** (0.14% ida y vuelta),
  Profit Factor y test binomial (`p-value < 0.05`). Ninguna estrategia se promueve sin pasar el auditor.

## PROHIBIDO: look-ahead bias
- POC/VAH/VAL de la vela `i` se calculan **solo** con las `N` velas cerradas previas `[i-N, i-1]`. La vela `i` nunca entra en su propio perfil.
- La señal se evalúa al cierre de `i`; la entrada se ejecuta al **open de `i+1`**.
- La última vela devuelta por Binance (aún abierta) se descarta en el fetcher.
- Si TP y SL caen en la misma vela, se asume **SL primero** (peor caso).
- Cualquier cambio en el motor debe pasar `python -m src.frvp_engine --selftest` (test de causalidad).

## Comandos
```bash
python -m src.data_fetcher --symbol BTCUSDT --interval 5m --limit 1500 > logs/fetch.log 2>&1
python -m src.probability_auditor > logs/audit.log 2>&1; tail -n 12 logs/audit.log
```
