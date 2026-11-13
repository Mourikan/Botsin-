# AUDIT_REPORT — Auditoría forense FRVP (A2, R1, R2, R3 × 3 datasets)

Muestra: 252 trades | Net total $-661.26 | Gross $-7.22 | Fees $654.04 (24% del |gross|)

## 1. Asimetría de pagos y efecto TP1/BE
```
        model       trades      win_pct      avg_win     avg_loss       payoff        be_wr          net
           A2          146        45.89        11.41       -12.75         0.90        52.76      -242.25
           R1            7        57.14         2.41       -12.85         0.19        84.22       -28.92
           R2           48        39.58         7.10       -10.53         0.67        59.73      -170.39
           R3           51        39.22        10.14       -13.63         0.74        57.34      -219.70

         kind            n          pct      avg_net      sum_net        avg_R
   SL directo          136        53.97       -13.00     -1768.11         9.95
    TP1 -> BE           62        24.60         4.60       285.26         9.98
TP2 -> stop POC           27        10.71        13.69       369.56         9.85
 TP3 completo           26        10.32        17.35       451.17        10.00
  Timeout/fin            1         0.40         0.85         0.85        10.00
```
Trades que tocan TP1: 115 (46%). Net real $+1106.00 vs cerrar 100% en TP1 $+1212.84. Net total con 100% en TP1: $-554.42.

## 2. MFE / MAE (unidades ATR)
```
metrica                            p25     p50     p75     p90
stop_dist_ATR (todos)             1.09    1.47    1.80    2.03
MFE_ATR perdedores SL             0.12    0.42    0.97    1.56
MFE_R perdedores SL               0.08    0.30    0.76    1.07
MAE_ATR ganadores                 0.29    0.62    0.92    1.54
stop_ATR necesario->TP1 (SL)      1.61    2.25    3.02    4.81
```
SL directo: 136 | cazados (precio alcanza TP1 en ≤24 velas tras el stop): 47% | alcanzan TP1 en ≤48 velas desde la entrada: 54% | perdedores SL con MFE ≥ 0.5R antes de caer: 39%

## 3. Selección adversa del limit y estabilidad del anclaje
```
       filled            n    fwd12_ATR   to_poc_pct
        False           11         0.04        90.91
         True          354         0.02        76.27
```
Fill rate de señales elegibles: 97%. fwd12 = retorno a favor 12 velas después de la señal (ATR); to_poc = señal alcanza el POC en ≤24 velas.
Anclaje: desplazamiento del POC a favor del trade relativo a la distancia entrada→POC: mediana +6v -0.00, +12v -0.06; trades cuyo POC retrocede >50% del objetivo en 12 velas: 34%; cambia >50% en cualquier sentido: 46%.

## 4. Anatomía de pérdidas
```
Pérdida bruta total de perdedores: $1773.30 en 142 trades
(a) ruptura violenta sin freno (SL directo, MFE<0.25ATR, sigue ≥1ATR tras el stop):  19.0%  (34 trades)
(b) giros en falso / ruido dentro del rango:                                   56.6%  (108 trades)
(c) fricción comisiones + slippage:                                             24.5%
Stop mediano 0.355% del precio -> fees medianas = 0.21R por trade; stops < 0.25%: 33% de trades, fee mediana 0.48R
```

5 peores clusters de pérdidas consecutivas:
```
R3  SOL reciente (IS)         8 trades   -114.63  2026-10-31 01:00 -> 2026-11-04 03:15
R3  BTC reciente              5 trades    -77.84  2026-11-03 12:40 -> 2026-11-04 15:25
A2  BTC reciente              5 trades    -68.89  2026-11-03 09:30 -> 2026-11-04 00:45
A2  SOL reciente (IS)         5 trades    -60.87  2026-11-08 16:40 -> 2026-11-09 16:30
A2  BTC reciente              4 trades    -60.03  2026-11-02 12:55 -> 2026-11-03 05:40
```

## 5. Dictamen
```
=== DICTAMEN Y SOLUCIÓN PARA GEMINI ===
FALLA #1: el gatillo no tiene ventaja direccional. 252 trades: Gross $-7.22 (PF bruto 0.99);
retorno medio 12 velas tras la señal = +0.02 ATR. Las fees ($654) son el 100% de la pérdida neta.
No es un problema de escalera: cerrar 100% en TP1 solo mueve el Net $+107. Payoff real < 1 con WR < WR de breakeven en los 4 modelos.
FALLA #2 (stop dentro del ruido): 47% de los SL directos alcanzan TP1 en <= 24 velas
después del stop; 39% iban +0.5R antes de caer. Stop actual p50 1.47 ATR vs 2.25 ATR necesarios. 'Ruido' = 57% de la pérdida.
FALLA #3 (fricción): fees = 24% de la pérdida; con stop < 0.25% (83 trades) cuestan 0.48R/trade.
FALLA #4 (anclaje que repinta): en 34% de trades el POC retrocede > 50% del objetivo en 12 velas.
NO es selección adversa del limit: fill 97%, solo 11 señales sin llenar.
CAMBIOS PROPUESTOS (hipótesis derivadas de los datos, NO validadas; nada aquí garantiza PF > 1.35):
1) Anclaje fijo sin repintado: VAL/POC/VAH de la sesión UTC ANTERIOR (o semana anterior), congelados todo
   el día. Ataca la falla #4 y da niveles que el mercado entero ve, en vez de un swing de 12 velas.
2) Entrada tras el barrido, no en la mecha: esperar que el precio pierda el low de la vela de señal (donde
   se cazan los stops) y recupere el VAL al cierre; SL bajo ese segundo extremo. Ataca la falla #2.
3) Suelo de distancia de stop: >= 0.35% del precio (fees <= ~0.25R). Si la estructura da menos, no hay trade.
4) Complemento cuantitativo: CVD acumulado desde la apertura de sesión con divergencia en el barrido
   (precio hace nuevo mínimo, CVD no). Ya tenemos taker_buy_volume; es la medida directa de absorción.
5) Validar el gatillo ANTES del simulador: event study sin SL/TP (retorno a 6/12/24 velas en ATR). Solo
   pasa al simulador si el retorno medio supera ~0.3 ATR (coste de fees) con p < 0.05.
6) Muestra: >= 6 meses por activo (el tope de 5,000 velas = 17 días) + holdout intocable. Los 3 datasets
   actuales ya están contaminados. Nota: A2/R2/R3 comparten señales; los 252 trades no son independientes.
```
