# Polymarket Arbitrage Scanner

Herramienta de deteccion de arbitraje en mercados de prediccion de [Polymarket](https://polymarket.com).

Basado en el paper: **"Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"** (Saguillo, Ghafouri, Kiffer, Suarez-Tangil, 2025).

## Tipos de Arbitraje Detectados

### 1. Single Condition Arbitrage
Cuando los precios YES + NO de una condicion no suman $1:
- **YES + NO < $1**: Comprar ambos -> beneficio garantizado al resolver
- **YES + NO > $1**: Split y vender ambos -> beneficio instantaneo

### 2. Market Rebalancing Arbitrage (Definicion 3 del paper)
En mercados NegRisk (multiples condiciones), cuando la suma de todos los precios YES != $1:
- **Sum(YES) < $1 (Long)**: Comprar YES de cada condicion -> beneficio = 1 - Sum
- **Sum(YES) > $1 (Short)**: Comprar NO de cada condicion -> beneficio = Sum - 1

### 3. Combinatorial Arbitrage (Definicion 4 del paper)
Entre mercados dependientes (mismo evento, misma fecha):
- Mercados con condiciones semanticamente relacionadas con precios inconsistentes
- Beneficio = diferencia de precios entre condiciones dependientes

## Instalacion

```bash
git clone https://github.com/TU_USUARIO/polymarket-arbitrage-bot.git
cd polymarket-arbitrage-bot

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Escaneo basico - usa precios de la Gamma API
python main.py

# Escaneo con precios en vivo del CLOB (mas preciso, mas lento)
python main.py --refresh

# Configurar margen minimo de beneficio (default: 0.02 = 2 centavos/$)
python main.py --min-margin 0.05

# Modo verbose para debugging
python main.py -v
```

## Arquitectura

```
polymarket_bot/
├── main.py          # Entry point - orquesta el escaneo y muestra resultados
├── config.py        # Configuracion (URLs, umbrales, parametros)
├── models.py        # Modelos de datos (Market, Condition, ArbitrageOpportunity)
├── fetcher.py       # Cliente API (Gamma API + CLOB API)
├── arbitrage.py     # Motor de deteccion de arbitraje
├── requirements.txt
└── README.md
```

## APIs Utilizadas

| API | URL | Uso |
|-----|-----|-----|
| **Gamma API** | `gamma-api.polymarket.com` | Descubrimiento de mercados y metadatos |
| **CLOB API** | `clob.polymarket.com` | Precios en vivo, order books |

No se requiere autenticacion para lectura de datos.

## Parametros Configurables

En `config.py`:

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `MIN_PROFIT_MARGIN` | 0.02 | Margen minimo para reportar oportunidad |
| `MAX_CONDITION_PRICE` | 0.95 | Precio maximo para considerar una condicion "incierta" |
| `MIN_DISPLAY_PROFIT_USD` | 1.0 | Beneficio minimo en USD para mostrar |

## Referencia

```
@article{saguillo2025unravelling,
  title={Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets},
  author={Saguillo, Oriol and Ghafouri, Vahid and Kiffer, Lucianna and Suarez-Tangil, Guillermo},
  journal={arXiv preprint arXiv:2508.03474},
  year={2025}
}
```

## Disclaimer

Esta herramienta es solo para fines educativos y de investigacion. El trading en mercados de prediccion conlleva riesgo financiero. El arbitraje en Polymarket es **no-atomico** (las ordenes se ejecutan por separado), por lo que existe riesgo de ejecucion parcial.
