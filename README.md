# Polymarket Arbitrage Scanner

Herramienta de deteccion de arbitraje en mercados de prediccion de [Polymarket](https://polymarket.com).

Basado en el paper: **"Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"** (Saguillo, Ghafouri, Kiffer, Suarez-Tangil, 2025) - [arXiv:2508.03474](https://arxiv.org/abs/2508.03474)

## Tipos de Arbitraje Detectados

### 1. Single Condition Arbitrage (Section 6.1)
Cuando los precios YES + NO de una condicion no suman $1:
- **YES + NO < $1**: Comprar ambos -> beneficio garantizado al resolver
- **YES + NO > $1**: Split y vender ambos -> beneficio instantaneo

### 2. Market Rebalancing Arbitrage (Definition 3, Section 6.2)
En mercados NegRisk (multiples condiciones), cuando la suma de todos los precios YES != $1:
- **Sum(YES) < $1 (Long)**: Comprar YES de cada condicion -> beneficio = 1 - Sum
- **Sum(YES) > $1 (Short)**: Comprar NO de cada condicion -> beneficio = Sum - 1

### 3. Combinatorial Arbitrage (Definition 4, Section 6.3)
Entre mercados dependientes (mismo evento, misma fecha):
- Deteccion de dependencias logicas entre mercados usando LLM (Section 5)
- Pre-filtrado por topic con embeddings (Section 4.1.1)
- Beneficio = diferencia de precios entre condiciones dependientes

## Modos de Ejecucion

| Modo | Comando | Descripcion |
|------|---------|-------------|
| Heuristico | `python main.py` | Rapido, usa similitud de texto |
| LLM | `python main.py --llm` | Preciso, usa LLM para detectar dependencias logicas |
| Spark | `python main.py --spark` | Escala, procesamiento paralelo con PySpark |
| Full | `python main.py --llm --spark` | LLM + Spark combinados |

## Instalacion

```bash
git clone https://github.com/Ibonarambarri/polymarket-arbitrage-bot.git
cd polymarket-arbitrage-bot

python -m venv venv
source venv/bin/activate

# Instalar todo
pip install -r requirements.txt

# O solo lo basico (sin LLM ni Spark)
pip install requests
```

## Uso

```bash
# Escaneo basico (solo requiere requests)
python main.py

# Con precios en vivo del CLOB
python main.py --refresh

# Con LLM (requiere openai + sentence-transformers)
python main.py --llm

# Con LLM usando Ollama local
python main.py --llm --llm-api-base http://localhost:11434/v1 --llm-model deepseek-r1:latest

# Con LLM usando OpenAI
LLM_API_KEY=sk-... python main.py --llm --llm-api-base https://api.openai.com/v1 --llm-model gpt-4

# Con LLM usando DeepSeek API
LLM_API_KEY=sk-... python main.py --llm --llm-api-base https://api.deepseek.com/v1 --llm-model deepseek-chat

# Con PySpark (requiere pyspark)
python main.py --spark

# Modo completo
python main.py --llm --spark --refresh --min-margin 0.05 -v
```

## Arquitectura

```
polymarket_bot/
├── main.py            # Entry point - CLI con flags --llm, --spark, --refresh
├── config.py          # Configuracion (API, LLM, embeddings, Spark)
├── models.py          # Modelos de datos (Market, Condition, Token, ArbitrageOpportunity)
├── fetcher.py         # Cliente API (Gamma API + CLOB API)
├── arbitrage.py       # Motor de deteccion (3 tipos de arbitraje)
├── llm_detector.py    # Deteccion de dependencias con LLM (paper Section 5)
├── llm_prompts.py     # Templates de prompts (paper Appendix B)
├── embeddings.py      # Clasificacion por topic con embeddings (Section 4.1.1)
├── spark_analyzer.py  # Analisis paralelo con PySpark
└── requirements.txt
```

### Flujo de Deteccion Combinatorial (con --llm)

```
Markets -> [Embeddings: clasificar por topic]
        -> [Agrupar por (topic, end_date)]
        -> [Filtrar pares por similitud semantica]
        -> [LLM: detectar dependencias logicas]
        -> [Validar output LLM: JSON valido, 1 True por mercado]
        -> [Calcular arbitraje en pares dependientes]
```

## APIs Utilizadas

| API | URL | Uso |
|-----|-----|-----|
| **Gamma API** | `gamma-api.polymarket.com` | Descubrimiento de mercados y metadatos |
| **CLOB API** | `clob.polymarket.com` | Precios en vivo, order books |

No se requiere autenticacion para lectura de datos.

## Configuracion

Variables de entorno (o editar `config.py`):

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `LLM_API_BASE_URL` | `http://localhost:11434/v1` | URL de la API del LLM |
| `LLM_API_KEY` | `` | API key (no necesaria para Ollama) |
| `LLM_MODEL` | `deepseek-r1:latest` | Modelo a usar |

Parametros en `config.py`:

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `MIN_PROFIT_MARGIN` | 0.02 | Margen minimo para reportar oportunidad |
| `MAX_CONDITION_PRICE` | 0.95 | Precio maximo para considerar condicion "incierta" |
| `MAX_POSITION_SIZE` | 10000 | Tamano maximo de posicion para estimar beneficio |
| `EMBEDDING_SIMILARITY_THRESHOLD` | 0.7 | Similitud minima para comprobar par con LLM |
| `SPARK_MIN_MARKETS` | 100 | Minimo de mercados para justificar Spark |

## Referencia

```bibtex
@article{saguillo2025unravelling,
  title={Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets},
  author={Saguillo, Oriol and Ghafouri, Vahid and Kiffer, Lucianna and Suarez-Tangil, Guillermo},
  journal={arXiv preprint arXiv:2508.03474},
  year={2025}
}
```

## Disclaimer

Esta herramienta es solo para fines educativos y de investigacion. El trading en mercados de prediccion conlleva riesgo financiero. El arbitraje en Polymarket es **no-atomico** (las ordenes se ejecutan por separado), por lo que existe riesgo de ejecucion parcial.
