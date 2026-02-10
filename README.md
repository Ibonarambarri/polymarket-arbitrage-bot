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

## Requisitos

- **Python 3.10+**
- **Ollama** (para LLM local, activado por defecto): [ollama.com](https://ollama.com)

## Instalacion

```bash
git clone https://github.com/Ibonarambarri/polymarket-arbitrage-bot.git
cd polymarket-arbitrage-bot

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Instalar y arrancar Ollama

```bash
# macOS
brew install ollama

# Arrancar el servidor (dejar abierto)
ollama serve
```

El modelo `deepseek-r1:latest` se descarga automaticamente en la primera ejecucion.

## Uso

```bash
# Modo por defecto: LLM local (Ollama + deepseek-r1) + Spark
python main.py

# Sin LLM, solo heuristico (mas rapido)
python main.py --no-llm

# Con otro modelo
python main.py --llm-model qwen2.5:3b

# Con precios en vivo del CLOB
python main.py --refresh

# Margen de beneficio personalizado
python main.py --min-margin 0.05

# Logging detallado
python main.py -v

# Con LLM usando OpenAI
LLM_API_KEY=sk-... python main.py --llm-api-base https://api.openai.com/v1 --llm-model gpt-4

# Con LLM usando DeepSeek API
LLM_API_KEY=sk-... python main.py --llm-api-base https://api.deepseek.com/v1 --llm-model deepseek-chat
```

### Flags

| Flag | Descripcion |
|------|-------------|
| `--no-llm` | Desactiva LLM, usa solo heuristico |
| `--llm` | Fuerza LLM (activado por defecto) |
| `--llm-model MODEL` | Modelo a usar (default: `deepseek-r1:latest`) |
| `--llm-api-base URL` | URL de la API LLM (default: Ollama local) |
| `--spark` | Fuerza PySpark (activado por defecto) |
| `--refresh` | Precios en vivo del CLOB API |
| `--min-margin N` | Margen minimo de beneficio (default: 0.02) |
| `-v` / `--verbose` | Logging DEBUG |
| `-q` / `--quiet` | Solo warnings/errors |

### Comportamiento automatico

- Si Ollama esta corriendo y el modelo esta disponible -> usa LLM
- Si el modelo no esta descargado -> lo descarga automaticamente (auto-pull)
- Si Ollama no esta corriendo -> fallback a modo heuristico con mensaje claro
- Si hay 100+ mercados y PySpark esta instalado -> procesamiento paralelo

## Arquitectura

```
polymarket-arbitrage-bot/
├── main.py            # Entry point, CLI, check de Ollama
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

### Flujo de Deteccion

```
Markets -> [Embeddings: clasificar por topic]
        -> [Agrupar por (topic, end_date)]
        -> [Filtrar pares por similitud semantica]
        -> [LLM: detectar dependencias logicas]
        -> [Validar output LLM: JSON valido, 1 True por mercado]
        -> [Calcular arbitraje en pares dependientes]
```

Sin LLM, el flujo usa similitud de texto (`SequenceMatcher`) y agrupa solo por fecha.

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
| `LLM_ENABLED` | `True` | LLM activado por defecto |
| `LLM_TIMEOUT` | `120` | Timeout para modelos locales (segundos) |
| `SPARK_ENABLED` | `True` | PySpark activado por defecto |
| `MIN_PROFIT_MARGIN` | `0.02` | Margen minimo para reportar oportunidad |
| `MAX_CONDITION_PRICE` | `0.95` | Precio maximo para considerar condicion "incierta" |
| `MAX_POSITION_SIZE` | `10000` | Tamano maximo de posicion para estimar beneficio |
| `EMBEDDING_SIMILARITY_THRESHOLD` | `0.7` | Similitud minima para comprobar par con LLM |
| `SPARK_MIN_MARKETS` | `100` | Minimo de mercados para justificar Spark |

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
