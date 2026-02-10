## Roadmap del sistema de arbitraje en Polymarket

### 1. Arquitectura general
- [ ] Paralelizar el uso de LLMs
- [ ] Separar completamente los flujos:
  - [ ] Flujo A: detección de pares
  - [ ] Flujo B: ejecución de arbitraje

---

### 2. Lógica de detección de pares (LLM)
- [ ] Rehacer la lógica de búsqueda de pares reales usando LLM
- [ ] Definir criterios formales de “par real” (eventos, fechas, condiciones)
- [ ] Validar consistencia semántica entre mercados
- [ ] Reducir ruido y falsos positivos

---

### 3. Base de datos de pares
- [ ] Diseñar esquema de BD para pares a trackear
- [ ] Guardar:
  - [ ] IDs de mercados
  - [ ] Relación entre pares
  - [ ] Metadata del evento
  - [ ] Timestamp de creación
  - [ ] Estado del par (activo / descartado)
- [ ] Versionar pares (por cambios en condiciones)

---

### 4. Tracking de arbitraje
- [ ] Trackear pares en tiempo real
- [ ] Detectar oportunidades de arbitraje
- [ ] Calcular:
  - [ ] Spread efectivo
  - [ ] Fees
  - [ ] Liquidez disponible
- [ ] Definir umbrales mínimos de ejecución

---

### 5. Modelos de dependencia entre mercados
- [ ] Implementar detector de pares con modelos (PC / Erik)
- [ ] Evaluar dependencias causales entre mercados
- [ ] Comparar resultados vs LLM
- [ ] Usar modelos como filtro o confirmación

---

### 6. Infraestructura
- [ ] Desplegar detector de pares en una máquina dedicada
- [ ] Automatizar ejecución periódica
- [ ] Logging y monitoreo
- [ ] Manejo de fallos y reinicios

---

### 7. Ejecución de operaciones
- [ ] Investigar cómo operar en Polymarket (APIs, contratos, restricciones)
- [ ] Implementar bot de ejecución
- [ ] Gestión de riesgo:
  - [ ] Tamaño de posición
  - [ ] Exposición por evento
  - [ ] Límite de pérdidas
- [ ] Confirmación post-trade y auditoría

---

### 8. Integración final
- [ ] Conectar detector de pares → tracker
- [ ] Conectar tracker → bot de ejecución
- [ ] Tests end-to-end
- [ ] Simulación en paper trading antes de producción
