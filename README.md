# ML Analysis Service

Microservicio para análisis ML de audios infantiles.

## 🎯 Funcionalidad

Este servicio analiza audios de niños y retorna:
- **Fluidez Score** (Random Forest)
- **Ritmo Score** (XGBoost)
- **Pronunciación Score** (Azure Speech - opcional)
- **Overall Score** (promedio ponderado)

---

## 📁 Estructura

```
ml_analysis_service/
├── src/
│   └── ml_analysis/
│       ├── domain/
│       │   └── models/           # Modelos de dominio
│       ├── application/
│       │   ├── services/         # Lógica de negocio
│       │   └── use_cases/        # Casos de uso
│       └── infrastructure/
│           ├── ml_models/        # RF, XGB, Azure
│           ├── api/              # FastAPI controllers
│           └── config/           # Settings
│
├── models/                       # Modelos entrenados (.pkl)
├── tests/                        # Tests
├── main.py                       # Entry point
├── requirements.txt
└── .env
```

---

## 🚀 Setup

### 1. Instalar Dependencias

```bash
cd ml_analysis_service
pip install -r requirements.txt
```

### 2. Copiar Modelos Entrenados

```bash
# Desde el directorio donde entrenaste los modelos
cp models/*.pkl ml_analysis_service/models/
```

Deberías tener:
```
models/
├── fluency_rf_model.pkl
├── fluency_scaler.pkl
├── rhythm_xgb_model.pkl
└── rhythm_scaler.pkl
```

### 3. Configurar Variables de Entorno

```bash
# Copiar ejemplo
cp .env.example .env

# Editar .env
nano .env
```

Configurar:
- `MONGODB_URL` (tu cluster de MongoDB Atlas)
- `INTERNAL_API_KEY` (para auth entre servicios)
- `AZURE_SPEECH_KEY` y `AZURE_SPEECH_REGION` (opcional)

### 4. Ejecutar Servicio

```bash
python main.py
```

El servicio estará disponible en:
- **API:** http://localhost:8002
- **Docs:** http://localhost:8002/docs

---

## 📡 API Endpoints

### POST /api/v1/ml/analyze

Analiza un audio y retorna scores.

**Headers:**
```
X-API-Key: secret_key_12345
```

**Request:**
```json
{
  "attempt_id": "4e8536b2-505e-4f2e-b27c-87c5a9d1faea",
  "user_id": "00000000-0000-0000-0000-000000000001",
  "exercise_id": "fonema_r_suave_2",
  "audio_features": {
    "prosody": {
      "jitter": 0.02,
      "shimmer": 0.08,
      "f0_stats": {"mean": 125, "std": 35, "range": 50}
    },
    "rhythm": {
      "speech_rate": 3.2,
      "articulation_rate": 3.8,
      "pause_count": 3,
      "pause_durations_ms": [250, 180, 320],
      "speaking_time_ms": 2840,
      "total_duration_ms": 3590
    },
    "duration_seconds": 3.59
  },
  "reference_text": "rata, rana, rojo",
  "audio_base64": "UklGRiQAAABXQVZF..."
}
```

**Response:**
```json
{
  "attempt_id": "4e8536b2-...",
  "status": "completed",
  "scores": {
    "pronunciation": 85.3,
    "fluency": 78.6,
    "rhythm": 92.1,
    "overall": 85.0
  },
  "confidence": {
    "fluency": 0.92,
    "rhythm": 0.88
  },
  "model_versions": {
    "random_forest": "v1.0.0",
    "xgboost": "v1.0.0",
    "azure": "not_used"
  },
  "processing_info": {
    "prediction_time_ms": 850,
    "azure_api_time_ms": 0,
    "total_time_ms": 850
  }
}
```

### GET /api/v1/ml/models/info

Información de modelos cargados.

**Headers:**
```
X-API-Key: secret_key_12345
```

**Response:**
```json
{
  "service": "ml-analysis-service",
  "models": {
    "fluency": {
      "version": "v1.0.0",
      "model": "RandomForestRegressor",
      "features": ["jitter", "shimmer", ...]
    },
    "rhythm": {
      "version": "v1.0.0",
      "model": "XGBRegressor",
      "features": ["speech_rate", ...]
    },
    "pronunciation": {
      "version": "pending",
      "provider": "Azure Speech Service"
    }
  }
}
```

### GET /api/v1/ml/health

Health check (sin auth).

**Response:**
```json
{
  "service": "ml-analysis-service",
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## 🧪 Testing

### Con cURL

```bash
# Health check
curl http://localhost:8002/api/v1/ml/health

# Models info
curl -H "X-API-Key: secret_key_12345" \
  http://localhost:8002/api/v1/ml/models/info

# Analyze (ver archivo test_ml_service.sh)
```

### Con Python

Ver `tests/test_service.py`

---

## 🔧 Troubleshooting

### Error: "No se pudo cargar el modelo"

- Verifica que los archivos `.pkl` existan en `models/`
- Verifica las rutas en `.env`

### Error: "Invalid API Key"

- Verifica que el header `X-API-Key` coincida con `INTERNAL_API_KEY` en `.env`

### Error: "ModuleNotFoundError"

```bash
pip install -r requirements.txt
```

---

## 📊 Scores

### Pesos del Overall Score

- Pronunciación: **40%** (cuando disponible)
- Fluidez: **30%**
- Ritmo: **30%**

Si no hay pronunciation, se redistribuyen los pesos proporcionalmente.

### Interpretación de Scores

- **90-100**: Excelente
- **70-89**: Bueno
- **50-69**: Regular
- **0-49**: Necesita mejorar

---

## 🔐 Seguridad

- Auth entre servicios: API Key en headers
- CORS configurado para dominios específicos en producción
- Variables sensibles en `.env` (no commitear)

---

## 📝 Próximos Pasos

1. ✅ Servicio funcionando con RF y XGB
2. ⬜ Integrar con Audio Processing Service
3. ✅ Agregar Azure Speech Service
4. ⬜ Guardar predictions en PostgreSQL
5. ⬜ Agregar LLM Feedback Service

---

## 🐛 Logs

Los logs se muestran en consola con formato:
```
2024-11-12 10:30:00 - ml_analysis - INFO - ✅ Random Forest cargado
```

Para más detalle, cambiar `DEBUG=true` en `.env`