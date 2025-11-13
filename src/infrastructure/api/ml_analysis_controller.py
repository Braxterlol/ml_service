"""
ML Analysis API Controller

Endpoints para el ML Analysis Service.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from application.use_cases.analyze_attempt_use_case import AnalyzeAttemptUseCase
from infrastructure.config.settings import settings
from infrastructure.api.dependencies import get_analyze_use_case

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ML Analysis"])


# ===== Request/Response Models =====

class AnalyzeRequest(BaseModel):
    """Request para análisis ML"""
    attempt_id: str = Field(..., description="ID del attempt")
    user_id: str = Field(..., description="ID del usuario")
    exercise_id: str = Field(..., description="ID del ejercicio")
    
    # Features del audio
    audio_features: Dict[str, Any] = Field(..., description="Features extraídos del audio")
    
    # Opcional: para pronunciation
    reference_text: Optional[str] = Field(None, description="Texto esperado")
    audio_base64: Optional[str] = Field(None, description="Audio en base64")
    
    # Metadata del ejercicio
    exercise_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata del ejercicio")
    
    class Config:
        json_schema_extra = {
            "example": {
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
                        "pause_count": 3
                    },
                    "duration_seconds": 0.704
                },
                "reference_text": "rata, rana, rojo",
                "audio_base64": "UklGRiQAAABXQVZF..."
            }
        }


class AnalyzeResponse(BaseModel):
    """Response del análisis ML"""
    attempt_id: str
    status: str
    scores: Dict[str, Optional[float]]
    confidence: Optional[Dict[str, float]] = None
    model_versions: Dict[str, str]
    processing_info: Dict[str, int]
    
    class Config:
        json_schema_extra = {
            "example": {
                "attempt_id": "4e8536b2-505e-4f2e-b27c-87c5a9d1faea",
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
                    "azure": "2024-11-01"
                },
                "processing_info": {
                    "prediction_time_ms": 850,
                    "azure_api_time_ms": 1200,
                    "total_time_ms": 2050
                }
            }
        }


class ModelsInfoResponse(BaseModel):
    """Información de modelos cargados"""
    service: str
    models: Dict[str, Any]


# ===== Dependency para validar API Key =====

def verify_api_key(x_api_key: str = Header(...)):
    """Verifica el API key interno"""
    if x_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# ===== Endpoints =====

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    request: AnalyzeRequest,
    api_key: str = Depends(verify_api_key),
    analyze_use_case: AnalyzeAttemptUseCase = Depends(get_analyze_use_case)
):
    """
    Analiza un audio y retorna scores de ML.
    
    **Requiere:**
    - Header: X-API-Key
    
    **Retorna:**
    - Scores: pronunciation, fluency, rhythm, overall
    - Confianza de predicciones
    - Versiones de modelos
    - Timing de procesamiento
    """
    try:
        logger.info(f"📊 Solicitud de análisis para attempt: {request.attempt_id}")
        
        # Ejecutar análisis
        result = analyze_use_case.execute(
            attempt_id=request.attempt_id,
            audio_features_doc=request.audio_features,
            reference_text=request.reference_text,
            audio_base64=request.audio_base64
        )
        
        # Construir response
        response = AnalyzeResponse(
            attempt_id=result.attempt_id,
            status="completed",
            scores={
                "pronunciation": result.pronunciation_score,
                "fluency": result.fluency_score,
                "rhythm": result.rhythm_score,
                "overall": result.overall_score
            },
            confidence={
                "fluency": result.fluency_confidence,
                "rhythm": result.rhythm_confidence
            } if result.fluency_confidence else None,
            model_versions={
                "random_forest": result.random_forest_version,
                "xgboost": result.xgboost_version,
                "azure": result.azure_model_version or "not_used"
            },
            processing_info={
                "prediction_time_ms": result.prediction_time_ms or 0,
                "azure_api_time_ms": result.azure_api_time_ms or 0,
                "total_time_ms": result.prediction_time_ms or 0
            }
        )
        
        logger.info(f"✅ Análisis completado: {result.overall_score:.1f}/100")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis ML: {str(e)}")


@router.get("/models/info", response_model=ModelsInfoResponse)
async def get_models_info(
    api_key: str = Depends(verify_api_key),
    analyze_use_case: AnalyzeAttemptUseCase = Depends(get_analyze_use_case)
):
    """
    Retorna información sobre los modelos cargados.
    
    **Requiere:**
    - Header: X-API-Key
    
    **Retorna:**
    - Versiones de modelos
    - Features utilizados
    - Estado de carga
    """
    try:
        models_info = analyze_use_case.ml_analysis_service.get_models_info()
        
        return ModelsInfoResponse(
            service="ml-analysis-service",
            models=models_info
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo info de modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint (sin auth)"""
    return {
        "service": "ml-analysis-service",
        "status": "healthy",
        "version": "1.0.0"
    }