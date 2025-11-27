"""
ML Analysis API Controller - CON FEEDBACK

Versión mejorada que integra LLM Feedback Service.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import uuid

from application.use_cases.analyze_attempt_use_case import AnalyzeAttemptUseCase
from infrastructure.config.settings import settings
from infrastructure.api.dependencies import get_analyze_use_case, get_llm_feedback_client
from infrastructure.clients import LLMFeedbackClient

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
    
    # Para pronunciation
    reference_text: Optional[str] = Field(None, description="Texto esperado")
    audio_base64: Optional[str] = Field(None, description="Audio en base64")
    
    # Metadata del ejercicio (NUEVO - para feedback)
    exercise_metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata: type, content, difficulty_level, etc."
    )
    
    # Info del usuario (NUEVO - para feedback)
    user_age: Optional[int] = Field(None, description="Edad del usuario")
    attempt_number: Optional[int] = Field(1, description="Número de intento")
    previous_best_score: Optional[float] = Field(None, description="Mejor score anterior")
    
    class Config:
        json_schema_extra = {
            "example": {
                "attempt_id": "4e8536b2-505e-4f2e-b27c-87c5a9d1faea",
                "user_id": "00000000-0000-0000-0000-000000000001",
                "exercise_id": "fonema_r_suave_2",
                "audio_features": {
                    "jitter": 0.01,
                    "shimmer": 0.05,
                    "pause_count": 3,
                    "pause_duration_mean": 0.3,
                    "pause_duration_std": 0.1,
                    "speaking_time_ratio": 0.85,
                    "f0_std": 15.0,
                    "f0_range": 50.0,
                    "energy_std": 0.2,
                    "speech_rate_normalized": 0.8,
                    "articulation_rate_normalized": 0.75
                },
                "reference_text": "rata, rana, rojo",
                "exercise_metadata": {
                    "type": "fonema",
                    "content": "palabras con /r/ suave",
                    "difficulty_level": 2
                },
                "user_age": 7,
                "attempt_number": 3
            }
        }


class FeedbackData(BaseModel):
    """Datos del feedback generado"""
    main_message: str
    strengths: List[str]
    areas_to_improve: List[str]
    specific_tip: str
    celebration: Optional[str]
    encouragement: str
    tone: str


class AnalyzeResponse(BaseModel):
    """Response del análisis ML con feedback"""
    attempt_id: str
    status: str
    scores: Dict[str, Optional[float]]
    confidence: Optional[Dict[str, float]] = None
    model_versions: Dict[str, str]
    processing_info: Dict[str, int]
    
    # NUEVO: Datos de progresión
    progression: Optional[Dict[str, Any]] = None
    
    # NUEVO: Feedback personalizado
    feedback: Optional[FeedbackData] = None
    
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
                "progression": {
                    "passed": True,
                    "stars_earned": 2,
                    "unlocked_next": True
                },
                "feedback": {
                    "main_message": "¡Excelente trabajo!",
                    "strengths": ["Tu ritmo fue muy natural"],
                    "areas_to_improve": ["Trabaja la fluidez"],
                    "specific_tip": "Practica sin pausas",
                    "celebration": "¡Desbloqueaste el siguiente nivel! 🎉",
                    "encouragement": "¡Sigue así!",
                    "tone": "positive"
                }
            }
        }


# ===== Helper Functions =====

def calculate_progression(overall_score: float) -> Dict[str, Any]:
    """
    Calcula la progresión del usuario basado en el score.
    
    Args:
        overall_score: Score general (0-100)
    
    Returns:
        Dict con passed, stars_earned, unlocked_next
    """
    passed = overall_score >= 70.0
    
    # Calcular estrellas
    if overall_score >= 90:
        stars_earned = 3
    elif overall_score >= 80:
        stars_earned = 2
    elif overall_score >= 70:
        stars_earned = 1
    else:
        stars_earned = 0
    
    # Desbloquear siguiente si pasó
    unlocked_next = passed
    
    return {
        "passed": passed,
        "stars_earned": stars_earned,
        "unlocked_next": unlocked_next
    }


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
    analyze_use_case: AnalyzeAttemptUseCase = Depends(get_analyze_use_case),
    feedback_client: LLMFeedbackClient = Depends(get_llm_feedback_client)  # NUEVO
):
    """
    Analiza un audio y retorna scores de ML + feedback personalizado.
    
    **NUEVO:** Ahora incluye feedback generado por LLM.
    
    **Requiere:**
    - Header: X-API-Key
    
    **Retorna:**
    - Scores: pronunciation, fluency, rhythm, overall
    - Progresión: passed, stars, unlocked
    - Feedback: mensaje motivador personalizado
    - Confianza de predicciones
    - Versiones de modelos
    - Timing de procesamiento
    """
    try:
        logger.info(f"📊 Solicitud de análisis para attempt: {request.attempt_id}")
        
        # ===== PASO 1: ANÁLISIS ML (sin cambios) =====
        result = analyze_use_case.execute(
            attempt_id=request.attempt_id,
            audio_features_doc=request.audio_features,
            reference_text=request.reference_text,
            audio_base64=request.audio_base64
        )
        
        logger.info(f"✅ Análisis ML completado: {result.overall_score:.1f}/100")
        
        # ===== PASO 2: CALCULAR PROGRESIÓN =====
        progression = calculate_progression(result.overall_score)
        
        # ===== PASO 3: GENERAR FEEDBACK (NUEVO) =====
        feedback_data = None
        
        if settings.LLM_FEEDBACK_ENABLED:
            try:
                logger.info(f"📝 Generando feedback personalizado...")
                
                # Preparar datos para feedback
                scores = {
                    "pronunciation_score": result.pronunciation_score or 0.0,
                    "fluency_score": result.fluency_score or 0.0,
                    "rhythm_score": result.rhythm_score or 0.0,
                    "overall_score": result.overall_score
                }
                
                # Obtener info del ejercicio
                exercise_metadata = request.exercise_metadata or {}
                
                # Validar exercise_type (debe ser: fonema, ritmo, o entonacion)
                exercise_type = exercise_metadata.get("type", "fonema")
                if exercise_type not in ["fonema", "ritmo", "entonacion"]:
                    exercise_type = "fonema"  # Default seguro
                
                exercise_info = {
                    "exercise_type": exercise_type,
                    "exercise_content": exercise_metadata.get("content", "ejercicio de pronunciación"),
                    "difficulty_level": exercise_metadata.get("difficulty_level", 1),
                    "reference_text": request.reference_text or ""
                }
                
                progression_info = {
                    "passed": progression["passed"],
                    "stars_earned": progression["stars_earned"],
                    "unlocked_next": progression["unlocked_next"],
                    "previous_best_score": request.previous_best_score
                }
                
                # Llamar al LLM Feedback Service
                feedback_response = await feedback_client.generate_feedback(
                    attempt_id=request.attempt_id,
                    user_id=request.user_id,
                    exercise_id=request.exercise_id,
                    scores=scores,
                    exercise_info=exercise_info,
                    progression_info=progression_info,
                    user_age=request.user_age,
                    attempt_number=request.attempt_number or 1
                )
                
                if feedback_response:
                    feedback_data = FeedbackData(
                        main_message=feedback_response.main_message,
                        strengths=feedback_response.strengths,
                        areas_to_improve=feedback_response.areas_to_improve,
                        specific_tip=feedback_response.specific_tip,
                        celebration=feedback_response.celebration,
                        encouragement=feedback_response.encouragement,
                        tone=feedback_response.tone
                    )
                    logger.info(f"✅ Feedback generado exitosamente")
                else:
                    logger.warning(f"⚠️ No se pudo generar feedback, usando valores por defecto")
                    
            except Exception as e:
                logger.error(f"⚠️ Error generando feedback: {e}")
                # Continuar sin feedback (no es crítico)
        
        # ===== PASO 4: CONSTRUIR RESPONSE =====
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
            },
            progression=progression,  # NUEVO
            feedback=feedback_data    # NUEVO
        )
        
        logger.info(f"✨ Respuesta completa generada")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis ML: {str(e)}")


@router.get("/models/info")
async def get_models_info(
    api_key: str = Depends(verify_api_key),
    analyze_use_case: AnalyzeAttemptUseCase = Depends(get_analyze_use_case)
):
    """Retorna información sobre los modelos cargados"""
    try:
        models_info = analyze_use_case.ml_analysis_service.get_models_info()
        
        return {
            "service": "ml-analysis-service",
            "models": models_info
        }
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo info de modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    feedback_client: LLMFeedbackClient = Depends(get_llm_feedback_client)  # NUEVO
):
    """
    Health check endpoint.
    
    NUEVO: También verifica la disponibilidad del LLM Feedback Service.
    """
    # Verificar LLM Feedback Service
    llm_service_available = False
    if settings.LLM_FEEDBACK_ENABLED:
        try:
            llm_service_available = await feedback_client.health_check()
        except:
            pass
    
    return {
        "service": "ml-analysis-service",
        "status": "healthy",
        "version": "1.0.0",
        "dependencies": {
            "llm_feedback_service": "available" if llm_service_available else "unavailable"
        }
    }