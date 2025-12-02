"""
Domain Models para ML Analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class MLPredictionResult:
    """
    Resultado del análisis ML completo.
    """
    attempt_id: str
    
    # Scores principales
    pronunciation_score: Optional[float] = None
    fluency_score: Optional[float] = None
    rhythm_score: Optional[float] = None
    overall_score: Optional[float] = None
    
    # Confianza de predicciones
    fluency_confidence: Optional[float] = None
    rhythm_confidence: Optional[float] = None
    
    # Metadata de modelos
    random_forest_version: str = "v1.0.0"
    xgboost_version: str = "v1.0.0"
    azure_model_version: Optional[str] = None
    
    # Timing
    prediction_time_ms: Optional[int] = None
    azure_api_time_ms: Optional[int] = None
    
    # Timestamp
    predicted_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            'attempt_id': self.attempt_id,
            'pronunciation_score': self.pronunciation_score,
            'fluency_score': self.fluency_score,
            'rhythm_score': self.rhythm_score,
            'overall_score': self.overall_score,
            'fluency_confidence': self.fluency_confidence,
            'rhythm_confidence': self.rhythm_confidence,
            'random_forest_version': self.random_forest_version,
            'xgboost_version': self.xgboost_version,
            'azure_model_version': self.azure_model_version,
            'prediction_time_ms': self.prediction_time_ms,
            'azure_api_time_ms': self.azure_api_time_ms,
            'predicted_at': self.predicted_at.isoformat() if self.predicted_at else None
        }


@dataclass
class AudioFeatures:
    """
    Features extraídos del audio para análisis ML.
    """
    attempt_id: str
    
    # Features para fluidez (11 features)
    jitter: float
    shimmer: float
    pause_count: int
    pause_duration_mean: float
    pause_duration_std: float
    speaking_time_ratio: float
    f0_std: float
    f0_range: float
    energy_std: float
    speech_rate_normalized: float
    articulation_rate_normalized: float
    
    # Features para ritmo (9 features)
    speech_rate: float
    articulation_rate: float
    pause_density: float
    average_pause_duration: float
    duration_seconds: float
    pause_pattern_regularity: float
    
    # Audio base64 para Azure (opcional)
    audio_base64: Optional[str] = None
    
    def get_fluency_features(self) -> list:
        """Retorna features para Random Forest (Fluidez)"""
        return [
            self.jitter,
            self.shimmer,
            self.pause_count,
            self.pause_duration_mean,
            self.pause_duration_std,
            self.speaking_time_ratio,
            self.f0_std,
            self.f0_range,
            self.energy_std,
            self.speech_rate_normalized,
            self.articulation_rate_normalized
        ]
    
    def get_rhythm_features(self) -> list:
        """Retorna features para XGBoost (Ritmo)"""
        features = [
            self.speech_rate,
            self.articulation_rate,
            self.pause_count,
            self.pause_density,
            self.speaking_time_ratio,
            self.average_pause_duration,
            self.pause_duration_std,
            self.duration_seconds,
            self.pause_pattern_regularity
        ]
        
        # DEBUG: Log para ver valores reales
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Rhythm features: speech_rate={self.speech_rate:.2f}, "
                    f"articulation_rate={self.articulation_rate:.2f}, "
                    f"pause_count={self.pause_count}, "
                    f"pause_density={self.pause_density:.4f}, "
                    f"speaking_time_ratio={self.speaking_time_ratio:.4f}, "
                    f"duration={self.duration_seconds:.2f}s")
        
        return features