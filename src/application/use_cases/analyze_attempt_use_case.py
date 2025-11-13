"""
Use Case: Analyze Attempt

Orquesta el análisis completo de un attempt.
"""

import logging
from typing import Optional
import numpy as np

from domain.models import MLPredictionResult, AudioFeatures

logger = logging.getLogger(__name__)


class AnalyzeAttemptUseCase:
    """
    Use case para analizar un attempt completo.
    """
    
    def __init__(self, ml_analysis_service):
        """
        Inicializa el use case.
        
        Args:
            ml_analysis_service: Servicio de análisis ML
        """
        self.ml_analysis_service = ml_analysis_service
    
    def execute(
        self,
        attempt_id: str,
        audio_features_doc: dict,
        reference_text: Optional[str] = None,
        audio_base64: Optional[str] = None
    ) -> MLPredictionResult:
        """
        Ejecuta el análisis ML para un attempt.
        
        Args:
            attempt_id: ID del attempt
            audio_features_doc: Documento de MongoDB con features
            reference_text: Texto esperado (opcional)
            audio_base64: Audio en base64 (opcional, para Azure)
        
        Returns:
            MLPredictionResult con todos los scores
        """
        logger.info(f"📊 Ejecutando análisis para attempt: {attempt_id}")
        
        # 1. Extraer features del documento MongoDB
        features = self._extract_features_from_doc(attempt_id, audio_features_doc, audio_base64)
        
        # 2. Ejecutar análisis
        result = self.ml_analysis_service.analyze(
            features=features,
            reference_text=reference_text,
            include_pronunciation=bool(reference_text and audio_base64)
        )
        
        return result
    
    def _extract_features_from_doc(
        self,
        attempt_id: str,
        doc: dict,
        audio_base64: Optional[str] = None
    ) -> AudioFeatures:
        """
        Extrae AudioFeatures desde un documento de MongoDB.
        
        Args:
            attempt_id: ID del attempt
            doc: Documento de MongoDB con features acústicos
            audio_base64: Audio en base64 (opcional)
        
        Returns:
            AudioFeatures poblado con los datos del documento
        """
        prosody = doc.get('prosody', {})
        rhythm = doc.get('rhythm', {})
        duration = doc.get('duration_seconds', 1.0)
        
        # Extraer features para fluidez
        jitter = prosody.get('jitter', 0.03)
        shimmer = prosody.get('shimmer', 0.10)
        
        pause_count = rhythm.get('pause_count', 0)
        pause_durations = rhythm.get('pause_durations_ms', [])
        
        if pause_durations and len(pause_durations) > 0:
            pause_duration_mean = float(np.mean(pause_durations))
            pause_duration_std = float(np.std(pause_durations))
            
            # Calcular regularidad de pausas
            cv = pause_duration_std / pause_duration_mean if pause_duration_mean > 0 else 1.0
            pause_pattern_regularity = max(0, 1.0 - cv)
        else:
            pause_duration_mean = 0.0
            pause_duration_std = 0.0
            pause_pattern_regularity = 1.0
        
        # Speaking time ratio
        speaking_time_ms = rhythm.get('speaking_time_ms', duration * 1000)
        total_duration_ms = rhythm.get('total_duration_ms', duration * 1000)
        speaking_time_ratio = speaking_time_ms / total_duration_ms if total_duration_ms > 0 else 0.8
        
        # F0 stats
        f0_stats = prosody.get('f0_stats', {})
        f0_std = f0_stats.get('std', 30.0)
        f0_range = f0_stats.get('range', 50.0)
        
        # Energy stats
        energy_stats = prosody.get('energy_stats', {})
        energy_std = energy_stats.get('std', 0.3)
        
        # Speech rates
        speech_rate = rhythm.get('speech_rate', 3.0)
        articulation_rate = rhythm.get('articulation_rate', 3.5)
        speech_rate_normalized = speech_rate / duration if duration > 0 else speech_rate
        articulation_rate_normalized = articulation_rate / duration if duration > 0 else articulation_rate
        
        # Pause density
        pause_density = pause_count / duration if duration > 0 else 0
        
        # Average pause duration
        average_pause_duration = pause_duration_mean
        
        # Crear AudioFeatures
        features = AudioFeatures(
            attempt_id=attempt_id,
            jitter=jitter,
            shimmer=shimmer,
            pause_count=pause_count,
            pause_duration_mean=pause_duration_mean,
            pause_duration_std=pause_duration_std,
            speaking_time_ratio=speaking_time_ratio,
            f0_std=f0_std,
            f0_range=f0_range,
            energy_std=energy_std,
            speech_rate_normalized=speech_rate_normalized,
            articulation_rate_normalized=articulation_rate_normalized,
            speech_rate=speech_rate,
            articulation_rate=articulation_rate,
            pause_density=pause_density,
            average_pause_duration=average_pause_duration,
            duration_seconds=duration,
            pause_pattern_regularity=pause_pattern_regularity,
            audio_base64=audio_base64
        )
        
        return features