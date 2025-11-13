"""
ML Analysis Service

Servicio principal que coordina el análisis ML completo:
- Fluidez (Random Forest)
- Ritmo (XGBoost)
- Pronunciación (Azure Speech)
"""

import logging
import time
from datetime import datetime
from typing import Optional

from domain.models import MLPredictionResult, AudioFeatures
from infrastructure.ml_models import (
    FluencyAnalyzerRF,
    RhythmAnalyzerXGB,
    PronunciationAnalyzerAzure
)

logger = logging.getLogger(__name__)


class MLAnalysisService:
    """
    Servicio de análisis ML que coordina todos los modelos.
    """
    
    def __init__(
        self,
        fluency_analyzer: FluencyAnalyzerRF,
        rhythm_analyzer: RhythmAnalyzerXGB,
        pronunciation_analyzer: PronunciationAnalyzerAzure
    ):
        """
        Inicializa el servicio con los analizadores.
        
        Args:
            fluency_analyzer: Analizador de fluidez (RF)
            rhythm_analyzer: Analizador de ritmo (XGB)
            pronunciation_analyzer: Analizador de pronunciación (Azure)
        """
        self.fluency_analyzer = fluency_analyzer
        self.rhythm_analyzer = rhythm_analyzer
        self.pronunciation_analyzer = pronunciation_analyzer
        
        logger.info("✅ ML Analysis Service inicializado")
    
    def analyze(
        self,
        features: AudioFeatures,
        reference_text: Optional[str] = None,
        include_pronunciation: bool = True
    ) -> MLPredictionResult:
        """
        Realiza el análisis ML completo.
        
        Args:
            features: Features extraídos del audio
            reference_text: Texto esperado (para pronunciation)
            include_pronunciation: Si incluir análisis de pronunciación
        
        Returns:
            MLPredictionResult con todos los scores
        """
        start_time = time.time()
        
        logger.info(f"🔬 Iniciando análisis ML para attempt: {features.attempt_id}")
        
        result = MLPredictionResult(
            attempt_id=features.attempt_id,
            predicted_at=datetime.utcnow()
        )
        
        # 1. Predecir Fluidez (Random Forest)
        try:
            logger.info("Analizando fluidez...")
            fluency_features = features.get_fluency_features()
            fluency_score, fluency_confidence = self.fluency_analyzer.predict(fluency_features)
            
            result.fluency_score = fluency_score
            result.fluency_confidence = fluency_confidence
            result.random_forest_version = self.fluency_analyzer.get_version()
            
            logger.info(f"✅ Fluidez: {fluency_score:.1f} (confidence: {fluency_confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de fluidez: {e}")
            result.fluency_score = None
            result.fluency_confidence = None
        
        # 2. Predecir Ritmo (XGBoost)
        try:
            logger.info("Analizando ritmo...")
            rhythm_features = features.get_rhythm_features()
            rhythm_score, rhythm_confidence = self.rhythm_analyzer.predict(rhythm_features)
            
            result.rhythm_score = rhythm_score
            result.rhythm_confidence = rhythm_confidence
            result.xgboost_version = self.rhythm_analyzer.get_version()
            
            logger.info(f"✅ Ritmo: {rhythm_score:.1f} (confidence: {rhythm_confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de ritmo: {e}")
            result.rhythm_score = None
            result.rhythm_confidence = None
        
        # 3. Predecir Pronunciación (Azure)
        azure_start = time.time()
        
        if include_pronunciation and reference_text:
            try:
                logger.info("Analizando pronunciación...")
                
                if features.audio_base64:
                    pronunciation_score, pronunciation_details = self.pronunciation_analyzer.predict(
                        audio_base64=features.audio_base64,
                        reference_text=reference_text,
                        language="es-MX"
                    )
                    
                    result.pronunciation_score = pronunciation_score
                    result.azure_model_version = self.pronunciation_analyzer.get_version()
                    
                    logger.info(f"✅ Pronunciación: {pronunciation_score:.1f}")
                else:
                    logger.warning("⚠️ No hay audio_base64 - skip pronunciación")
                    result.pronunciation_score = None
                
            except Exception as e:
                logger.error(f"❌ Error en análisis de pronunciación: {e}")
                result.pronunciation_score = None
        else:
            logger.info("⏭️ Skip pronunciación (no habilitado o sin reference_text)")
            result.pronunciation_score = None
        
        azure_time = int((time.time() - azure_start) * 1000)
        result.azure_api_time_ms = azure_time if include_pronunciation else None
        
        # 4. Calcular Overall Score
        result.overall_score = self._calculate_overall_score(
            pronunciation=result.pronunciation_score,
            fluency=result.fluency_score,
            rhythm=result.rhythm_score
        )
        
        # 5. Timing total
        total_time = int((time.time() - start_time) * 1000)
        result.prediction_time_ms = total_time
        
        logger.info(f"✅ Análisis completado en {total_time}ms")
        logger.info(f"   Overall Score: {result.overall_score:.1f}/100")
        
        return result
    
    def _calculate_overall_score(
        self,
        pronunciation: Optional[float],
        fluency: Optional[float],
        rhythm: Optional[float]
    ) -> Optional[float]:
        """
        Calcula el score general ponderado.
        
        Pesos:
        - Pronunciation: 40%
        - Fluency: 30%
        - Rhythm: 30%
        
        Si falta algún score, redistribuye los pesos.
        """
        scores = []
        weights = []
        
        if pronunciation is not None:
            scores.append(pronunciation)
            weights.append(0.40)
        
        if fluency is not None:
            scores.append(fluency)
            weights.append(0.30)
        
        if rhythm is not None:
            scores.append(rhythm)
            weights.append(0.30)
        
        if not scores:
            return None
        
        # Normalizar pesos si falta algún score
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calcular promedio ponderado
        overall = sum(s * w for s, w in zip(scores, normalized_weights))
        
        return round(overall, 1)
    
    def get_models_info(self) -> dict:
        """
        Retorna información de todos los modelos cargados.
        
        Returns:
            Dict con versiones y features de cada modelo
        """
        return {
            'fluency': {
                'version': self.fluency_analyzer.get_version(),
                'model': 'RandomForestRegressor',
                'features': list(self.fluency_analyzer.get_feature_importance().keys())
            },
            'rhythm': {
                'version': self.rhythm_analyzer.get_version(),
                'model': 'XGBRegressor',
                'features': list(self.rhythm_analyzer.get_feature_importance().keys())
            },
            'pronunciation': {
                'version': self.pronunciation_analyzer.get_version(),
                'provider': 'Azure Speech Service'
            }
        }