"""
Rhythm Analyzer usando XGBoost

Este módulo carga el modelo XGBoost entrenado
y predice scores de ritmo.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RhythmAnalyzerXGB:
    """
    Analizador de ritmo usando XGBoost.
    """
    
    def __init__(
        self,
        model_path: str = "models/rhythm_xgb_model.pkl",
        scaler_path: str = "models/rhythm_scaler.pkl"
    ):
        """
        Inicializa el analizador de ritmo.
        
        Args:
            model_path: Path al modelo XGBoost
            scaler_path: Path al StandardScaler
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self.version = "v1.0.0"
        
        self._load_models()
    
    def _load_models(self):
        """Carga el modelo y scaler"""
        try:
            logger.info(f"Cargando XGBoost desde {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Cargando scaler desde {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            
            logger.info("✅ XGBoost cargado exitosamente")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Archivo no encontrado: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo de ritmo: {e}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise RuntimeError(f"Error cargando modelo de ritmo: {e}")
    
    def predict(self, features: list) -> tuple[float, float]:
        """
        Predice el score de ritmo.
        
        Args:
            features: Lista de 9 features:
                [speech_rate, articulation_rate, pause_count,
                 pause_density, speaking_time_ratio,
                 average_pause_duration, pause_duration_std,
                 duration_seconds, pause_pattern_regularity]
        
        Returns:
            Tupla (score, confidence)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Modelos no cargados")
        
        if len(features) != 9:
            raise ValueError(f"Se esperan 9 features, se recibieron {len(features)}")
        
        try:
            # Convertir a array numpy
            features_array = np.array([features])
            
            # Normalizar
            features_scaled = self.scaler.transform(features_array)
            
            # Predecir
            score = self.model.predict(features_scaled)[0]
            
            # Para XGBoost, la confianza se puede estimar con el margin
            # (distancia de la predicción al threshold)
            # Por ahora, usamos una confianza fija alta
            # TODO: Implementar cálculo de confianza más sofisticado
            confidence = 0.90
            
            # Asegurar score en rango [0, 100]
            score = max(0, min(100, float(score)))
            
            logger.info(f"Rhythm score predicho: {score:.1f} (confidence: {confidence:.2f})")
            
            return score, confidence
            
        except Exception as e:
            logger.error(f"❌ Error prediciendo ritmo: {e}")
            raise RuntimeError(f"Error en predicción de ritmo: {e}")
    
    def get_version(self) -> str:
        """Retorna la versión del modelo"""
        return self.version
    
    def get_feature_importance(self) -> dict:
        """
        Retorna la importancia de features.
        
        Returns:
            Dict con nombres de features y sus importancias
        """
        if self.model is None:
            return {}
        
        feature_names = [
            'speech_rate', 'articulation_rate', 'pause_count',
            'pause_density', 'speaking_time_ratio',
            'average_pause_duration', 'pause_duration_std',
            'duration_seconds', 'pause_pattern_regularity'
        ]
        
        importances = self.model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }