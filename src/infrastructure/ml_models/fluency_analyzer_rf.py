"""
Fluency Analyzer usando Random Forest

Este módulo carga el modelo Random Forest entrenado
y predice scores de fluidez.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FluencyAnalyzerRF:
    """
    Analizador de fluidez usando Random Forest.
    """
    
    def __init__(
        self,
        model_path: str = "models/fluency_rf_model.pkl",
        scaler_path: str = "models/fluency_scaler.pkl"
    ):
        """
        Inicializa el analizador de fluidez.
        
        Args:
            model_path: Path al modelo Random Forest
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
            logger.info(f"Cargando Random Forest desde {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Cargando scaler desde {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            
            logger.info("✅ Random Forest cargado exitosamente")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Archivo no encontrado: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo de fluidez: {e}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise RuntimeError(f"Error cargando modelo de fluidez: {e}")
    
    def predict(self, features: list) -> tuple[float, float]:
        """
        Predice el score de fluidez.
        
        Args:
            features: Lista de 11 features:
                [jitter, shimmer, pause_count, pause_duration_mean,
                 pause_duration_std, speaking_time_ratio, f0_std,
                 f0_range, energy_std, speech_rate_normalized,
                 articulation_rate_normalized]
        
        Returns:
            Tupla (score, confidence)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Modelos no cargados")
        
        if len(features) != 11:
            raise ValueError(f"Se esperan 11 features, se recibieron {len(features)}")
        
        try:
            # Convertir a array numpy
            features_array = np.array([features])
            
            # Normalizar
            features_scaled = self.scaler.transform(features_array)
            
            # Predecir
            score = self.model.predict(features_scaled)[0]
            
            # Calcular confianza (usando desviación estándar de los árboles)
            # Obtener predicciones de todos los árboles
            tree_predictions = np.array([
                tree.predict(features_scaled)[0] 
                for tree in self.model.estimators_
            ])
            
            # Confianza = 1 - (std / score)
            std = np.std(tree_predictions)
            confidence = max(0, min(1, 1 - (std / (score + 1e-6))))
            
            # Asegurar score en rango [0, 100]
            score = max(0, min(100, float(score)))
            
            logger.info(f"Fluency score predicho: {score:.1f} (confidence: {confidence:.2f})")
            
            return score, confidence
            
        except Exception as e:
            logger.error(f"❌ Error prediciendo fluidez: {e}")
            raise RuntimeError(f"Error en predicción de fluidez: {e}")
    
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
            'jitter', 'shimmer', 'pause_count', 'pause_duration_mean',
            'pause_duration_std', 'speaking_time_ratio', 'f0_std',
            'f0_range', 'energy_std', 'speech_rate_normalized',
            'articulation_rate_normalized'
        ]
        
        importances = self.model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }