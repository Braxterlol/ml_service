"""
Settings para ML Analysis Service
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List


class Settings(BaseSettings):
    """Configuración del servicio"""
    
    # Servicio
    SERVICE_NAME: str = "ml-analysis-service"
    SERVICE_PORT: int = 8002
    DEBUG: bool = False
    
    # Modelos ML
    FLUENCY_MODEL_PATH: str = "models/fluency_rf_model.pkl"
    FLUENCY_SCALER_PATH: str = "models/fluency_scaler.pkl"
    RHYTHM_MODEL_PATH: str = "models/rhythm_xgb_model.pkl"
    RHYTHM_SCALER_PATH: str = "models/rhythm_scaler.pkl"
    
    # Azure Speech Service (opcional)
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    
    # MongoDB (para cargar features)
    MONGODB_URL: str = "mongodb://localhost:27017/"
    MONGODB_DB_NAME: str = "audio_features_db"
    COLLECTION_NAME: str = "audio_features"
    
    # PostgreSQL (para guardar predictions)
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/ml_analysis_db"
    
    # API Keys (para auth entre servicios)
    INTERNAL_API_KEY: str = "change-me-in-production"
    
    # CORS (CORREGIDO)
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # LLM Feedback Service
    LLM_FEEDBACK_SERVICE_URL: str = Field(
        default="http://localhost:8003",
        description="URL del LLM Feedback Service"
    )
    LLM_FEEDBACK_TIMEOUT: int = Field(
        default=30,
        description="Timeout para llamadas al LLM Service (segundos)"
    )
    LLM_FEEDBACK_ENABLED: bool = Field(
        default=True,
        description="Habilitar llamadas al LLM Feedback Service"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instancia global
settings = Settings()