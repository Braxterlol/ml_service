"""
Dependency Injection para FastAPI
"""
import logging
from typing import Optional

from infrastructure.ml_models import (
    FluencyAnalyzerRF,
    RhythmAnalyzerXGB,
    PronunciationAnalyzerAzure
)
from application.services.ml_analysis_service import MLAnalysisService
from application.use_cases.analyze_attempt_use_case import AnalyzeAttemptUseCase
from infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


# ===== Singletons Globales =====
_fluency_analyzer: Optional[FluencyAnalyzerRF] = None
_rhythm_analyzer: Optional[RhythmAnalyzerXGB] = None
_pronunciation_analyzer: Optional[PronunciationAnalyzerAzure] = None
_ml_analysis_service: Optional[MLAnalysisService] = None


def get_fluency_analyzer() -> FluencyAnalyzerRF:
    """Retorna singleton de FluencyAnalyzerRF"""
    global _fluency_analyzer
    if _fluency_analyzer is None:
        _fluency_analyzer = FluencyAnalyzerRF(
            model_path=settings.FLUENCY_MODEL_PATH,
            scaler_path=settings.FLUENCY_SCALER_PATH
        )
    return _fluency_analyzer


def get_rhythm_analyzer() -> RhythmAnalyzerXGB:
    """Retorna singleton de RhythmAnalyzerXGB"""
    global _rhythm_analyzer
    if _rhythm_analyzer is None:
        _rhythm_analyzer = RhythmAnalyzerXGB(
            model_path=settings.RHYTHM_MODEL_PATH,
            scaler_path=settings.RHYTHM_SCALER_PATH
        )
    return _rhythm_analyzer


def get_pronunciation_analyzer() -> PronunciationAnalyzerAzure:
    """Retorna singleton de PronunciationAnalyzerAzure"""
    global _pronunciation_analyzer
    if _pronunciation_analyzer is None:
        _pronunciation_analyzer = PronunciationAnalyzerAzure(
            speech_key=settings.AZURE_SPEECH_KEY,
            speech_region=settings.AZURE_SPEECH_REGION
        )
    return _pronunciation_analyzer


def get_ml_analysis_service() -> MLAnalysisService:
    """Retorna singleton de MLAnalysisService"""
    global _ml_analysis_service
    if _ml_analysis_service is None:
        _ml_analysis_service = MLAnalysisService(
            fluency_analyzer=get_fluency_analyzer(),
            rhythm_analyzer=get_rhythm_analyzer(),
            pronunciation_analyzer=get_pronunciation_analyzer()
        )
    return _ml_analysis_service


def get_analyze_use_case() -> AnalyzeAttemptUseCase:
    """Retorna instancia de AnalyzeAttemptUseCase"""
    return AnalyzeAttemptUseCase(
        ml_analysis_service=get_ml_analysis_service()
    )