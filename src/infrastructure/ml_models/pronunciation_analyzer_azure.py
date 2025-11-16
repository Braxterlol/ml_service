"""
Pronunciation Analyzer usando Azure Speech Service

Este módulo usa Azure Cognitive Services Speech SDK
para evaluar pronunciación.
"""

import logging
import base64
import io
import wave
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Intentar importar Azure SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("⚠️ Azure Speech SDK no instalado. Instala con: pip install azure-cognitiveservices-speech")


class PronunciationAnalyzerAzure:
    """
    Analizador de pronunciación usando Azure Speech Service.
    """
    
    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None
    ):
        """
        Inicializa el analizador de pronunciación.
        
        Args:
            speech_key: Azure Speech API key
            speech_region: Azure region (ej: 'eastus')
        """
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.version = "2024-11-01"
        self.is_configured = False
        
        if not AZURE_AVAILABLE:
            logger.warning("⚠️ Azure Speech SDK no disponible")
            return
        
        if speech_key and speech_region:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=speech_key,
                region=speech_region
            )
            self.is_configured = True
            logger.info("✅ Azure Speech Service configurado correctamente")
        else:
            logger.warning("⚠️ Azure Speech Service NO configurado (usando placeholder)")
    
    def predict(
        self,
        audio_base64: str,
        reference_text: str,
        language: str = "es-MX"
    ) -> tuple[float, dict]:
        """
        Predice el score de pronunciación usando Azure.
        
        Args:
            audio_base64: Audio en base64
            reference_text: Texto esperado
            language: Código de idioma (es-MX para español mexicano)
        
        Returns:
            Tupla (score, detalles)
        """
        if not self.is_configured or not AZURE_AVAILABLE:
            return self._placeholder_score(reference_text)
        
        try:
            import base64
            audio_bytes = base64.b64decode(audio_base64)
            logger.info(f"🎤 Audio recibido: {len(audio_bytes)} bytes")
            logger.info(f"📝 Reference text: '{reference_text}'")
            
            # Crear el speech config
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region
            )
            
            # Configurar para reconocimiento de pronunciación
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                enable_miscue=True
            )
            
            # Audio en memoria
            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,  # Azure prefiere 16kHz
                bits_per_sample=16,
                channels=1
            )
            
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
            push_stream.write(audio_bytes)
            push_stream.close()
            
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # Speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
                language="es-MX"  # Español de México
            )
            
            # Aplicar configuración de pronunciación
            pronunciation_config.apply_to(speech_recognizer)
            
            # Reconocer
            result = speech_recognizer.recognize_once_async().get()
            
            # ✅ DEBUG: Ver qué reconoció Azure
            logger.info(f"🔍 Azure result reason: {result.reason}")
            logger.info(f"🔍 Azure recognized text: '{result.text}'")
        
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
                
                score = pronunciation_result.pronunciation_score
                confidence = pronunciation_result.accuracy_score / 100.0
                
                logger.info(f"✅ Azure pronunciation score: {score}")
                logger.info(f"✅ Azure accuracy: {confidence}")
                logger.info(f"✅ Azure fluency: {pronunciation_result.fluency_score}")
                logger.info(f"✅ Azure completeness: {pronunciation_result.completeness_score}")
                
                return float(score), float(confidence)
            else:
                logger.warning(f"⚠️ Azure no reconoció voz. Reason: {result.reason}")
                if result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning(f"⚠️ No match details: {result.no_match_details}")
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    logger.error(f"❌ Canceled: {cancellation.reason}")
                    logger.error(f"❌ Error details: {cancellation.error_details}")
                
                return 0.0, 0.0      
        except Exception as e:
            logger.error(f"❌ Error en Azure Speech: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0, 0.0
    
    def _placeholder_score(
        self,
        reference_text: str,
        status: str = 'not_configured',
        error: Optional[str] = None
    ) -> tuple[float, dict]:
        """
        Retorna un score placeholder cuando Azure no está disponible.
        """
        # Score placeholder basado en longitud del texto (simple heuristic)
        score = 75.0
        
        details = {
            'status': status,
            'message': 'Azure Speech Service no disponible',
            'word_scores': [],
            'recognized_text': None
        }
        
        if error:
            details['error'] = error
        
        if status == 'not_configured':
            details['message'] = 'Azure Speech Service no configurado - usando score placeholder'
        elif status == 'no_match':
            details['message'] = 'Azure no pudo reconocer el audio'
            score = 50.0  # Score bajo si no reconoce
        elif status == 'error':
            details['message'] = 'Error en Azure Speech Service'
            score = 70.0
        
        logger.info(f"Usando placeholder score: {score:.1f}")
        return score, details
    
    def get_version(self) -> str:
        """Retorna la versión del servicio"""
        if self.is_configured:
            return self.version
        return "not_configured"