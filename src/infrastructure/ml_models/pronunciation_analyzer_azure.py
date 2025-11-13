"""
Pronunciation Analyzer usando Azure Speech Service

Este módulo usa Azure Cognitive Services Speech SDK
para evaluar pronunciación de niños.
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
            # Decodificar audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Configurar audio
            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # Configurar pronunciation assessment
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                enable_miscue=True
            )
            
            # Configurar idioma
            self.speech_config.speech_recognition_language = language
            
            # Crear recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Aplicar pronunciation assessment al recognizer
            pronunciation_config.apply_to(speech_recognizer)
            
            # Escribir audio al stream
            audio_stream.write(audio_bytes)
            audio_stream.close()
            
            # Reconocer
            result = speech_recognizer.recognize_once()
            
            # Procesar resultado
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
                
                # Extraer scores
                accuracy_score = pronunciation_result.accuracy_score
                fluency_score = pronunciation_result.fluency_score
                completeness_score = pronunciation_result.completeness_score
                pronunciation_score = pronunciation_result.pronunciation_score
                
                # Extraer detalles por palabra
                word_scores = []
                try:
                    import json
                    details = json.loads(result.properties.get(
                        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                    ))
                    
                    if 'NBest' in details and len(details['NBest']) > 0:
                        words = details['NBest'][0].get('Words', [])
                        for word_data in words:
                            word_scores.append({
                                'word': word_data.get('Word', ''),
                                'accuracy_score': word_data.get('PronunciationAssessment', {}).get('AccuracyScore', 0),
                                'error_type': word_data.get('PronunciationAssessment', {}).get('ErrorType', 'None')
                            })
                except Exception as e:
                    logger.warning(f"No se pudieron extraer detalles de palabras: {e}")
                
                logger.info(f"✅ Azure pronunciation score: {pronunciation_score:.1f}")
                
                return pronunciation_score, {
                    'status': 'success',
                    'accuracy_score': accuracy_score,
                    'fluency_score': fluency_score,
                    'completeness_score': completeness_score,
                    'word_scores': word_scores,
                    'recognized_text': result.text
                }
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("Azure no pudo reconocer el audio")
                return self._placeholder_score(reference_text, status='no_match')
            
            else:
                logger.error(f"Error en Azure: {result.reason}")
                return self._placeholder_score(reference_text, status='error')
        
        except Exception as e:
            logger.error(f"❌ Error en Azure pronunciation assessment: {e}")
            return self._placeholder_score(reference_text, status='exception', error=str(e))
    
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