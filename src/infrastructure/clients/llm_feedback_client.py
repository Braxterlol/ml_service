import httpx
from typing import Dict, Optional, List, Any
from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    """Request para generar feedback"""
    attempt_id: str
    user_id: str
    exercise_id: str
    pronunciation_score: float
    fluency_score: float
    rhythm_score: float
    overall_score: float
    exercise_type: str
    exercise_content: str
    difficulty_level: int
    reference_text: str
    user_age: Optional[int] = None
    attempt_number: int = 1
    passed: bool
    stars_earned: int
    unlocked_next: bool
    previous_best_score: Optional[float] = None


class FeedbackResponse(BaseModel):
    """Response del feedback generado"""
    main_message: str
    strengths: List[str]
    areas_to_improve: List[str]
    specific_tip: str
    celebration: Optional[str]
    encouragement: str
    tone: str


class LLMFeedbackClient:
    """
    Cliente HTTP para comunicarse con el LLM Feedback Service.
    """
    
    def __init__(self, base_url: str = "http://3.236.227.171:8003"):
        """
        Inicializa el cliente.
        
        Args:
            base_url: URL base del LLM Feedback Service
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = 30.0  # 30 segundos timeout
    
    async def generate_feedback(
        self,
        attempt_id: str,
        user_id: str,
        exercise_id: str,
        scores: Dict[str, float],
        exercise_info: Dict[str, Any],
        progression_info: Dict[str, Any],
        user_age: Optional[int] = None,
        attempt_number: int = 1
    ) -> Optional[FeedbackResponse]:
        """
        Genera feedback personalizado para un intento.
        
        Args:
            attempt_id: ID del intento
            user_id: ID del usuario
            exercise_id: ID del ejercicio
            scores: Dict con pronunciation_score, fluency_score, rhythm_score, overall_score
            exercise_info: Dict con exercise_type, exercise_content, difficulty_level, reference_text
            progression_info: Dict con passed, stars_earned, unlocked_next, previous_best_score
            user_age: Edad del usuario (opcional)
            attempt_number: Número de intento
        
        Returns:
            FeedbackResponse si tuvo éxito, None si falló
        """
        try:
            # Construir request
            request_data = FeedbackRequest(
                attempt_id=attempt_id,
                user_id=user_id,
                exercise_id=exercise_id,
                pronunciation_score=scores["pronunciation_score"],
                fluency_score=scores["fluency_score"],
                rhythm_score=scores["rhythm_score"],
                overall_score=scores["overall_score"],
                exercise_type=exercise_info["exercise_type"],
                exercise_content=exercise_info["exercise_content"],
                difficulty_level=exercise_info["difficulty_level"],
                reference_text=exercise_info["reference_text"],
                user_age=user_age,
                attempt_number=attempt_number,
                passed=progression_info["passed"],
                stars_earned=progression_info["stars_earned"],
                unlocked_next=progression_info["unlocked_next"],
                previous_best_score=progression_info.get("previous_best_score")
            )
            
            # Llamar al servicio
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/feedback/generate",
                    json=request_data.model_dump()
                )
                
                response.raise_for_status()
                
                # Parsear response
                feedback_data = response.json()
                return FeedbackResponse(**feedback_data)
                
        except httpx.TimeoutException:
            print(f"⚠️ Timeout llamando a LLM Feedback Service")
            return None
        except httpx.HTTPStatusError as e:
            print(f"⚠️ Error HTTP {e.response.status_code} del LLM Feedback Service: {e.response.text}")
            return None
        except Exception as e:
            print(f"⚠️ Error inesperado llamando a LLM Feedback Service: {e}")
            return None
    
    async def health_check(self) -> bool:
        """
        Verifica si el LLM Feedback Service está disponible.
        
        Returns:
            bool: True si está disponible
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/feedback/health")
                return response.status_code == 200
        except:
            return False


# Ejemplo de uso
"""
# En tu ML Service, después de calcular scores:

from llm_feedback_client import LLMFeedbackClient

feedback_client = LLMFeedbackClient(base_url="http://3.236.227.171:8003")

# Después de analizar el audio
scores = {
    "pronunciation_score": 85.5,
    "fluency_score": 78.2,
    "rhythm_score": 92.0,
    "overall_score": 85.2
}

exercise_info = {
    "exercise_type": "fonema",
    "exercise_content": "palabras con /r/ suave",
    "difficulty_level": 2,
    "reference_text": "raro, caro, pera, coro"
}

progression_info = {
    "passed": True,
    "stars_earned": 2,
    "unlocked_next": True,
    "previous_best_score": 78.0
}

feedback = await feedback_client.generate_feedback(
    attempt_id="attempt-123",
    user_id="user-456",
    exercise_id="fonema_r_1",
    scores=scores,
    exercise_info=exercise_info,
    progression_info=progression_info,
    user_age=7,
    attempt_number=3
)

if feedback:
    print(f"Feedback: {feedback.main_message}")
    print(f"Fortalezas: {feedback.strengths}")
    print(f"Tip: {feedback.specific_tip}")
else:
    print("No se pudo generar feedback")
"""