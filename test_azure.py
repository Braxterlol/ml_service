"""
Test Azure Speech Service

Script para probar que Azure está configurado correctamente.
"""

import requests
import json
import base64

# Configuración
ML_SERVICE_URL = "http://localhost:8002"
API_KEY = "edier_mampito"

# Audio de prueba (muy simple - solo para testing)
# Este es un audio WAV vacío válido (44 bytes)
SAMPLE_AUDIO = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="

def test_azure_pronunciation():
    """Prueba el endpoint de análisis con Azure"""
    
    print("🧪 Testing Azure Pronunciation Assessment")
    print("="*60)
    
    # Request
    payload = {
        "attempt_id": "test-azure-123",
        "user_id": "test-user",
        "exercise_id": "test-exercise",
        "audio_features": {
            "prosody": {
                "jitter": 0.02,
                "shimmer": 0.08,
                "f0_stats": {"mean": 125, "std": 35, "range": 50},
                "energy_stats": {"mean": 0.5, "std": 0.3}
            },
            "rhythm": {
                "speech_rate": 3.2,
                "articulation_rate": 3.8,
                "pause_count": 3,
                "pause_durations_ms": [250, 180, 320],
                "speaking_time_ms": 2840,
                "total_duration_ms": 3590
            },
            "duration_seconds": 3.59
        },
        "reference_text": "rata rana rojo",
        "audio_base64": SAMPLE_AUDIO
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("\n📤 Enviando request a ML Service...")
    print(f"   URL: {ML_SERVICE_URL}/api/v1/ml/analyze")
    print(f"   Reference text: '{payload['reference_text']}'")
    
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/api/v1/ml/analyze",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Response exitoso!")
            print("="*60)
            print(f"\n📊 Scores:")
            print(f"   Pronunciation: {result['scores'].get('pronunciation', 'N/A')}/100")
            print(f"   Fluency:       {result['scores'].get('fluency', 'N/A')}/100")
            print(f"   Rhythm:        {result['scores'].get('rhythm', 'N/A')}/100")
            print(f"   Overall:       {result['scores'].get('overall', 'N/A')}/100")
            
            print(f"\n🔧 Model Versions:")
            versions = result['model_versions']
            print(f"   Random Forest: {versions.get('random_forest', 'N/A')}")
            print(f"   XGBoost:       {versions.get('xgboost', 'N/A')}")
            print(f"   Azure:         {versions.get('azure', 'N/A')}")
            
            print(f"\n⏱️  Processing Time:")
            timing = result['processing_info']
            print(f"   Prediction:    {timing.get('prediction_time_ms', 0)}ms")
            print(f"   Azure API:     {timing.get('azure_api_time_ms', 0)}ms")
            print(f"   Total:         {timing.get('total_time_ms', 0)}ms")
            
            # Verificar si Azure funcionó
            azure_version = versions.get('azure', 'not_used')
            pronunciation = result['scores'].get('pronunciation')
            
            if azure_version == 'not_configured' or pronunciation is None:
                print("\n⚠️  Azure NO está configurado correctamente")
                print("   Verifica:")
                print("   1. AZURE_SPEECH_KEY en .env")
                print("   2. AZURE_SPEECH_REGION en .env")
                print("   3. Reiniciaste el servicio después de configurar")
            elif azure_version == '2024-11-01' and pronunciation is not None:
                print("\n🎉 ¡Azure está funcionando correctamente!")
                print(f"   Pronunciation score: {pronunciation}/100")
            else:
                print(f"\n⚠️  Estado de Azure: {azure_version}")
            
            print("\n" + "="*60)
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ No se pudo conectar al ML Service")
        print("   Verifica que el servicio esté corriendo en http://localhost:8002")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_models_info():
    """Prueba el endpoint de información de modelos"""
    
    print("\n\n🔍 Testing Models Info Endpoint")
    print("="*60)
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(
            f"{ML_SERVICE_URL}/api/v1/ml/models/info",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Models Info:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   {response.text}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AZURE SPEECH SERVICE - TESTING")
    print("="*60)
    
    # Test 1: Análisis completo
    test_azure_pronunciation()
    
    # Test 2: Info de modelos
    test_models_info()
    
    print("\n✅ Testing completado\n")