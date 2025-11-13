"""
Testing de Modelos ML con Datos Reales de MongoDB

Este script:
1. Lee AudioFeatures desde MongoDB
2. Extrae los features necesarios para RF y XGB
3. Predice scores con los modelos entrenados
4. Muestra los resultados para validación manual
"""

import joblib
import certifi
import numpy as np
import pandas as pd
from pymongo import MongoClient
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelTester:
    """Tester para validar modelos con datos reales"""
    
    def __init__(
        self,
        fluency_model_path: str = 'models/fluency_rf_model.pkl',
        fluency_scaler_path: str = 'models/fluency_scaler.pkl',
        rhythm_model_path: str = 'models/rhythm_xgb_model.pkl',
        rhythm_scaler_path: str = 'models/rhythm_scaler.pkl'
    ):
        """
        Carga los modelos entrenados.
        
        Args:
            fluency_model_path: Path al modelo Random Forest
            fluency_scaler_path: Path al scaler de fluidez
            rhythm_model_path: Path al modelo XGBoost
            rhythm_scaler_path: Path al scaler de ritmo
        """
        print("🔄 Cargando modelos entrenados...")
        
        # Cargar Random Forest (Fluidez)
        self.fluency_model = joblib.load(fluency_model_path)
        self.fluency_scaler = joblib.load(fluency_scaler_path)
        print(f"   ✅ Random Forest cargado")
        
        # Cargar XGBoost (Ritmo)
        self.rhythm_model = joblib.load(rhythm_model_path)
        self.rhythm_scaler = joblib.load(rhythm_scaler_path)
        print(f"   ✅ XGBoost cargado")
    
    def extract_fluency_features(self, audio_doc: Dict) -> Optional[np.ndarray]:
        """
        Extrae features de fluidez desde un documento de MongoDB.
        
        Features necesarios (11):
        - jitter, shimmer
        - pause_count, pause_duration_mean, pause_duration_std
        - speaking_time_ratio
        - f0_std, f0_range
        - energy_std
        - speech_rate_normalized, articulation_rate_normalized
        """
        try:
            prosody = audio_doc.get('prosody', {})
            rhythm = audio_doc.get('rhythm', {})
            duration = audio_doc.get('duration_seconds', 1.0)
            
            # Extraer features básicos
            jitter = prosody.get('jitter', 0.03)
            shimmer = prosody.get('shimmer', 0.10)
            
            pause_count = rhythm.get('pause_count', 0)
            pause_durations = rhythm.get('pause_durations_ms', [])
            
            if pause_durations and len(pause_durations) > 0:
                pause_duration_mean = np.mean(pause_durations)
                pause_duration_std = np.std(pause_durations)
            else:
                pause_duration_mean = 0
                pause_duration_std = 0
            
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
            
            # Speech rates normalizados
            speech_rate = rhythm.get('speech_rate', 3.0)
            articulation_rate = rhythm.get('articulation_rate', 3.5)
            speech_rate_normalized = speech_rate / duration if duration > 0 else speech_rate
            articulation_rate_normalized = articulation_rate / duration if duration > 0 else articulation_rate
            
            # Crear array de features
            features = np.array([[
                jitter,
                shimmer,
                pause_count,
                pause_duration_mean,
                pause_duration_std,
                speaking_time_ratio,
                f0_std,
                f0_range,
                energy_std,
                speech_rate_normalized,
                articulation_rate_normalized
            ]])
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ Error extrayendo features de fluidez: {e}")
            return None
    
    def extract_rhythm_features(self, audio_doc: Dict) -> Optional[np.ndarray]:
        """
        Extrae features de ritmo desde un documento de MongoDB.
        
        Features necesarios (9):
        - speech_rate, articulation_rate
        - pause_count, pause_density
        - speaking_time_ratio
        - average_pause_duration, pause_duration_std
        - duration_seconds
        - pause_pattern_regularity
        """
        try:
            rhythm = audio_doc.get('rhythm', {})
            duration = audio_doc.get('duration_seconds', 1.0)
            
            # Features básicos
            speech_rate = rhythm.get('speech_rate', 3.0)
            articulation_rate = rhythm.get('articulation_rate', 3.5)
            pause_count = rhythm.get('pause_count', 0)
            
            # Pause density (pausas por segundo)
            pause_density = pause_count / duration if duration > 0 else 0
            
            # Speaking time ratio
            speaking_time_ms = rhythm.get('speaking_time_ms', duration * 1000)
            total_duration_ms = rhythm.get('total_duration_ms', duration * 1000)
            speaking_time_ratio = speaking_time_ms / total_duration_ms if total_duration_ms > 0 else 0.8
            
            # Pause durations
            pause_durations = rhythm.get('pause_durations_ms', [])
            if pause_durations and len(pause_durations) > 0:
                average_pause_duration = np.mean(pause_durations)
                pause_duration_std = np.std(pause_durations)
                
                # Calcular regularidad de pausas
                # Regularidad = 1 - CV (coeficiente de variación)
                cv = pause_duration_std / average_pause_duration if average_pause_duration > 0 else 1.0
                pause_pattern_regularity = max(0, 1.0 - cv)
            else:
                average_pause_duration = 0
                pause_duration_std = 0
                pause_pattern_regularity = 1.0
            
            # Crear array de features
            features = np.array([[
                speech_rate,
                articulation_rate,
                pause_count,
                pause_density,
                speaking_time_ratio,
                average_pause_duration,
                pause_duration_std,
                duration,
                pause_pattern_regularity
            ]])
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ Error extrayendo features de ritmo: {e}")
            return None
    
    def predict_fluency(self, features: np.ndarray) -> float:
        """Predice score de fluidez"""
        features_scaled = self.fluency_scaler.transform(features)
        score = self.fluency_model.predict(features_scaled)[0]
        return float(score)
    
    def predict_rhythm(self, features: np.ndarray) -> float:
        """Predice score de ritmo"""
        features_scaled = self.rhythm_scaler.transform(features)
        score = self.rhythm_model.predict(features_scaled)[0]
        return float(score)
    
    def analyze_audio(self, audio_doc: Dict) -> Dict:
        """
        Analiza un audio completo (fluidez + ritmo).
        
        Returns:
            Dict con scores y detalles
        """
        attempt_id = audio_doc.get('attempt_id', 'unknown')
        
        print(f"\n📊 Analizando attempt: {attempt_id}")
        
        # Extraer y predecir fluidez
        fluency_features = self.extract_fluency_features(audio_doc)
        if fluency_features is not None:
            fluency_score = self.predict_fluency(fluency_features)
            print(f"   🎵 Fluidez: {fluency_score:.1f}/100")
        else:
            fluency_score = None
            print(f"   ⚠️ No se pudo calcular fluidez")
        
        # Extraer y predecir ritmo
        rhythm_features = self.extract_rhythm_features(audio_doc)
        if rhythm_features is not None:
            rhythm_score = self.predict_rhythm(rhythm_features)
            print(f"   ⏱️  Ritmo: {rhythm_score:.1f}/100")
        else:
            rhythm_score = None
            print(f"   ⚠️ No se pudo calcular ritmo")
        
        # Calcular overall (sin pronunciation todavía, será 0)
        if fluency_score and rhythm_score:
            # Por ahora solo promedio de fluidez y ritmo (sin pronunciation)
            overall_score = (fluency_score * 0.4 + rhythm_score * 0.4)  # Falta 20% de pronunciation
            print(f"   📈 Overall (parcial): {overall_score:.1f}/100")
        else:
            overall_score = None
        
        return {
            'attempt_id': attempt_id,
            'fluency_score': fluency_score,
            'rhythm_score': rhythm_score,
            'overall_score': overall_score,
            'pronunciation_score': None  # Pendiente: Azure
        }


def connect_to_mongodb(mongodb_url: str = 'mongodb+srv://223221_db_user:yiUvhHklDudvXHKm@vocalis-cluster.astawly.mongodb.net/audio_features_db', db_name: str = 'audio_features_db'):
    """Conecta a MongoDB"""
    print(f"🔌 Conectando a MongoDB...")
    print(f"   URL: {mongodb_url}")
    print(f"   Database: {db_name}")
    
    try:
        client = MongoClient(mongodb_url, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        db = client[db_name]
        print(f"   ✅ Conectado exitosamente")
        return db
    except Exception as e:
        print(f"   ❌ Error conectando a MongoDB: {e}")
        print("\n💡 Asegúrate de que:")
        print("   1. MongoDB está corriendo")
        print("   2. La URL es correcta")
        print("   3. El nombre de la base de datos es correcto")
        return None


def main():
    """Pipeline principal de testing"""
    print("="*70)
    print("  TESTING DE MODELOS ML CON DATOS REALES")
    print("="*70)
    
    # Configuración (ajusta según tu setup)
    MONGODB_URL = 'mongodb+srv://223221_db_user:yiUvhHklDudvXHKm@vocalis-cluster.astawly.mongodb.net/audio_features_db'
    DB_NAME = 'audio_features_db'
    COLLECTION_NAME = 'audio_features'
    
    # 1. Conectar a MongoDB
    db = connect_to_mongodb(MONGODB_URL, DB_NAME)
    if db is None:
        print("\n❌ No se pudo conectar a MongoDB. Terminando.")
        return
    
    collection = db[COLLECTION_NAME]
    
    # 2. Verificar que hay datos
    count = collection.count_documents({})
    print(f"\n📊 Documentos en '{COLLECTION_NAME}': {count}")
    
    if count == 0:
        print("\n⚠️ No hay documentos en MongoDB.")
        print("   Ejecuta primero el endpoint POST /audio/process para generar datos.")
        return
    
    # 3. Cargar modelos
    tester = ModelTester()
    
    # 4. Obtener algunos documentos para testing
    print(f"\n🔍 Obteniendo hasta 5 documentos para testing...")
    docs = list(collection.find().limit(5))
    
    if not docs:
        print("   ⚠️ No se encontraron documentos")
        return
    
    print(f"   ✅ {len(docs)} documentos obtenidos")
    
    # 5. Analizar cada documento
    results = []
    
    print("\n" + "="*70)
    print("  RESULTADOS DE PREDICCIONES")
    print("="*70)
    
    for i, doc in enumerate(docs, 1):
        print(f"\n{'='*70}")
        print(f"  AUDIO #{i}")
        print(f"{'='*70}")
        
        # Mostrar info del audio
        print(f"📝 Información:")
        print(f"   Attempt ID: {doc.get('attempt_id', 'N/A')}")
        print(f"   User ID: {doc.get('user_id', 'N/A')}")
        print(f"   Exercise ID: {doc.get('exercise_id', 'N/A')}")
        print(f"   Duration: {doc.get('duration_seconds', 0):.2f}s")
        
        # Analizar
        result = tester.analyze_audio(doc)
        results.append(result)
    
    # 6. Resumen final
    print("\n" + "="*70)
    print("  RESUMEN DE RESULTADOS")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    print("\n📊 Estadísticas de Scores:")
    if 'fluency_score' in df.columns and df['fluency_score'].notna().any():
        print(f"\n   Fluidez:")
        print(f"      Promedio: {df['fluency_score'].mean():.1f}")
        print(f"      Mínimo: {df['fluency_score'].min():.1f}")
        print(f"      Máximo: {df['fluency_score'].max():.1f}")
    
    if 'rhythm_score' in df.columns and df['rhythm_score'].notna().any():
        print(f"\n   Ritmo:")
        print(f"      Promedio: {df['rhythm_score'].mean():.1f}")
        print(f"      Mínimo: {df['rhythm_score'].min():.1f}")
        print(f"      Máximo: {df['rhythm_score'].max():.1f}")
    
    # 7. Guardar resultados
    output_file = 'test_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Resultados guardados en: {output_file}")
    
    print("\n" + "="*70)
    print("  ✅ TESTING COMPLETADO")
    print("="*70)
    
    print("\n💡 Próximos pasos:")
    print("   1. Revisar los scores generados")
    print("   2. Validar si los scores tienen sentido para cada audio")
    print("   3. Si los scores son razonables, proceder a crear el ML Service")
    print("   4. Integrar Azure Speech para pronunciation scores")


if __name__ == "__main__":
    main()