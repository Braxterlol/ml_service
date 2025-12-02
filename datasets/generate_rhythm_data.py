"""
Generador de datos sintéticos para entrenar XGBoost (Ritmo)

Este script genera datos sintéticos basados en reglas expertas
sobre qué características temporales determinan un buen ritmo.

VERSIÓN MEJORADA: Permite 0 pausas en audios cortos (<2.5s)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_rhythm_score(features: Dict) -> float:
    """
    Calcula score de ritmo basado en reglas expertas.
    
    REGLAS MEJORADAS:
    - Audios cortos (<2.5s): No se penalizan por falta de pausas
    - Audios largos (>2.5s): Se evalúan pausas normalmente
    
    Factores principales:
    - Speech rate óptimo (2.5-4.0 sílabas/seg): 35%
    - Pausas (cantidad, densidad, duración): 30%
    - Speaking ratio: 20%
    - Regularidad de pausas: 15%
    """
    score = 100.0
    duration = features['duration_seconds']
    pause_count = features['pause_count']
    
    # Clasificar audio por duración
    is_very_short = duration < 2.5  # Palabras cortas o frases muy breves
    
    # Factor 1: Speech rate (máx -35 puntos)
    if features['speech_rate'] < 1.5 or features['speech_rate'] > 5.5:
        score -= 35  # Muy lento o muy rápido
    elif features['speech_rate'] < 2.0 or features['speech_rate'] > 5.0:
        score -= 20
    elif features['speech_rate'] < 2.5 or features['speech_rate'] > 4.0:
        score -= 10
    
    # Factor 2: Articulación rate (máx -10 puntos)
    if features['articulation_rate'] < 2.0 or features['articulation_rate'] > 6.5:
        score -= 10
    elif features['articulation_rate'] < 2.5 or features['articulation_rate'] > 6.0:
        score -= 5
    
    # Factor 3 y 4: Pausas (solo para audios largos)
    if not is_very_short:
        # Factor 3: Pausas excesivas (máx -15 puntos)
        if pause_count > 8:
            score -= 15
        elif pause_count > 6:
            score -= 10
        elif pause_count > 4:
            score -= 5
        
        # Factor 4: Densidad de pausas (máx -15 puntos)
        if features['pause_density'] > 3.0:
            score -= 15
        elif features['pause_density'] > 2.0:
            score -= 10
        elif features['pause_density'] > 1.5:
            score -= 5
    else:
        # Para audios cortos, solo penalizar pausas extremadamente excesivas
        if pause_count > 5:
            score -= 10
        elif pause_count > 3:
            score -= 5
    
    # Factor 5: Speaking ratio (ajustado por duración)
    if is_very_short:
        # Audios cortos: Aceptar 100% habla (sin pausas está bien)
        if features['speaking_time_ratio'] < 0.70:
            score -= 15
        elif features['speaking_time_ratio'] < 0.85:
            score -= 5
    else:
        # Audios largos: Requieren algunas pausas naturales
        if features['speaking_time_ratio'] < 0.50:
            score -= 20
        elif features['speaking_time_ratio'] < 0.65:
            score -= 10
        elif features['speaking_time_ratio'] < 0.80:
            score -= 5
    
    # Factor 6: Pausas muy largas (máx -10 puntos)
    # Solo aplicar si hay pausas
    if pause_count > 0:
        if features['average_pause_duration'] > 500:
            score -= 10
        elif features['average_pause_duration'] > 350:
            score -= 5
    
    # Factor 7: Regularidad de pausas (máx -15 puntos)
    # Solo aplicar si hay suficientes pausas para evaluar patrón
    if pause_count >= 2:
        if features['pause_pattern_regularity'] < 0.3:
            score -= 15  # Pausas muy irregulares
        elif features['pause_pattern_regularity'] < 0.5:
            score -= 10
        elif features['pause_pattern_regularity'] < 0.7:
            score -= 5
    
    # Bonus: Ritmo excepcional (+5 puntos)
    if (2.5 <= features['speech_rate'] <= 4.0 and
        (is_very_short or (pause_count <= 4 and features['speaking_time_ratio'] > 0.82)) and
        (pause_count < 2 or features['pause_pattern_regularity'] > 0.75)):
        score += 5
    
    return max(0, min(100, score))


def generate_sample(score_range: Tuple[int, int]) -> Dict:
    """
    Genera una muestra sintética para un rango de score específico.
    
    Args:
        score_range: Tupla (min_score, max_score)
    
    Returns:
        Dict con features y score
    """
    min_score, max_score = score_range
    
    # Duración del audio (incluyendo audios muy cortos)
    duration_seconds = np.random.uniform(1.0, 8.0)
    is_very_short = duration_seconds < 2.5
    
    if min_score >= 90:
        # Excelente ritmo
        speech_rate = np.random.uniform(2.5, 4.0)
        articulation_rate = np.random.uniform(3.0, 5.0)
        
        if is_very_short:
            # Audios cortos: 0-2 pausas es perfectamente válido
            pause_count = np.random.randint(0, 3)
        else:
            pause_count = np.random.randint(2, 5)
        
        pause_density = pause_count / duration_seconds if duration_seconds > 0 else 0
        
        if pause_count == 0:
            speaking_ratio = 1.0  # 100% habla
            avg_pause = 0.0
            pause_duration_std = 0.0
            regularity = 1.0
        else:
            speaking_ratio = np.random.uniform(0.80, 0.95)
            avg_pause = np.random.uniform(180, 280)
            regularity = np.random.uniform(0.7, 1.0)
            pause_duration_std = avg_pause * (1.0 - regularity) * np.random.uniform(0.2, 0.4)
        
    elif min_score >= 70:
        # Buen ritmo
        speech_rate = np.random.uniform(2.0, 5.0)
        articulation_rate = np.random.uniform(2.5, 6.0)
        
        if is_very_short:
            pause_count = np.random.randint(0, 4)
        else:
            pause_count = np.random.randint(1, 7)
        
        pause_density = pause_count / duration_seconds if duration_seconds > 0 else 0
        
        if pause_count == 0:
            speaking_ratio = 1.0
            avg_pause = 0.0
            pause_duration_std = 0.0
            regularity = 1.0
        else:
            speaking_ratio = np.random.uniform(0.65, 0.90)
            avg_pause = np.random.uniform(150, 350)
            regularity = np.random.uniform(0.5, 0.8)
            pause_duration_std = avg_pause * (1.0 - regularity) * np.random.uniform(0.3, 0.5)
        
    elif min_score >= 50:
        # Regular
        if np.random.random() < 0.5:
            speech_rate = np.random.uniform(1.5, 2.5)  # Muy lento
        else:
            speech_rate = np.random.uniform(4.5, 5.5)  # Muy rápido
        
        articulation_rate = np.random.uniform(2.0, 6.5)
        
        if is_very_short:
            pause_count = np.random.randint(0, 6)
        else:
            pause_count = np.random.randint(4, 9)
        
        pause_density = pause_count / duration_seconds if duration_seconds > 0 else 0
        
        if pause_count == 0:
            speaking_ratio = 1.0
            avg_pause = 0.0
            pause_duration_std = 0.0
            regularity = 1.0
        else:
            speaking_ratio = np.random.uniform(0.50, 0.75)
            avg_pause = np.random.uniform(300, 450)
            regularity = np.random.uniform(0.3, 0.6)
            pause_duration_std = avg_pause * (1.0 - regularity) * np.random.uniform(0.4, 0.6)
        
    else:
        # Necesita mejorar
        if np.random.random() < 0.5:
            speech_rate = np.random.uniform(0.8, 1.5)  # Extremadamente lento
        else:
            speech_rate = np.random.uniform(5.5, 6.5)  # Extremadamente rápido
        
        articulation_rate = np.random.uniform(1.5, 7.0)
        
        if is_very_short:
            # Incluso audios cortos pueden tener mal ritmo por pausas excesivas
            pause_count = np.random.randint(3, 8)
        else:
            pause_count = np.random.randint(8, 12)
        
        pause_density = pause_count / duration_seconds if duration_seconds > 0 else 0
        
        if pause_count == 0:
            speaking_ratio = 1.0
            avg_pause = 0.0
            pause_duration_std = 0.0
            regularity = 1.0
        else:
            speaking_ratio = np.random.uniform(0.35, 0.60)
            avg_pause = np.random.uniform(400, 600)
            regularity = np.random.uniform(0.0, 0.4)
            pause_duration_std = avg_pause * (1.0 - regularity) * np.random.uniform(0.5, 0.8)
    
    features = {
        'speech_rate': speech_rate,
        'articulation_rate': articulation_rate,
        'pause_count': pause_count,
        'pause_density': pause_density,
        'speaking_time_ratio': speaking_ratio,
        'average_pause_duration': avg_pause,
        'pause_duration_std': pause_duration_std,
        'duration_seconds': duration_seconds,
        'pause_pattern_regularity': regularity
    }
    
    # Calcular score basado en las reglas
    score = calculate_rhythm_score(features)
    
    # Añadir ruido al score (±3 puntos para mayor variabilidad)
    score += np.random.uniform(-3, 3)
    score = max(0, min(100, score))
    
    features['rhythm_score'] = score
    
    return features


def generate_rhythm_dataset(n_samples: int = 3000, output_file: str = 'rhythm_training_data.csv'):
    """
    Genera dataset completo de rhythm.
    
    Args:
        n_samples: Número de muestras a generar
        output_file: Archivo de salida CSV
    """
    print(f"Generando {n_samples} muestras sintéticas de ritmo...")
    print("✨ VERSIÓN MEJORADA: Soporta audios cortos sin pausas")
    
    # Distribuir muestras entre rangos de score
    score_ranges = [
        (90, 100, int(n_samples * 0.20)),  # 20% excelente
        (70, 89, int(n_samples * 0.40)),   # 40% bueno
        (50, 69, int(n_samples * 0.25)),   # 25% regular
        (0, 49, int(n_samples * 0.15))     # 15% necesita mejorar
    ]
    
    samples = []
    
    for min_score, max_score, count in score_ranges:
        print(f"  Generando {count} muestras con score {min_score}-{max_score}...")
        for _ in range(count):
            sample = generate_sample((min_score, max_score))
            samples.append(sample)
    
    # Crear DataFrame
    df = pd.DataFrame(samples)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Guardar
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Dataset guardado en: {output_file}")
    print(f"   Total muestras: {len(df)}")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Target: rhythm_score")
    print(f"\nEstadísticas del score:")
    print(df['rhythm_score'].describe())
    print(f"\nDistribución:")
    print(f"  Excelente (90-100): {len(df[df['rhythm_score'] >= 90])}")
    print(f"  Bueno (70-89): {len(df[(df['rhythm_score'] >= 70) & (df['rhythm_score'] < 90)])}")
    print(f"  Regular (50-69): {len(df[(df['rhythm_score'] >= 50) & (df['rhythm_score'] < 70)])}")
    print(f"  Necesita mejorar (0-49): {len(df[df['rhythm_score'] < 50])}")
    
    # Estadísticas adicionales
    print(f"\nEstadísticas de pausas:")
    print(f"  Muestras sin pausas (0): {len(df[df['pause_count'] == 0])} ({len(df[df['pause_count'] == 0])/len(df)*100:.1f}%)")
    print(f"  Muestras con 1-2 pausas: {len(df[(df['pause_count'] >= 1) & (df['pause_count'] <= 2)])} ({len(df[(df['pause_count'] >= 1) & (df['pause_count'] <= 2)])/len(df)*100:.1f}%)")
    print(f"  Muestras con 3+ pausas: {len(df[df['pause_count'] >= 3])} ({len(df[df['pause_count'] >= 3])/len(df)*100:.1f}%)")
    
    print(f"\nEstadísticas de duración:")
    print(f"  Audios cortos (<2.5s): {len(df[df['duration_seconds'] < 2.5])} ({len(df[df['duration_seconds'] < 2.5])/len(df)*100:.1f}%)")
    print(f"  Audios largos (>=2.5s): {len(df[df['duration_seconds'] >= 2.5])} ({len(df[df['duration_seconds'] >= 2.5])/len(df)*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    # Generar dataset
    df = generate_rhythm_dataset(n_samples=3000, output_file='rhythm_training_data.csv')
    
    # Mostrar ejemplos específicos
    print("\n" + "="*60)
    print("EJEMPLOS DE MUESTRAS GENERADAS:")
    print("="*60)
    
    # Ejemplo 1: Audio corto sin pausas con buen score
    short_no_pause = df[(df['duration_seconds'] < 2.5) & (df['pause_count'] == 0) & (df['rhythm_score'] >= 80)]
    if len(short_no_pause) > 0:
        print("\n📝 Ejemplo: Audio corto SIN pausas (Score alto)")
        print(short_no_pause.iloc[0])
    
    # Ejemplo 2: Audio largo con pausas balanceadas
    long_balanced = df[(df['duration_seconds'] >= 4.0) & (df['pause_count'] >= 3) & (df['rhythm_score'] >= 80)]
    if len(long_balanced) > 0:
        print("\n📝 Ejemplo: Audio largo CON pausas balanceadas (Score alto)")
        print(long_balanced.iloc[0])
    
    # Ejemplo 3: Audio con mal ritmo
    bad_rhythm = df[df['rhythm_score'] < 40]
    if len(bad_rhythm) > 0:
        print("\n📝 Ejemplo: Audio con mal ritmo (Score bajo)")
        print(bad_rhythm.iloc[0])