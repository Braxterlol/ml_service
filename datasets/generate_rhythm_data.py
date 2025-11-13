"""
Generador de datos sintéticos para entrenar XGBoost (Ritmo)

Este script genera datos sintéticos basados en reglas expertas
sobre qué características temporales determinan un buen ritmo.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_rhythm_score(features: Dict) -> float:
    """
    Calcula score de ritmo basado en reglas expertas.
    
    Factores principales:
    - Speech rate óptimo (2.5-4.0 sílabas/seg): 35%
    - Pausas (cantidad, densidad, duración): 30%
    - Speaking ratio: 20%
    - Regularidad de pausas: 15%
    """
    score = 100.0
    
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
    
    # Factor 3: Pausas excesivas (máx -15 puntos)
    if features['pause_count'] > 8:
        score -= 15
    elif features['pause_count'] > 6:
        score -= 10
    elif features['pause_count'] > 4:
        score -= 5
    
    # Factor 4: Densidad de pausas (máx -15 puntos)
    if features['pause_density'] > 3.0:
        score -= 15
    elif features['pause_density'] > 2.0:
        score -= 10
    elif features['pause_density'] > 1.5:
        score -= 5
    
    # Factor 5: Speaking ratio (máx -20 puntos)
    if features['speaking_time_ratio'] < 0.50:
        score -= 20
    elif features['speaking_time_ratio'] < 0.65:
        score -= 10
    elif features['speaking_time_ratio'] < 0.80:
        score -= 5
    
    # Factor 6: Pausas muy largas (máx -10 puntos)
    if features['average_pause_duration'] > 500:
        score -= 10
    elif features['average_pause_duration'] > 350:
        score -= 5
    
    # Factor 7: Regularidad de pausas (máx -15 puntos)
    if features['pause_pattern_regularity'] < 0.3:
        score -= 15  # Pausas muy irregulares
    elif features['pause_pattern_regularity'] < 0.5:
        score -= 10
    elif features['pause_pattern_regularity'] < 0.7:
        score -= 5
    
    # Bonus: Ritmo excepcional (+5 puntos)
    if (2.5 <= features['speech_rate'] <= 4.0 and
        features['pause_count'] <= 4 and
        features['speaking_time_ratio'] > 0.82 and
        features['pause_pattern_regularity'] > 0.75):
        score += 5
    
    # Penalización por duración muy corta (difícil evaluar ritmo)
    if features['duration_seconds'] < 1.0:
        score *= 0.9
    
    return max(0, min(100, score))


def calculate_pause_pattern_regularity(pause_durations: list, pause_count: int) -> float:
    """
    Calcula qué tan regulares son las pausas.
    
    Returns:
        float: 0-1, donde 1 = pausas muy regulares
    """
    if pause_count <= 1:
        return 1.0  # Con 0-1 pausas, no hay patrón que evaluar
    
    # Calcular desviación estándar de las duraciones
    std = np.std(pause_durations)
    mean = np.mean(pause_durations)
    
    # Coeficiente de variación
    cv = std / mean if mean > 0 else 1.0
    
    # Mapear CV a regularidad (0-1)
    # CV bajo = alta regularidad
    regularity = max(0, 1.0 - cv)
    
    return regularity


def generate_sample(score_range: Tuple[int, int]) -> Dict:
    """
    Genera una muestra sintética para un rango de score específico.
    
    Args:
        score_range: Tupla (min_score, max_score)
    
    Returns:
        Dict con features y score
    """
    min_score, max_score = score_range
    
    # Duración del audio
    duration_seconds = np.random.uniform(2.0, 8.0)
    
    if min_score >= 90:
        # Excelente ritmo
        speech_rate = np.random.uniform(2.5, 4.0)
        articulation_rate = np.random.uniform(3.0, 5.0)
        pause_count = np.random.randint(2, 5)
        pause_density = np.random.uniform(0.5, 1.5)
        speaking_ratio = np.random.uniform(0.80, 0.90)
        avg_pause = np.random.uniform(180, 280)
        regularity = np.random.uniform(0.7, 1.0)
        
    elif min_score >= 70:
        # Buen ritmo
        speech_rate = np.random.uniform(2.0, 5.0)
        articulation_rate = np.random.uniform(2.5, 6.0)
        pause_count = np.random.randint(1, 7)
        pause_density = np.random.uniform(0.3, 2.0)
        speaking_ratio = np.random.uniform(0.65, 0.85)
        avg_pause = np.random.uniform(150, 350)
        regularity = np.random.uniform(0.5, 0.8)
        
    elif min_score >= 50:
        # Regular
        # Puede ser muy lento, muy rápido, o muchas pausas
        if np.random.random() < 0.5:
            speech_rate = np.random.uniform(1.5, 2.5)  # Muy lento
        else:
            speech_rate = np.random.uniform(4.5, 5.5)  # Muy rápido
        
        articulation_rate = np.random.uniform(2.0, 6.5)
        pause_count = np.random.randint(5, 9)
        pause_density = np.random.uniform(1.5, 3.0)
        speaking_ratio = np.random.uniform(0.50, 0.70)
        avg_pause = np.random.uniform(300, 450)
        regularity = np.random.uniform(0.3, 0.6)
        
    else:
        # Necesita mejorar
        # Extremos en speech rate
        if np.random.random() < 0.5:
            speech_rate = np.random.uniform(0.8, 1.5)  # Extremadamente lento
        else:
            speech_rate = np.random.uniform(5.5, 6.5)  # Extremadamente rápido
        
        articulation_rate = np.random.uniform(1.5, 7.0)
        pause_count = np.random.randint(8, 11)
        pause_density = np.random.uniform(2.5, 4.0)
        speaking_ratio = np.random.uniform(0.35, 0.55)
        avg_pause = np.random.uniform(400, 600)
        regularity = np.random.uniform(0.0, 0.4)
    
    # Calcular pause_duration_std basado en regularidad
    # Baja regularidad = alta desviación
    pause_duration_std = avg_pause * (1.0 - regularity) * np.random.uniform(0.3, 0.6)
    
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
    
    # Añadir ruido al score (±5 puntos)
    score += np.random.uniform(-5, 5)
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
    
    return df


if __name__ == "__main__":
    # Generar dataset
    df = generate_rhythm_dataset(n_samples=3000, output_file='rhythm_training_data.csv')
    
    # Mostrar primeras filas
    print("\nPrimeras 5 muestras:")
    print(df.head())