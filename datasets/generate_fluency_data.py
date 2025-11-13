"""
Generador de datos sintéticos para entrenar Random Forest (Fluidez)

Este script genera datos sintéticos basados en reglas expertas
sobre qué características acústicas determinan una buena fluidez.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_fluency_score(features: Dict) -> float:
    """
    Calcula score de fluidez basado en reglas expertas.
    
    Factores principales:
    - Jitter y shimmer (estabilidad vocal): 35%
    - Pausas (cantidad y duración): 30%
    - Speaking ratio: 20%
    - Variabilidad F0 y energía: 15%
    """
    score = 100.0
    
    # Factor 1: Jitter (máx -35 puntos)
    if features['jitter'] > 0.05:
        score -= 35
    elif features['jitter'] > 0.03:
        score -= 20
    elif features['jitter'] > 0.02:
        score -= 10
    
    # Factor 2: Shimmer (máx -35 puntos)
    if features['shimmer'] > 0.13:
        score -= 35
    elif features['shimmer'] > 0.10:
        score -= 20
    elif features['shimmer'] > 0.08:
        score -= 10
    
    # Factor 3: Pausas excesivas (máx -30 puntos)
    if features['pause_count'] > 7:
        score -= 30
    elif features['pause_count'] > 4:
        score -= 15
    elif features['pause_count'] > 2:
        score -= 5
    
    # Factor 4: Pausas muy largas (máx -20 puntos)
    if features['pause_duration_mean'] > 500:
        score -= 20
    elif features['pause_duration_mean'] > 350:
        score -= 10
    elif features['pause_duration_mean'] > 250:
        score -= 5
    
    # Factor 5: Speaking ratio bajo (máx -20 puntos)
    if features['speaking_time_ratio'] < 0.55:
        score -= 20
    elif features['speaking_time_ratio'] < 0.70:
        score -= 10
    elif features['speaking_time_ratio'] < 0.85:
        score -= 5
    
    # Factor 6: F0 muy variable o muy plano (máx -10 puntos)
    if features['f0_std'] > 80 or features['f0_std'] < 15:
        score -= 10
    elif features['f0_std'] > 60 or features['f0_std'] < 20:
        score -= 5
    
    # Factor 7: Energía muy variable (máx -5 puntos)
    if features['energy_std'] > 0.7:
        score -= 5
    elif features['energy_std'] > 0.5:
        score -= 2
    
    # Bonus: Fluidez excepcional (+5 puntos)
    if (features['jitter'] < 0.015 and 
        features['shimmer'] < 0.07 and 
        features['pause_count'] <= 2 and
        features['speaking_time_ratio'] > 0.88):
        score += 5
    
    # Asegurar que esté en rango [0, 100]
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
    
    if min_score >= 90:
        # Excelente fluidez
        jitter = np.random.uniform(0.010, 0.020)
        shimmer = np.random.uniform(0.050, 0.080)
        pause_count = np.random.randint(0, 3)
        pause_duration_mean = np.random.uniform(150, 250)
        speaking_ratio = np.random.uniform(0.85, 0.95)
        f0_std = np.random.uniform(20, 40)
        energy_std = np.random.uniform(0.2, 0.4)
        
    elif min_score >= 70:
        # Buena fluidez
        jitter = np.random.uniform(0.020, 0.030)
        shimmer = np.random.uniform(0.080, 0.100)
        pause_count = np.random.randint(2, 5)
        pause_duration_mean = np.random.uniform(200, 350)
        speaking_ratio = np.random.uniform(0.70, 0.85)
        f0_std = np.random.uniform(30, 60)
        energy_std = np.random.uniform(0.3, 0.5)
        
    elif min_score >= 50:
        # Regular
        jitter = np.random.uniform(0.030, 0.050)
        shimmer = np.random.uniform(0.100, 0.130)
        pause_count = np.random.randint(4, 8)
        pause_duration_mean = np.random.uniform(300, 500)
        speaking_ratio = np.random.uniform(0.55, 0.70)
        f0_std = np.random.uniform(50, 80)
        energy_std = np.random.uniform(0.4, 0.7)
        
    else:
        # Necesita mejorar
        jitter = np.random.uniform(0.050, 0.080)
        shimmer = np.random.uniform(0.130, 0.150)
        pause_count = np.random.randint(7, 11)
        pause_duration_mean = np.random.uniform(400, 600)
        speaking_ratio = np.random.uniform(0.40, 0.55)
        f0_std = np.random.uniform(70, 100)
        energy_std = np.random.uniform(0.6, 0.8)
    
    # Calcular features derivadas
    pause_duration_std = pause_duration_mean * np.random.uniform(0.2, 0.4)
    f0_range = f0_std * np.random.uniform(2.0, 3.5)
    
    # Speech rates (correlacionados con speaking_ratio y pausas)
    base_rate = 3.5 if speaking_ratio > 0.75 else 2.5
    speech_rate_normalized = base_rate + np.random.uniform(-1.0, 1.0)
    articulation_rate_normalized = speech_rate_normalized * np.random.uniform(1.2, 1.5)
    
    features = {
        'jitter': jitter,
        'shimmer': shimmer,
        'pause_count': pause_count,
        'pause_duration_mean': pause_duration_mean,
        'pause_duration_std': pause_duration_std,
        'speaking_time_ratio': speaking_ratio,
        'f0_std': f0_std,
        'f0_range': f0_range,
        'energy_std': energy_std,
        'speech_rate_normalized': speech_rate_normalized,
        'articulation_rate_normalized': articulation_rate_normalized
    }
    
    # Calcular score basado en las reglas
    score = calculate_fluency_score(features)
    
    # Añadir ruido al score (±5 puntos)
    score += np.random.uniform(-5, 5)
    score = max(0, min(100, score))
    
    features['fluency_score'] = score
    
    return features


def generate_fluency_dataset(n_samples: int = 3000, output_file: str = 'fluency_training_data.csv'):
    """
    Genera dataset completo de fluency.
    
    Args:
        n_samples: Número de muestras a generar
        output_file: Archivo de salida CSV
    """
    print(f"Generando {n_samples} muestras sintéticas de fluidez...")
    
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
    print(f"   Target: fluency_score")
    print(f"\nEstadísticas del score:")
    print(df['fluency_score'].describe())
    print(f"\nDistribución:")
    print(f"  Excelente (90-100): {len(df[df['fluency_score'] >= 90])}")
    print(f"  Bueno (70-89): {len(df[(df['fluency_score'] >= 70) & (df['fluency_score'] < 90)])}")
    print(f"  Regular (50-69): {len(df[(df['fluency_score'] >= 50) & (df['fluency_score'] < 70)])}")
    print(f"  Necesita mejorar (0-49): {len(df[df['fluency_score'] < 50])}")
    
    return df


if __name__ == "__main__":
    # Generar dataset
    df = generate_fluency_dataset(n_samples=3000, output_file='fluency_training_data.csv')
    
    # Mostrar primeras filas
    print("\nPrimeras 5 muestras:")
    print(df.head())