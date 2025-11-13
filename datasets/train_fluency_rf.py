"""
Entrenamiento de Random Forest para Fluidez

Este script entrena un modelo Random Forest Regressor
para predecir scores de fluidez basados en features acústicos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path: str = 'fluency_training_data.csv'):
    """Carga el dataset de entrenamiento"""
    print(f"Cargando datos desde {file_path}...")
    df = pd.read_csv(file_path)
    
    # Separar features y target
    X = df.drop('fluency_score', axis=1)
    y = df['fluency_score']
    
    print(f"✅ Datos cargados: {len(df)} muestras, {X.shape[1]} features")
    print(f"   Features: {list(X.columns)}")
    
    return X, y, df


def train_model(X_train, y_train, X_test, y_test):
    """
    Entrena el modelo Random Forest con los mejores hiperparámetros.
    """
    print("\n🌲 Entrenando Random Forest...")
    
    # Hiperparámetros optimizados para regresión de scores
    model = RandomForestRegressor(
        n_estimators=150,        # Número de árboles
        max_depth=15,            # Profundidad máxima
        min_samples_split=5,     # Mínimo para split
        min_samples_leaf=2,      # Mínimo en hoja
        max_features='sqrt',     # Features por split
        random_state=42,
        n_jobs=-1,              # Usar todos los cores
        verbose=1
    )
    
    # Entrenar
    model.fit(X_train, y_train)
    
    print("✅ Modelo entrenado")
    
    # Evaluar en train
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    print(f"\n📊 Métricas en TRAIN:")
    print(f"   MAE: {train_mae:.2f}")
    print(f"   R²: {train_r2:.4f}")
    
    # Evaluar en test
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n📊 Métricas en TEST:")
    print(f"   MAE: {test_mae:.2f} puntos")
    print(f"   RMSE: {test_rmse:.2f} puntos")
    print(f"   R²: {test_r2:.4f}")
    
    # Cross-validation
    print("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    print(f"   MAE promedio: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    return model, test_pred


def analyze_feature_importance(model, feature_names):
    """Analiza importancia de features"""
    print("\n🔍 Importancia de Features:")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 features más importantes:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return importances, indices


def plot_results(y_test, test_pred, importances, feature_names, output_dir='plots'):
    """Genera gráficos de resultados"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predicciones vs Real
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_pred, alpha=0.5)
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('Score Real')
    plt.ylabel('Score Predicho')
    plt.title('Random Forest - Fluidez: Predicciones vs Real')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fluency_rf_predictions.png', dpi=300)
    print(f"\n📈 Gráfico guardado: {output_dir}/fluency_rf_predictions.png")
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    indices = np.argsort(importances)[::-1][:10]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title('Top 10 Features Más Importantes')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fluency_rf_importance.png', dpi=300)
    print(f"📈 Gráfico guardado: {output_dir}/fluency_rf_importance.png")
    plt.close()
    
    # 3. Distribución de errores
    errors = test_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Error (Predicho - Real)')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de Errores\nMAE: {np.abs(errors).mean():.2f}')
    plt.axvline(0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fluency_rf_errors.png', dpi=300)
    print(f"📈 Gráfico guardado: {output_dir}/fluency_rf_errors.png")
    plt.close()


def save_model(model, scaler, feature_names, output_dir='models'):
    """Guarda el modelo y metadatos"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = f'{output_dir}/fluency_rf_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Modelo guardado: {model_path}")
    
    # Guardar scaler
    scaler_path = f'{output_dir}/fluency_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler guardado: {scaler_path}")
    
    # Guardar metadatos
    metadata = {
        'model_type': 'RandomForestRegressor',
        'version': 'v1.0.0',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'features': list(feature_names),
        'trained_on': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = f'{output_dir}/fluency_rf_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata guardado: {metadata_path}")


def main():
    """Pipeline completo de entrenamiento"""
    print("="*60)
    print("  ENTRENAMIENTO RANDOM FOREST - FLUIDEZ")
    print("="*60)
    
    # 1. Cargar datos
    X, y, df = load_data('fluency_training_data.csv')
    
    # 2. Split train/test
    print("\n📊 Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    
    # 3. Normalizar features
    print("\n🔄 Normalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✅ Features normalizados con StandardScaler")
    
    # 4. Entrenar modelo
    model, test_pred = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 5. Analizar feature importance
    importances, indices = analyze_feature_importance(model, X.columns)
    
    # 6. Generar gráficos
    plot_results(y_test, test_pred, importances, X.columns)
    
    # 7. Guardar modelo
    save_model(model, scaler, X.columns)
    
    print("\n" + "="*60)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("  📁 models/fluency_rf_model.pkl")
    print("  📁 models/fluency_scaler.pkl")
    print("  📁 models/fluency_rf_metadata.json")
    print("  📁 plots/fluency_rf_*.png")
    
    # Ejemplo de uso
    print("\n💡 Ejemplo de uso del modelo:")
    print("""
    import joblib
    import numpy as np
    
    # Cargar modelo y scaler
    model = joblib.load('models/fluency_rf_model.pkl')
    scaler = joblib.load('models/fluency_scaler.pkl')
    
    # Predecir
    features = np.array([[0.02, 0.08, 3, 250, 60, 0.85, 35, 45, 0.3, 3.5, 4.2]])
    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled)[0]
    
    print(f"Fluency Score: {score:.1f}")
    """)


if __name__ == "__main__":
    main()