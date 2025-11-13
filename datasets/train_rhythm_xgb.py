"""
Entrenamiento de XGBoost para Ritmo (Versión Compatible)

Versión simplificada sin early_stopping para máxima compatibilidad.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path: str = 'rhythm_training_data.csv'):
    """Carga el dataset de entrenamiento"""
    print(f"Cargando datos desde {file_path}...")
    df = pd.read_csv(file_path)
    
    # Separar features y target
    X = df.drop('rhythm_score', axis=1)
    y = df['rhythm_score']
    
    print(f"✅ Datos cargados: {len(df)} muestras, {X.shape[1]} features")
    print(f"   Features: {list(X.columns)}")
    
    return X, y, df


def train_model(X_train, y_train, X_test, y_test):
    """
    Entrena el modelo XGBoost (versión compatible).
    """
    print("\n🚀 Entrenando XGBoost...")
    
    # Versión simplificada sin early_stopping
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenar simple
    print("Entrenando modelo...")
    model.fit(X_train, y_train)
    
    print(f"✅ Modelo entrenado")
    
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
    cv_model = xgb.XGBRegressor(**model.get_params())
    cv_scores = cross_val_score(
        cv_model, X_train, y_train,
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
    
    print("\nTop features más importantes (por gain):")
    for i in range(len(feature_names)):
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
    plt.title('XGBoost - Ritmo: Predicciones vs Real')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rhythm_xgb_predictions.png', dpi=300)
    print(f"\n📈 Gráfico guardado: {output_dir}/rhythm_xgb_predictions.png")
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 7))
    indices = np.argsort(importances)[::-1]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia (Gain)')
    plt.title('Importancia de Features')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rhythm_xgb_importance.png', dpi=300)
    print(f"📈 Gráfico guardado: {output_dir}/rhythm_xgb_importance.png")
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
    plt.savefig(f'{output_dir}/rhythm_xgb_errors.png', dpi=300)
    print(f"📈 Gráfico guardado: {output_dir}/rhythm_xgb_errors.png")
    plt.close()


def save_model(model, scaler, feature_names, output_dir='models'):
    """Guarda el modelo y metadatos"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = f'{output_dir}/rhythm_xgb_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Modelo guardado: {model_path}")
    
    # Guardar scaler
    scaler_path = f'{output_dir}/rhythm_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler guardado: {scaler_path}")
    
    # Guardar metadatos
    metadata = {
        'model_type': 'XGBRegressor',
        'version': 'v1.0.0',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'features': list(feature_names),
        'trained_on': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = f'{output_dir}/rhythm_xgb_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata guardado: {metadata_path}")


def main():
    """Pipeline completo de entrenamiento"""
    print("="*60)
    print("  ENTRENAMIENTO XGBOOST - RITMO")
    print("="*60)
    
    # 1. Cargar datos
    X, y, df = load_data('rhythm_training_data.csv')
    
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
    print("  📁 models/rhythm_xgb_model.pkl")
    print("  📁 models/rhythm_scaler.pkl")
    print("  📁 models/rhythm_xgb_metadata.json")
    print("  📁 plots/rhythm_xgb_*.png")
    
    # Ejemplo de uso
    print("\n💡 Ejemplo de uso del modelo:")
    print("""
    import joblib
    import numpy as np
    
    # Cargar modelo y scaler
    model = joblib.load('models/rhythm_xgb_model.pkl')
    scaler = joblib.load('models/rhythm_scaler.pkl')
    
    # Predecir
    features = np.array([[3.2, 3.8, 3, 1.2, 0.85, 200, 50, 3.5, 0.8]])
    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled)[0]
    
    print(f"Rhythm Score: {score:.1f}")
    """)


if __name__ == "__main__":
    main()