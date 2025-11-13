"""
Pipeline Completo de Entrenamiento de Modelos ML

Este script ejecuta todo el proceso:
1. Genera datos sintéticos
2. Entrena Random Forest (Fluidez)
3. Entrena XGBoost (Ritmo)
4. Genera reportes y visualizaciones
"""

import subprocess
import sys
import os
from datetime import datetime


def run_command(command, description):
    """Ejecuta un comando y muestra el progreso"""
    print("\n" + "="*70)
    print(f"  {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} - COMPLETADO")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR en {description}")
        print(f"   Código de error: {e.returncode}")
        return False


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print("\n🔍 Verificando dependencias...")
    
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            import_name = package.replace('-', '_')
            if package == 'scikit-learn':
                import_name = 'sklearn'
                __import__(import_name)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NO INSTALADO")
            missing.append(package)
        
    if missing:
        print(f"\n⚠️  Faltan paquetes: {', '.join(missing)}")
        print("\nInstala con:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("\n✅ Todas las dependencias están instaladas")
    return True


def create_directories():
    """Crea directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    dirs = ['models', 'plots', 'datasets']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    return True


def main():
    """Pipeline principal"""
    print("="*70)
    print("  PIPELINE DE ENTRENAMIENTO DE MODELOS ML")
    print("  Fluidez (Random Forest) + Ritmo (XGBoost)")
    print("="*70)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Paso 0: Verificar dependencias
    if not check_dependencies():
        print("\n❌ Por favor instala las dependencias faltantes")
        sys.exit(1)
    
    # Paso 1: Crear directorios
    if not create_directories():
        print("\n❌ Error creando directorios")
        sys.exit(1)
    
    # Paso 2: Generar datos de fluidez
    if not run_command(
        f"{sys.executable} generate_fluency_data.py",
        "PASO 1/4: Generando datos sintéticos de FLUIDEZ"
    ):
        print("\n❌ Error generando datos de fluidez")
        sys.exit(1)
    
    # Paso 3: Generar datos de ritmo
    if not run_command(
        f"{sys.executable} generate_rhythm_data.py",
        "PASO 2/4: Generando datos sintéticos de RITMO"
    ):
        print("\n❌ Error generando datos de ritmo")
        sys.exit(1)
    
    # Paso 4: Entrenar Random Forest
    if not run_command(
        f"{sys.executable} train_fluency_rf.py",
        "PASO 3/4: Entrenando Random Forest (FLUIDEZ)"
    ):
        print("\n❌ Error entrenando Random Forest")
        sys.exit(1)
    
    # Paso 5: Entrenar XGBoost
    if not run_command(
        f"{sys.executable} train_rhythm_xgb.py",
        "PASO 4/4: Entrenando XGBoost (RITMO)"
    ):
        print("\n❌ Error entrenando XGBoost")
        sys.exit(1)
    
    # Resumen final
    print("\n" + "="*70)
    print("  ✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    
    print("\n📦 Modelos entrenados:")
    print("   1. Random Forest (Fluidez)")
    print("      - models/fluency_rf_model.pkl")
    print("      - models/fluency_scaler.pkl")
    print("      - models/fluency_rf_metadata.json")
    
    print("\n   2. XGBoost (Ritmo)")
    print("      - models/rhythm_xgb_model.pkl")
    print("      - models/rhythm_scaler.pkl")
    print("      - models/rhythm_xgb_metadata.json")
    
    print("\n📊 Datasets generados:")
    print("      - fluency_training_data.csv")
    print("      - rhythm_training_data.csv")
    
    print("\n📈 Visualizaciones:")
    print("      - plots/fluency_rf_predictions.png")
    print("      - plots/fluency_rf_importance.png")
    print("      - plots/fluency_rf_errors.png")
    print("      - plots/rhythm_xgb_predictions.png")
    print("      - plots/rhythm_xgb_importance.png")
    print("      - plots/rhythm_xgb_errors.png")
    
    print("\n💡 Próximos pasos:")
    print("   1. Revisar los gráficos en plots/")
    print("   2. Probar los modelos con datos reales de MongoDB")
    print("   3. Integrar en ML Analysis Service")
    print("   4. Configurar Azure Speech Service")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()