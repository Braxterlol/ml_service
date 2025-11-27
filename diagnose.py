"""
Script de diagnóstico para ML Analysis Service
"""

import os
from pathlib import Path

def check_models():
    """Verifica que los modelos existan"""
    print("🔍 Verificando modelos ML...")
    print("="*60)
    
    model_files = [
        "models/fluency_rf_model.pkl",
        "models/fluency_scaler.pkl",
        "models/rhythm_xgb_model.pkl",
        "models/rhythm_scaler.pkl"
    ]
    
    all_exist = True
    
    for model_file in model_files:
        path = Path(model_file)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        
        status = "✅" if exists else "❌"
        size_str = f"({size / 1024:.1f} KB)" if exists else "(NO EXISTE)"
        
        print(f"{status} {model_file} {size_str}")
        
        if not exists:
            all_exist = False
    
    print("")
    
    if all_exist:
        print("✅ Todos los modelos están presentes")
    else:
        print("❌ Faltan modelos!")
        print("\n💡 Solución:")
        print("   1. Busca donde entrenaste los modelos:")
        print("      find ~/Desktop/integrador -name 'fluency_rf_model.pkl'")
        print("")
        print("   2. Copia los modelos:")
        print("      cp /ruta/de/origen/models/*.pkl models/")
    
    return all_exist


def check_env():
    """Verifica variables de entorno"""
    print("\n🔍 Verificando variables de entorno...")
    print("="*60)
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ Archivo .env NO existe")
        print("\n💡 Solución:")
        print("   cp .env.example .env")
        return False
    
    print("✅ Archivo .env existe")
    
    # Leer .env
    with open(env_file) as f:
        env_content = f.read()
    
    # Verificar variables importantes
    checks = {
        "INTERNAL_API_KEY": "secret_key_12345" in env_content or "INTERNAL_API_KEY=" in env_content,
        "FLUENCY_MODEL_PATH": "FLUENCY_MODEL_PATH=" in env_content,
        "RHYTHM_MODEL_PATH": "RHYTHM_MODEL_PATH=" in env_content,
        "AZURE_SPEECH_KEY": "AZURE_SPEECH_KEY=" in env_content,
        "AZURE_SPEECH_REGION": "AZURE_SPEECH_REGION=" in env_content
    }
    
    print("\nVariables de entorno:")
    for var, exists in checks.items():
        status = "✅" if exists else "❌"
        print(f"{status} {var}")
    
    return all(checks.values())


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print("\n🔍 Verificando dependencias...")
    print("="*60)
    
    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib"
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Faltan paquetes: {', '.join(missing)}")
        print("\n💡 Solución:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ Todas las dependencias están instaladas")
    return True


def check_directory_structure():
    """Verifica la estructura de directorios"""
    print("\n🔍 Verificando estructura de directorios...")
    print("="*60)
    
    required_dirs = [
        "models",
        "src/",
        "src/domain/models",
        "src/application/services",
        "src/infrastructure/ml_models"
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}")
        
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Ejecuta todos los checks"""
    print("\n" + "="*60)
    print("  ML ANALYSIS SERVICE - DIAGNÓSTICO")
    print("="*60)
    
    checks = {
        "Modelos": check_models(),
        "Variables de Entorno": check_env(),
        "Dependencias": check_dependencies(),
        "Estructura": check_directory_structure()
    }
    
    print("\n" + "="*60)
    print("  RESUMEN")
    print("="*60)
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 ¡Todo está configurado correctamente!")
        print("\n💡 Próximos pasos:")
        print("   1. Ejecuta: python main.py")
        print("   2. Prueba: python test_azure.py")
    else:
        print("\n⚠️  Hay problemas de configuración")
        print("   Revisa los errores arriba y sigue las soluciones sugeridas")
    
    print("")


if __name__ == "__main__":
    main()