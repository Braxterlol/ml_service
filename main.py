"""
ML Analysis Service - Entry Point

Servicio de análisis ML para speech therapy.
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Agregar src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from infrastructure.config.settings import settings
from infrastructure.api.ml_analysis_controller import router
from infrastructure.api.dependencies import (
    get_fluency_analyzer,
    get_rhythm_analyzer,
    get_pronunciation_analyzer
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ===== FastAPI App =====

app = FastAPI(
    title="ML Analysis Service",
    description="Servicio de análisis ML para evaluación de habla infantil",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir router
app.include_router(router)


# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar el servicio"""
    logger.info("="*70)
    logger.info("  ML ANALYSIS SERVICE - STARTING")
    logger.info("="*70)
    
    # Pre-cargar modelos
    try:
        logger.info("🔄 Cargando modelos ML...")
        
        # Cargar Random Forest
        fluency = get_fluency_analyzer()
        logger.info(f"   ✅ Random Forest v{fluency.get_version()}")
        
        # Cargar XGBoost
        rhythm = get_rhythm_analyzer()
        logger.info(f"   ✅ XGBoost v{rhythm.get_version()}")
        
        # Cargar Azure (opcional)
        pronunciation = get_pronunciation_analyzer()
        if settings.AZURE_SPEECH_KEY:
            logger.info(f"   ✅ Azure Speech Service configurado")
        else:
            logger.info(f"   ⚠️  Azure Speech Service NO configurado (placeholder activo)")
        
        logger.info("✅ Todos los modelos cargados exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {e}")
        logger.error("   El servicio continuará pero puede no funcionar correctamente")
    
    logger.info(f"\n🚀 Servicio corriendo en: http://0.0.0.0:{settings.SERVICE_PORT}")
    logger.info(f"📚 Docs disponibles en: http://0.0.0.0:{settings.SERVICE_PORT}/docs")
    logger.info("="*70)


@app.on_event("shutdown")
async def shutdown_event():
    """Se ejecuta al apagar el servicio"""
    logger.info("👋 ML Analysis Service apagándose...")


# ===== Root Endpoint =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ml-analysis-service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/v1/ml/health",
            "analyze": "/api/v1/ml/analyze",
            "models_info": "/api/v1/ml/models/info",
            "docs": "/docs"
        }
    }


# ===== Run =====

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )