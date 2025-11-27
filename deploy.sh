#!/bin/bash

###############################################################################
# ML Analysis Service - Deployment Script para EC2
# Uso: bash deploy.sh
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "  ML Analysis Service - Deployment"
echo "=========================================="

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Variables
PROJECT_DIR="/home/ubuntu/ml_analysis_service"
SERVICE_USER="ubuntu"
SERVICE_PORT=8002

echo -e "${YELLOW}📦 Step 1: Actualizando sistema...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

echo -e "${YELLOW}📦 Step 2: Instalando dependencias del sistema...${NC}"
sudo apt-get install -y \
    python3.13.3 \
    python3.13.3-venv \
    python3-pip \
    git \
    nginx \
    supervisor

echo -e "${YELLOW}📦 Step 3: Creando directorio del proyecto...${NC}"
sudo mkdir -p $PROJECT_DIR
sudo chown -R $SERVICE_USER:$SERVICE_USER $PROJECT_DIR

echo -e "${YELLOW}📦 Step 4: Configurando entorno virtual Python...${NC}"
cd $PROJECT_DIR
python3.13.3 -m venv venv
source venv/bin/activate

echo -e "${YELLOW}📦 Step 5: Instalando dependencias Python...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}📦 Step 6: Verificando archivo .env...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: Archivo .env no encontrado${NC}"
    echo -e "${YELLOW}Por favor crea un archivo .env basado en .env.example${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Step 7: Verificando modelos ML...${NC}"
if [ ! -d "models" ]; then
    echo -e "${RED}❌ Error: Directorio models/ no encontrado${NC}"
    echo -e "${YELLOW}Por favor sube los modelos ML al servidor${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Step 8: Configurando Systemd Service...${NC}"
sudo tee /etc/systemd/system/ml-analysis.service > /dev/null <<EOF
[Unit]
Description=ML Analysis Service
After=network.target

[Service]
Type=notify
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/gunicorn main:app \\
    --workers 2 \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind 0.0.0.0:$SERVICE_PORT \\
    --timeout 120 \\
    --access-logfile /var/log/ml-analysis/access.log \\
    --error-logfile /var/log/ml-analysis/error.log \\
    --log-level info
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo -e "${YELLOW}📦 Step 9: Creando directorio de logs...${NC}"
sudo mkdir -p /var/log/ml-analysis
sudo chown -R $SERVICE_USER:$SERVICE_USER /var/log/ml-analysis

echo -e "${YELLOW}📦 Step 10: Habilitando y iniciando servicio...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable ml-analysis
sudo systemctl restart ml-analysis

echo -e "${YELLOW}📦 Step 11: Verificando estado del servicio...${NC}"
sleep 3
sudo systemctl status ml-analysis --no-pager

echo ""
echo -e "${GREEN}=========================================="
echo "  ✅ Deployment Completado"
echo "==========================================${NC}"
echo ""
echo "Comandos útiles:"
echo "  - Ver logs:        sudo journalctl -u ml-analysis -f"
echo "  - Reiniciar:       sudo systemctl restart ml-analysis"
echo "  - Detener:         sudo systemctl stop ml-analysis"
echo "  - Ver estado:      sudo systemctl status ml-analysis"
echo "  - Ver logs de acceso: tail -f /var/log/ml-analysis/access.log"
echo "  - Ver logs de error:  tail -f /var/log/ml-analysis/error.log"
echo ""
echo "Servicio corriendo en: http://$(curl -s ifconfig.me):$SERVICE_PORT"
echo "Docs disponibles en: http://$(curl -s ifconfig.me):$SERVICE_PORT/docs"
echo ""

