# 🚀 Guía de Deployment - ML Analysis Service en AWS EC2

Esta guía te ayudará a desplegar el servicio ML Analysis en una instancia EC2 de AWS.

---

## 📋 Prerequisitos

- Cuenta de AWS activa
- Conocimientos básicos de SSH y terminal
- Azure Speech Service key (para análisis de pronunciación)
- Modelos ML ya entrenados (archivos `.pkl` en `models/`)

---

## 🖥️ Paso 1: Crear Instancia EC2

### 1.1 Configuración de la Instancia

1. **Ir a AWS Console** → EC2 → Launch Instance

2. **Configuración recomendada:**
   - **Nombre:** `ml-analysis-service`
   - **AMI:** Ubuntu Server 22.04 LTS (64-bit x86)
   - **Tipo de instancia:** 
     - Mínimo: `t3.medium` (2 vCPU, 4 GB RAM)
     - Recomendado: `t3.large` (2 vCPU, 8 GB RAM)
   - **Par de claves:** Crea o selecciona un par de claves SSH
   - **Almacenamiento:** 20 GB gp3 (mínimo)

### 1.2 Configurar Security Group

Crea un Security Group con las siguientes reglas:

| Tipo | Protocolo | Puerto | Origen | Descripción |
|------|-----------|--------|--------|-------------|
| SSH | TCP | 22 | Tu IP | Acceso SSH |
| HTTP | TCP | 80 | 0.0.0.0/0 | Nginx (si usas) |
| Custom TCP | TCP | 8002 | 0.0.0.0/0 | ML Service |

**Nota:** Para producción, restringe el acceso según tus necesidades.

### 1.3 Lanzar Instancia

- Haz clic en **Launch Instance**
- Espera a que el estado sea `running`
- Anota la **IP pública** de la instancia

---

## 🔐 Paso 2: Conectarse a la Instancia

```bash
# Dar permisos al archivo de clave
chmod 400 tu-clave.pem

# Conectarse vía SSH
ssh -i tu-clave.pem ubuntu@<IP-PUBLICA>
```

---

## 📦 Paso 3: Preparar el Servidor

### 3.1 Actualizar el Sistema

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 3.2 Instalar Python 3.11

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.13.3 python3.13.3-venv python3.13.3-dev python3-pip
```

### 3.3 Instalar Dependencias del Sistema

```bash
sudo apt-get install -y git nginx supervisor build-essential
```

---

## 📂 Paso 4: Subir el Proyecto al Servidor

### Opción A: Usando Git (Recomendado)

```bash
# En el servidor
cd /home/ubuntu
git clone https://github.com/Braxterlol/ml_service.git ml_analysis_service
cd ml_analysis_service

# Cambiar a la rama correcta
git checkout main
```

### Opción B: Usando SCP

```bash
# En tu máquina local
cd /path/to/ml_analysis_service
tar -czf ml_service.tar.gz .
scp -i tu-clave.pem ml_service.tar.gz ubuntu@<IP-PUBLICA>:/home/ubuntu/

# En el servidor
cd /home/ubuntu
mkdir ml_analysis_service
cd ml_analysis_service
tar -xzf ../ml_service.tar.gz
```

---

## ⚙️ Paso 5: Configurar el Proyecto

### 5.1 Crear Entorno Virtual

```bash
cd /home/ubuntu/ml_analysis_service
python3.11 -m venv venv
source venv/bin/activate
```

### 5.2 Instalar Dependencias Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.3 Crear Archivo `.env`

```bash
nano .env
```

Contenido del archivo `.env`:

```bash
# Azure Speech Service
AZURE_SPEECH_KEY=tu_azure_key_aqui
AZURE_SPEECH_REGION=eastus

# LLM Feedback Service
LLM_FEEDBACK_SERVICE_URL=http://54.157.58.202:8003
LLM_FEEDBACK_TIMEOUT=30
LLM_FEEDBACK_ENABLED=true

# Security
INTERNAL_API_KEY=tu_api_key_segura_aqui

# Service
SERVICE_PORT=8002
DEBUG=false
```

**Guardar:** `Ctrl + O`, Enter, `Ctrl + X`

### 5.4 Verificar Modelos ML

```bash
# Asegúrate de que estos archivos existen
ls -lh models/
# Deberías ver:
# - fluency_rf_model.pkl
# - fluency_scaler.pkl
# - rhythm_xgb_model.pkl
# - rhythm_scaler.pkl
```

---

## 🚀 Paso 6: Configurar Systemd Service

### 6.1 Crear Service File

```bash
sudo nano /etc/systemd/system/ml-analysis.service
```

Contenido:

```ini
[Unit]
Description=ML Analysis Service
After=network.target

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/ml_analysis_service
Environment="PATH=/home/ubuntu/ml_analysis_service/venv/bin"
ExecStart=/home/ubuntu/ml_analysis_service/venv/bin/gunicorn main:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8002 \
    --timeout 120 \
    --access-logfile /var/log/ml-analysis/access.log \
    --error-logfile /var/log/ml-analysis/error.log \
    --log-level info
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always

[Install]
WantedBy=multi-user.target
```

### 6.2 Crear Directorio de Logs

```bash
sudo mkdir -p /var/log/ml-analysis
sudo chown -R ubuntu:ubuntu /var/log/ml-analysis
```

### 6.3 Habilitar e Iniciar el Servicio

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-analysis
sudo systemctl start ml-analysis
```

### 6.4 Verificar Estado

```bash
sudo systemctl status ml-analysis
```

Deberías ver: `Active: active (running)`

---

## 🌐 Paso 7: Configurar Nginx (Opcional pero Recomendado)

### 7.1 Crear Configuración de Nginx

```bash
sudo nano /etc/nginx/sites-available/ml-analysis
```

Pega el contenido del archivo `nginx.conf` incluido en el proyecto.

### 7.2 Habilitar Sitio

```bash
# Crear symlink
sudo ln -s /etc/nginx/sites-available/ml-analysis /etc/nginx/sites-enabled/

# Eliminar configuración por defecto
sudo rm /etc/nginx/sites-enabled/default

# Verificar configuración
sudo nginx -t

# Reiniciar Nginx
sudo systemctl restart nginx
```

---

## ✅ Paso 8: Verificar Deployment

### 8.1 Test Local en el Servidor

```bash
# Verificar que el servicio responde
curl http://localhost:8002/

# Ver logs en tiempo real
sudo journalctl -u ml-analysis -f
```

### 8.2 Test desde tu Máquina

```bash
# Health check
curl http://<IP-PUBLICA>:8002/api/v1/ml/health

# Ver documentación
# Abre en navegador: http://<IP-PUBLICA>:8002/docs
```

---

## 🔧 Comandos Útiles

### Gestión del Servicio

```bash
# Ver logs en tiempo real
sudo journalctl -u ml-analysis -f

# Ver últimas 100 líneas de logs
sudo journalctl -u ml-analysis -n 100

# Reiniciar servicio
sudo systemctl restart ml-analysis

# Detener servicio
sudo systemctl stop ml-analysis

# Iniciar servicio
sudo systemctl start ml-analysis

# Ver estado
sudo systemctl status ml-analysis
```

### Logs de Aplicación

```bash
# Logs de acceso
tail -f /var/log/ml-analysis/access.log

# Logs de error
tail -f /var/log/ml-analysis/error.log
```

### Actualizar Código

```bash
cd /home/ubuntu/ml_analysis_service

# Pull cambios
git pull origin feedback

# Activar venv
source venv/bin/activate

# Instalar nuevas dependencias (si hay)
pip install -r requirements.txt

# Reiniciar servicio
sudo systemctl restart ml-analysis
```

---

## 🔒 Paso 9: Seguridad (Producción)

### 9.1 Configurar Firewall

```bash
# Habilitar UFW
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS (si usas SSL)
sudo ufw allow 8002/tcp  # ML Service (temporal)
sudo ufw enable
```

### 9.2 Configurar SSL con Certbot (Opcional)

```bash
# Instalar Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Obtener certificado (requiere dominio)
sudo certbot --nginx -d tu-dominio.com
```

### 9.3 Generar API Key Segura

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Copia el output y úsalo en .env como INTERNAL_API_KEY
```

---

## 📊 Paso 10: Monitoreo

### 10.1 Verificar Uso de Recursos

```bash
# CPU y RAM
htop

# Espacio en disco
df -h

# Procesos del servicio
ps aux | grep gunicorn
```

### 10.2 Logs de Sistema

```bash
# Logs generales del sistema
sudo tail -f /var/log/syslog

# Logs de Nginx
sudo tail -f /var/log/nginx/error.log
```

---

## 🆘 Troubleshooting

### El servicio no inicia

```bash
# Ver logs detallados
sudo journalctl -u ml-analysis -xe

# Verificar sintaxis del service file
sudo systemd-analyze verify ml-analysis.service

# Verificar permisos
ls -la /home/ubuntu/ml_analysis_service
```

### Error de módulos Python

```bash
cd /home/ubuntu/ml_analysis_service
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
sudo systemctl restart ml-analysis
```

### Error de modelos no encontrados

```bash
# Verificar que los modelos existen
ls -la /home/ubuntu/ml_analysis_service/models/

# Si faltan, súbelos:
scp -i tu-clave.pem -r models/ ubuntu@<IP-PUBLICA>:/home/ubuntu/ml_analysis_service/
```

### Puerto ya en uso

```bash
# Ver qué está usando el puerto 8002
sudo lsof -i :8002

# Matar proceso si es necesario
sudo kill -9 <PID>
```

---

## 📝 Deployment Automatizado

También puedes usar el script `deploy.sh` incluido:

```bash
cd /home/ubuntu/ml_analysis_service
chmod +x deploy.sh
./deploy.sh
```

Este script automatiza los pasos 3-6.

---

## 📞 Endpoints Principales

Una vez desplegado:

- **Root:** `http://<IP-PUBLICA>:8002/`
- **Health:** `http://<IP-PUBLICA>:8002/api/v1/ml/health`
- **Analyze:** `POST http://<IP-PUBLICA>:8002/api/v1/ml/analyze`
- **Models Info:** `http://<IP-PUBLICA>:8002/api/v1/ml/models/info`
- **Docs:** `http://<IP-PUBLICA>:8002/docs`

---

## ✨ ¡Listo!

Tu servicio ML Analysis está ahora corriendo en producción en AWS EC2.

Para cualquier problema, revisa los logs:
```bash
sudo journalctl -u ml-analysis -f
```

---

**Creado por:** Livia Ramos  
**Fecha:** Noviembre 2025  
**Repositorio:** https://github.com/Braxterlol/ml_service

