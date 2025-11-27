# 🚀 Quick Deploy - Comandos Rápidos

Esta es una versión resumida para deployment rápido.

## 📋 Pre-requisitos en AWS

1. **EC2 Instance:** Ubuntu 22.04, t3.medium o mayor
2. **Security Group:** Puertos abiertos: 22, 80, 8002
3. **SSH Key:** Descargado y con permisos 400

---

## ⚡ Deployment en 5 Pasos

### 1️⃣ Conectar a EC2

```bash
chmod 400 tu-clave.pem
ssh -i tu-clave.pem ubuntu@<IP-PUBLICA>
```

### 2️⃣ Clonar Proyecto

```bash
cd /home/ubuntu
git clone https://github.com/Braxterlol/ml_service.git ml_analysis_service
cd ml_analysis_service
git checkout feedback
```

### 3️⃣ Crear Archivo .env

```bash
nano .env
```

Pega esto (ajusta los valores):

```bash
AZURE_SPEECH_KEY=tu_azure_key
AZURE_SPEECH_REGION=eastus
LLM_FEEDBACK_SERVICE_URL=http://54.157.58.202:8003
LLM_FEEDBACK_ENABLED=true
INTERNAL_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
SERVICE_PORT=8002
DEBUG=false
```

### 4️⃣ Ejecutar Script de Deploy

```bash
chmod +x deploy.sh
./deploy.sh
```

### 5️⃣ Verificar

```bash
# Ver logs
sudo journalctl -u ml-analysis -f

# Test
curl http://localhost:8002/api/v1/ml/health
```

---

## 🌐 Acceder al Servicio

- **Docs:** `http://<IP-PUBLICA>:8002/docs`
- **Health:** `http://<IP-PUBLICA>:8002/api/v1/ml/health`
- **API:** `http://<IP-PUBLICA>:8002/api/v1/ml/analyze`

---

## 🔧 Comandos Útiles

```bash
# Ver logs en vivo
sudo journalctl -u ml-analysis -f

# Reiniciar servicio
sudo systemctl restart ml-analysis

# Ver estado
sudo systemctl status ml-analysis

# Actualizar código
cd /home/ubuntu/ml_analysis_service
git pull origin feedback
sudo systemctl restart ml-analysis

# Ver recursos
htop
df -h
```

---

## 🆘 Si algo falla

```bash
# Logs detallados
sudo journalctl -u ml-analysis -xe

# Ver qué usa el puerto
sudo lsof -i :8002

# Reiniciar todo
sudo systemctl restart ml-analysis
sudo systemctl restart nginx
```

---

## 📝 Configurar IP Elástica (Recomendado)

1. En AWS Console → EC2 → Elastic IPs
2. Allocate Elastic IP address
3. Associate with your instance
4. Ahora tu IP no cambiará al reiniciar

---

¡Listo! 🎉

