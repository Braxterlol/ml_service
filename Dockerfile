FROM python:3.12

# Instalar dependencias del sistema necesarias para Azure Speech SDK y ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libasound2 \
    libasound2-dev \
    alsa-utils \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero para aprovechar el cache de Docker
COPY requirements.txt .

# Actualizar pip y instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Crear directorio para logs si es necesario
RUN mkdir -p /app/logs

# Variables de entorno para Azure Speech SDK
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/azure/cognitiveservices/speech/lib:$LD_LIBRARY_PATH

# Exponer el puerto (Railway lo detectará automáticamente)
EXPOSE 8002

# Comando para iniciar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

