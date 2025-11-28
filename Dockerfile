FROM python:3.12-slim

# Instalar dependencias del sistema necesarias para Azure Speech SDK y ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libasound2 \
    wget \
    ca-certificates \
    libgomp1 \
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

# Exponer el puerto (Railway lo detectará automáticamente)
EXPOSE 8002

# Comando para iniciar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

