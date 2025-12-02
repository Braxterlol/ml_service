import joblib
import numpy as np

# Cargar modelo local
model = joblib.load('models/rhythm_xgb_model.pkl')
scaler = joblib.load('models/rhythm_scaler.pkl')

# Features de tu audio de prueba (con pause_count=0)
features = np.array([[3.0, 3.5, 0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0]])
features_scaled = scaler.transform(features)
score = model.predict(features_scaled)[0]

print(f"Rhythm Score: {score:.1f}")
# Debería dar ~90, no 28