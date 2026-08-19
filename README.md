# 🚦 Traffic Violation Detector

API que analiza videos y detecta infracciones de tránsito usando YOLOv8 + OpenCV.

## ¿Qué detecta?

### Vehículos
- 🚗 Carros
- 🏍️ Motos
- 🚌 Buses
- 🚚 Camiones

### Señales de Tránsito
- 🔴 Semáforos (determina estado: rojo, amarillo, verde)
- ✋ Señales de PARE (Stop)

### Infracciones
| Tipo | Descripción | Severidad |
|------|-------------|-----------|
| `red_light_violation` | Cruzar semáforo en rojo | Alta |
| `stop_sign_violation` | No detenerse en señal de PARE | Media |
| `potential_speeding` | Velocidad elevada detectada | Media |

## Endpoints

### `POST /analyze`
Sube un video y recibe las infracciones detectadas.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@video.mp4" \
  -F "frame_skip=5" \
  -F "save_evidence=true"
```

**Parámetros:**
- `file`: Archivo de video (mp4, avi, mov, mkv, webm)
- `frame_skip`: Procesar cada N frames (default: 5, menor = más preciso pero más lento)
- `save_evidence`: Guardar frames donde ocurren infracciones (default: false)

**Respuesta:**
```json
{
  "video_info": {
    "filename": "video.mp4",
    "fps": 30.0,
    "total_frames": 900,
    "duration_seconds": 30.0,
    "processed_frames": 180
  },
  "detections_summary": {
    "max_vehicles_in_frame": 5,
    "traffic_light_detections": 60,
    "stop_sign_detections": 3
  },
  "total_violations": 2,
  "violations": [
    {
      "type": "red_light_violation",
      "vehicle_id": 3,
      "vehicle_type": "car",
      "timestamp": 12.5,
      "frame_number": 375,
      "description": "Vehículo (car) cruzó semáforo en rojo",
      "severity": "alta"
    }
  ],
  "summary_by_type": {
    "red_light_violation": {
      "count": 1,
      "violations": [...]
    }
  }
}
```

### `GET /health`
Health check del servicio.

## Instalación local

### Con Docker (recomendado)
```bash
docker-compose up --build
```

### Sin Docker
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment

### Railway
```bash
# Instalar Railway CLI
npm i -g @railway/cli
railway login
railway init
railway up
```

### Render
1. Conectar repo a Render
2. Tipo: Web Service
3. Build: `docker build -t traffic-detector .`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Fly.io
```bash
flyctl launch
flyctl deploy
```

## Configuración

| Variable | Default | Descripción |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Modelo YOLO (n=fast, s=balanced, m=accurate) |
| `CONF_THRESHOLD` | `0.4` | Umbral de confianza de detección (0-1) |
| `PORT` | `8000` | Puerto del servidor |

## Modelos YOLO disponibles

| Modelo | Tamaño | Velocidad | Precisión | Uso |
|--------|--------|-----------|-----------|-----|
| `yolov8n.pt` | 6MB | ⚡⚡⚡ | ⭐⭐ | Prototipo / rápido |
| `yolov8s.pt` | 22MB | ⚡⚡ | ⭐⭐⭐ | Balanceado (recomendado) |
| `yolov8m.pt` | 52MB | ⚡ | ⭐⭐⭐⭐ | Alta precisión |
| `yolov8l.pt` | 89MB | 🐢 | ⭐⭐⭐⭐⭐ | Máima precisión |

## Próximas mejoras
- [ ] Soporte para RTSP / streaming en vivo
- [ ] Detección de más señales (límite de velocidad, no pasar, etc.)
- [ ] Cálculo de velocidad con calibración de cámara
- [ ] Integración con placas vehiculares (OCR)
- [ ] Dashboard web para revisión de infracciones
- [ ] Exportar reportes en PDF
