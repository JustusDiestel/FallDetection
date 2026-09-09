# Fall Detection

Ein kleines Fall-Erkennungssystem auf Basis von Pose Estimation und einem zeitlichen neuronalen Netz.

Die Anwendung erkennt Personen über YOLO11-Pose, extrahiert deren Keypoints und bewertet anschließend eine Sequenz mehrerer Frames mit einem kleinen 1D-CNN. Ziel ist es, zwischen normalen Bewegungen und möglichen Stürzen zu unterscheiden.

## Funktionsweise

```text
Kamera
  ↓
YOLO11-Pose
  ↓
17 Körper-Keypoints
  ↓
30-Frame-Sequenz
  ↓
Feature Extraction
  ↓
1D-CNN
  ↓
NORMAL / FALL
```

Pro Frame werden die erkannten Körperpunkte verarbeitet und um zusätzliche Bewegungsmerkmale ergänzt, unter anderem:

- normalisierte Keypoint-Positionen
- Hüftposition
- Hüftgeschwindigkeit
- Körperwinkel
- Verhältnis von Körperbreite zu Körperhöhe

Das Modell wertet jeweils die letzten 30 Frames als zusammenhängende Bewegungssequenz aus.

## Web Interface

Zusätzlich gibt es eine einfache FastAPI-Webanwendung.

Der Browser fragt nach Zugriff auf die Kamera und zeigt anschließend:

- den Live-Kamerafeed
- erkannte Körper-Keypoints
- das Pose-Skelett
- den aktuellen Status `NORMAL` oder `FALL`
- die geschätzte Fall-Wahrscheinlichkeit

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO11-Pose
- OpenCV
- NumPy
- FastAPI
- JavaScript
- HTML / CSS

## Datensatz

Für die ersten Experimente wird der UR Fall Detection Dataset verwendet.

Die RGB-Frames werden mit YOLO11-Pose verarbeitet und anschließend als Keypoint-Sequenzen gespeichert. Fehlende Pose-Erkennungen werden bei der Vorverarbeitung behandelt.

Die erzeugten Trainingssamples bestehen aus Sliding Windows mit jeweils 30 Frames.

## Projektstruktur

```text
FallDetection/
├── src/
│   ├── extract_keypoint.py
│   ├── preprocess_keypoints.py
│   ├── create_windows.py
│   ├── fall_dataset.py
│   ├── fall_model.py
│   ├── train.py
│   └── web_app.py
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── fall_classifier.pt
└── yolo11n-pose.pt
```

## Starten

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

Webanwendung starten:

```bash
python -m uvicorn src.web_app:app --reload
```

Anschließend:

```text
http://127.0.0.1:8000
```

im Browser öffnen und den Kamerazugriff erlauben.

## Status

Das Projekt ist aktuell ein Prototyp. Die grundlegende Pipeline von Kamera über Pose Estimation bis zur zeitlichen Fallklassifikation funktioniert, die Erkennungsqualität hängt jedoch noch stark von Trainingsdaten, Kameraperspektive und Umgebung ab.
